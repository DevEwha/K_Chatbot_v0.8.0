#!/usr/bin/env python3
"""
Falcon-specific lossless PPL evaluator.

This wrapper preserves the original stage-transition experiment in
`eval_ppl_lossless.py`:
  - Stage 1: score only A
  - Stage 2: score only newly added B
  - Stage 3: score only newly added C

But it adds Falcon-only safeguards that are still faithful to the original
intent:
  - keep Falcon left-context behavior, with an optional explicit override
  - clear persistent GPU buffers on every `full_recompute` stage transition
    to avoid stale stage-dependent state surviving into the next turn
  - use a separate Falcon log directory so this variant is easy to compare

Usage:
  python eval_ppl_lossless_falcon.py --mode full_recompute ...
  python eval_ppl_lossless_falcon.py --modes full_recompute,naive,surgery ...
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch

import eval_ppl_lossless as base


_FALCON_LEFT_CONTEXT_OVERRIDE: Optional[int] = None
_CLEAR_PERSISTENT_BUFFERS_ON_FULL_RECOMPUTE = True

_ORIG_TRANSITION_STAGE = base.transition_stage
_ORIG_DEFAULT_LEFT_CONTEXT_TOKENS = base.default_left_context_tokens


def _extract_option_value(argv: list[str], option_name: str) -> Optional[str]:
    for idx, arg in enumerate(argv):
        if arg == option_name:
            if idx + 1 < len(argv):
                return argv[idx + 1]
            return None
        prefix = f"{option_name}="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _ensure_falcon_model_only(argv: list[str]) -> None:
    raw_model = _extract_option_value(argv, "--model")
    if raw_model is None:
        return
    canonical = base.canonical_model_name(raw_model)
    if canonical != "falcon-7b":
        raise SystemExit(
            "eval_ppl_lossless_falcon.py only supports Falcon. "
            f"Received --model {raw_model!r} -> canonical {canonical!r}."
        )


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--falcon-left-context-tokens",
        type=int,
        default=None,
        help=(
            "Override Falcon left-context tokens. By default, the wrapper keeps "
            "the base script's Falcon auto-context policy."
        ),
    )
    parser.add_argument(
        "--clear-persistent-buffers-on-full-recompute",
        dest="clear_persistent_buffers_on_full_recompute",
        action="store_true",
        default=True,
        help=(
            "On Falcon full_recompute transitions, also clear persistent GPU "
            "buffers after reset_prefix_cache()+clear_hidden_cache()."
        ),
    )
    parser.add_argument(
        "--no-clear-persistent-buffers-on-full-recompute",
        dest="clear_persistent_buffers_on_full_recompute",
        action="store_false",
        help="Disable the Falcon persistent-buffer clear step on full_recompute.",
    )
    return parser.parse_known_args(argv)


def clear_persistent_buffers(model) -> bool:
    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "clear_persistent_buffers"):
        return False
    inner_model.clear_persistent_buffers()
    return True


def falcon_default_left_context_tokens(
    model_name: str,
    max_model_len: int,
    target_total_tokens: int,
) -> int:
    base_value = _ORIG_DEFAULT_LEFT_CONTEXT_TOKENS(
        model_name=model_name,
        max_model_len=max_model_len,
        target_total_tokens=target_total_tokens,
    )
    if str(model_name).strip().lower() != "falcon-7b":
        return base_value
    if _FALCON_LEFT_CONTEXT_OVERRIDE is None:
        return base_value

    max_allowed = max(0, int(max_model_len) - int(target_total_tokens))
    return max(0, min(int(_FALCON_LEFT_CONTEXT_OVERRIDE), max_allowed))


def falcon_transition_stage(
    llm: base.LLM,
    model,
    mode: str,
    config: dict[str, object],
    target_stage: int,
    transition_context_len: Optional[int] = None,
    surgery_seq_len_override: Optional[int] = None,
) -> dict[str, object]:
    info = _ORIG_TRANSITION_STAGE(
        llm=llm,
        model=model,
        mode=mode,
        config=config,
        target_stage=target_stage,
        transition_context_len=transition_context_len,
        surgery_seq_len_override=surgery_seq_len_override,
    )

    if base.canonical_eval_mode(mode) != "full_recompute":
        return info

    info["clear_persistent_buffers_ok"] = None
    if not _CLEAR_PERSISTENT_BUFFERS_ON_FULL_RECOMPUTE:
        info["persistent_buffer_policy"] = "disabled"
        return info

    cleared = clear_persistent_buffers(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    info["clear_persistent_buffers_ok"] = bool(cleared)
    info["persistent_buffer_policy"] = "clear_persistent_buffers_after_full_recompute"

    if cleared:
        print("  [Policy] Falcon full_recompute -> clear_persistent_buffers()")
    else:
        print("  [Policy] Falcon full_recompute -> clear_persistent_buffers() unavailable")
    return info


def install_wrapper_patches() -> None:
    base.DEFAULT_MODEL = "falcon-7b"
    base.DEFAULT_LOG_DIR = os.path.join(base.SCRIPT_DIR, "results_ppl_lossless_falcon_logs")
    base.default_left_context_tokens = falcon_default_left_context_tokens
    base.transition_stage = falcon_transition_stage


def main() -> None:
    global _FALCON_LEFT_CONTEXT_OVERRIDE
    global _CLEAR_PERSISTENT_BUFFERS_ON_FULL_RECOMPUTE

    wrapper_args, passthrough_argv = parse_wrapper_args(sys.argv[1:])
    _ensure_falcon_model_only(passthrough_argv)

    _FALCON_LEFT_CONTEXT_OVERRIDE = wrapper_args.falcon_left_context_tokens
    _CLEAR_PERSISTENT_BUFFERS_ON_FULL_RECOMPUTE = (
        bool(wrapper_args.clear_persistent_buffers_on_full_recompute)
    )

    install_wrapper_patches()

    sys.argv = [sys.argv[0], *passthrough_argv]
    base.main()


if __name__ == "__main__":
    main()
