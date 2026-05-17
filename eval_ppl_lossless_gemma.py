#!/usr/bin/env python3
"""
Gemma-specific wrapper for `eval_ppl_lossless.py`.

This keeps the original stage-transition experiment:
  - Stage 1: score only A
  - Stage 2: score only newly added B
  - Stage 3: score only newly added C

But it adds Gemma-specific safeguards inspired by
`gemma_prune_lora/eval_ppl_mergedmodel.py`:
  - use an HF tokenizer fallback for eval sample construction
  - prepend a single BOS token to every scored prompt and shift the scored span
    so the first content token is evaluated with Gemma's expected BOS context
  - use a separate Gemma log directory for easier comparison
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import eval_ppl_lossless as base


_GEMMA_BOS_TOKEN_ID: Optional[int] = None
_PRINTED_GEMMA_BOS_POLICY = False

_ORIG_RESOLVE_EVAL_TOKENIZER = base.resolve_eval_tokenizer


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


def _ensure_gemma_model_only(argv: list[str]) -> None:
    raw_model = _extract_option_value(argv, "--model")
    if raw_model is None:
        return
    canonical = base.canonical_model_name(raw_model)
    if canonical != "gemma-7b":
        raise SystemExit(
            "eval_ppl_lossless_gemma.py only supports Gemma. "
            f"Received --model {raw_model!r} -> canonical {canonical!r}."
        )


def _prepend_bos_once(token_ids: list[int]) -> tuple[list[int], int]:
    if _GEMMA_BOS_TOKEN_ID is None:
        return list(token_ids), 0

    ids = [int(token_id) for token_id in token_ids]
    if ids and int(ids[0]) == int(_GEMMA_BOS_TOKEN_ID):
        return ids, 0
    return [int(_GEMMA_BOS_TOKEN_ID), *ids], 1


def gemma_resolve_eval_tokenizer(
    config: dict[str, Any],
    runtime_tokenizer: Any,
) -> tuple[Any, dict[str, Any]]:
    global _GEMMA_BOS_TOKEN_ID

    model_name = str(config.get("canonical_name", "")).strip().lower()
    if model_name != "gemma-7b":
        tokenizer, meta = _ORIG_RESOLVE_EVAL_TOKENIZER(
            config=config,
            runtime_tokenizer=runtime_tokenizer,
        )
        _GEMMA_BOS_TOKEN_ID = getattr(tokenizer, "bos_token_id", None)
        return tokenizer, meta

    model_path = str(config.get("progressive_path", "") or "")
    fallback_paths = base.find_tokenizer_fallbacks(
        model_path=model_path,
        baseline_path=config.get("baseline_path"),
    )

    try:
        eval_tokenizer, resolved_path = base.load_hf_tokenizer_with_fallbacks(
            model_path=model_path,
            fallback_paths=fallback_paths,
        )
        if getattr(eval_tokenizer, "pad_token", None) is None and getattr(
            eval_tokenizer, "eos_token", None
        ) is not None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token

        runtime_probe_match = None
        try:
            probe_text = "Gemma tokenizer alignment probe.\nSecond line."
            runtime_ids = base.tokenize_text_to_ids(runtime_tokenizer, probe_text)
            hf_ids = base.tokenize_text_to_ids(eval_tokenizer, probe_text)
            runtime_probe_match = runtime_ids == hf_ids
        except Exception:
            runtime_probe_match = None

        _GEMMA_BOS_TOKEN_ID = getattr(eval_tokenizer, "bos_token_id", None)

        print(f"  [Tokenizer] Gemma eval tokenizer: {resolved_path}")
        if runtime_probe_match is False:
            print(
                "  [Tokenizer] Gemma runtime tokenizer ids differ from HF tokenizer ids; "
                "using HF tokenizer for eval sample construction."
            )
        elif runtime_probe_match is True:
            print(
                "  [Tokenizer] Gemma runtime tokenizer matches HF tokenizer; "
                "still using HF tokenizer to mirror merged-model PPL evaluation."
            )

        if _GEMMA_BOS_TOKEN_ID is not None:
            print(f"  [Tokenizer] Gemma bos_token_id={int(_GEMMA_BOS_TOKEN_ID)}")
        else:
            print("  [Warn] Gemma tokenizer has no bos_token_id; BOS prefix policy is disabled.")

        return eval_tokenizer, {
            "kind": "hf_auto",
            "path": resolved_path,
            "fallback_paths": fallback_paths,
            "runtime_probe_match": runtime_probe_match,
            "bos_token_id": _GEMMA_BOS_TOKEN_ID,
        }
    except Exception as exc:
        _GEMMA_BOS_TOKEN_ID = getattr(runtime_tokenizer, "bos_token_id", None)
        print(
            "  [Warn] Gemma HF tokenizer load failed; "
            "falling back to the runtime tokenizer."
        )
        print(f"  [Warn] tokenizer_error={exc}")
        if _GEMMA_BOS_TOKEN_ID is not None:
            print(f"  [Tokenizer] Runtime Gemma bos_token_id={int(_GEMMA_BOS_TOKEN_ID)}")
        else:
            print("  [Warn] Runtime Gemma tokenizer has no bos_token_id; BOS prefix policy is disabled.")
        return runtime_tokenizer, {
            "kind": "runtime_fallback",
            "path": model_path or None,
            "fallback_paths": fallback_paths,
            "runtime_probe_match": None,
            "bos_token_id": _GEMMA_BOS_TOKEN_ID,
            "error": str(exc),
        }


def gemma_run_single_sample_eval(
    llm: base.LLM,
    model,
    mode: str,
    config: dict[str, str],
    history_ids: Optional[list[int]],
    a_ids: list[int],
    b_ids: list[int],
    c_ids: list[int],
    strict_logprob_matching: bool,
) -> dict[str, Any]:
    global _PRINTED_GEMMA_BOS_POLICY

    effective_mode = base.canonical_eval_mode(mode)
    history_ids = [int(token_id) for token_id in (history_ids or [])]
    a_ids = [int(token_id) for token_id in a_ids]
    b_ids = [int(token_id) for token_id in b_ids]
    c_ids = [int(token_id) for token_id in c_ids]

    n_history = len(history_ids)
    n_a = len(a_ids)
    n_b = len(b_ids)
    n_c = len(c_ids)

    stage1_prompt_ids, bos_tokens_added = _prepend_bos_once(history_ids + a_ids)
    ab_ids = stage1_prompt_ids + b_ids
    abc_ids = ab_ids + c_ids

    stage1_start = bos_tokens_added + n_history
    stage1_end = stage1_start + n_a
    stage2_start = stage1_end
    stage2_end = stage2_start + n_b
    stage3_start = stage2_end
    stage3_end = stage3_start + n_c

    if bos_tokens_added > 0 and not _PRINTED_GEMMA_BOS_POLICY:
        print(
            "  [Policy] Gemma prompts prepend a single BOS token and shift the "
            "scored spans so the first content token is evaluated with BOS context."
        )
        _PRINTED_GEMMA_BOS_POLICY = True

    sp = base.SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )

    stage1_res = base.compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=stage1_prompt_ids,
        span_start_idx=stage1_start,
        span_end_idx=stage1_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    base.print_turn_result(stage=1, turn=1, span_name="A (warmup span)", result=stage1_res)

    stage2_transition = base.transition_stage(
        llm=llm,
        model=model,
        mode=effective_mode,
        config=config,
        target_stage=2,
        transition_context_len=stage1_end,
        surgery_seq_len_override=stage1_end if effective_mode == "surgery" else None,
    )
    stage2_res = base.compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=ab_ids,
        span_start_idx=stage2_start,
        span_end_idx=stage2_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    base.print_turn_result(stage=2, turn=2, span_name="B (newly added tokens)", result=stage2_res)

    stage3_transition = base.transition_stage(
        llm=llm,
        model=model,
        mode=effective_mode,
        config=config,
        target_stage=3,
        transition_context_len=stage2_end,
        surgery_seq_len_override=stage2_end if effective_mode == "surgery" else None,
    )
    stage3_res = base.compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=abc_ids,
        span_start_idx=stage3_start,
        span_end_idx=stage3_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    base.print_turn_result(stage=3, turn=3, span_name="C (newly added tokens)", result=stage3_res)

    return {
        "chunk_tokens": {
            "bos_prefix": bos_tokens_added,
            "history": n_history,
            "A": n_a,
            "B": n_b,
            "C": n_c,
            "total": n_a + n_b + n_c,
            "prompt_total_with_history": n_history + n_a + n_b + n_c,
            "prompt_total_with_history_and_bos": bos_tokens_added + n_history + n_a + n_b + n_c,
        },
        "transitions": {
            "1_to_2": stage2_transition,
            "2_to_3": stage3_transition,
        },
        "turn_results": {
            "stage1_turn1_A": stage1_res,
            "stage2_turn2_B": stage2_res,
            "stage3_turn3_C": stage3_res,
        },
    }


def install_wrapper_patches() -> None:
    base.DEFAULT_MODEL = "gemma-7b"
    base.DEFAULT_LOG_DIR = os.path.join(base.SCRIPT_DIR, "results_ppl_lossless_gemma_logs")
    base.resolve_eval_tokenizer = gemma_resolve_eval_tokenizer
    base.run_single_sample_eval = gemma_run_single_sample_eval


def main() -> None:
    passthrough_argv = sys.argv[1:]
    _ensure_gemma_model_only(passthrough_argv)
    install_wrapper_patches()
    sys.argv = [sys.argv[0], *passthrough_argv]
    base.main()


if __name__ == "__main__":
    main()
