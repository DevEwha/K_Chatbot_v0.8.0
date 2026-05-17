#!/usr/bin/env python3
"""
eval_ppl_lossless.py
====================

  목적:
    ProgressiveServe Stage 전환(1->2->3)에서 KV cache 처리 방식에 따른
    PPL 변화를 측정한다.

    모드:
      - full_recompute: 전환 직후 reset_prefix_cache()+clear_hidden_cache()
                         로 origin-style full prefill 유도
      - naive         : 전환 직후 cache 무처리(stale KV 재사용)
      - surgery       : 전환 직후 inject_upper_layer_kv(boundary) 수행

평가 방식:
  - Turn 1 (Stage 1): prompt=A
  - Turn 2 (Stage 2): prompt=A+B, B 토큰만 PPL 계산
  - Turn 3 (Stage 3): prompt=A+B+C, C 토큰만 PPL 계산
  - SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)

사용 예시:
  python eval_ppl_lossless.py --model llama2-7b --mode full_recompute
  python eval_ppl_lossless.py --model llama2-7b --mode naive
  python eval_ppl_lossless.py --model llama2-7b --mode surgery
  python eval_ppl_lossless.py --model llama2-7b --modes full_recompute,naive,surgery
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from datetime import datetime
from typing import Any, Optional

# vLLM v0 엔진 강제
os.environ["VLLM_USE_V1"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from shared_model_configs import (
    DEFAULT_MODEL,
    EVAL_PPL_MODEL_CONFIGS as MODEL_CONFIGS,
    MODEL_CHOICES,
    canonical_model_name,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_PROGRESSIVE_IMPLS = ("progressive_serve", "progressive_serve5")
DEFAULT_PROGRESSIVE_IMPL = os.environ.get(
    "ASPLOS_PROGRESSIVE_IMPL",
    "progressive_serve5",
)
if DEFAULT_PROGRESSIVE_IMPL not in SUPPORTED_PROGRESSIVE_IMPLS:
    DEFAULT_PROGRESSIVE_IMPL = "progressive_serve5"
_PROGRESSIVE_MODULES = (
    "progressive_for_causal_lm",
    "progressive_model_dual_path",
    "model_config",
    "universal_bypass_layer",
)

# vLLM v0.8.0 workaround: custom 모델 멀티모달 오인 방지(prefix caching 유지)
import vllm.config

vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)


STAGE_CONFIG = {
    2: {
        "name": "1->2",
        "checkpoint_key": "stage_b_checkpoint",
        "prefetch_fn": "prefetch_stage2",
        "advance_fn": "advance_to_stage2_instant",
        "indices_fn": "_get_b_indices",
    },
    3: {
        "name": "2->3",
        "checkpoint_key": "stage_c_checkpoint",
        "prefetch_fn": "prefetch_stage3",
        "advance_fn": "advance_to_stage3_instant",
        "indices_fn": "_get_c_indices",
    },
}

VALID_MODES = ("full_recompute", "naive", "surgery")
PAPER_REQUIRED_MODES = ("full_recompute", "naive", "surgery")
RESULT_SCHEMA_VERSION = 2
DEFAULT_MIN_PAPER_SAMPLES = 16
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, "results_ppl_lossless_logs")
DEFAULT_FALCON_HF_SANITY_MAX_SAMPLES = 1


def canonical_eval_mode(mode: str) -> str:
    return str(mode).strip().lower()


def stable_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def setup_terminal_log(log_dir: str, tag: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
    log_path = os.path.join(log_dir, f"eval_ppl_lossless_{safe_tag}.log")
    log_file = open(log_path, "w", encoding="utf-8")

    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Tee(stdout, log_file)
    sys.stderr = Tee(stderr, log_file)

    def _close():
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_close)
    return log_path


def resolve_progressive_impl(progressive_impl: str) -> str:
    requested = str(progressive_impl).strip().lower()
    if requested == "auto":
        return DEFAULT_PROGRESSIVE_IMPL
    if requested not in SUPPORTED_PROGRESSIVE_IMPLS:
        raise ValueError(
            f"Unsupported progressive impl: {progressive_impl}. "
            f"Valid: ['auto', {', '.join(repr(x) for x in SUPPORTED_PROGRESSIVE_IMPLS)}]"
        )
    return requested


def load_progressive_model_class(progressive_impl: str):
    impl_root = os.path.abspath(os.path.join(SCRIPT_DIR, progressive_impl))
    if not os.path.isdir(impl_root):
        raise FileNotFoundError(f"Progressive implementation directory not found: {impl_root}")
    if impl_root not in sys.path:
        sys.path.insert(0, impl_root)

    for module_name in _PROGRESSIVE_MODULES:
        cached = sys.modules.get(module_name)
        if cached is None:
            continue
        cached_file = os.path.abspath(str(getattr(cached, "__file__", "")))
        if not cached_file.startswith(impl_root):
            del sys.modules[module_name]

    from progressive_for_causal_lm import ProgressiveForCausalLM  # noqa: E402

    return ProgressiveForCausalLM


# Wikipedia 스타일 장문 텍스트(반복 확장 후 약 1000 토큰으로 자름)
WIKI_BASE_TEXT = """
Computer science is the study of computation, information, and automation. The
discipline emerged from mathematics, electrical engineering, and logic, and it
grew rapidly in the twentieth century as programmable electronic computers became
practical. Early foundations include algorithmic reasoning, formal languages,
and the concept of a stored-program machine. Researchers and engineers then
developed compilers, operating systems, databases, and networking protocols,
which transformed isolated machines into globally connected platforms.

The history of computing hardware includes mechanical calculators, vacuum tube
machines, transistors, integrated circuits, and microprocessors. Each transition
reduced cost and size while increasing reliability and performance. Mainframes
served governments and large enterprises, while minicomputers and personal
computers expanded access to universities, laboratories, and households. Mobile
devices and cloud infrastructure later shifted computation toward distributed
services with massive parallel workloads.

The internet began as a research network connecting institutions that needed
robust communication across heterogeneous systems. Packet switching and layered
protocol design enabled interoperability at global scale. The World Wide Web
added a document model and hyperlink structure that made information publishing
and discovery broadly accessible. Search engines, e-commerce, social media, and
streaming services then became dominant application categories on top of shared
transport and routing standards.

Artificial intelligence has roots in symbolic reasoning, optimization, and
statistics. Machine learning methods improved with larger datasets, specialized
hardware, and better training algorithms. Neural networks, especially deep
architectures based on attention mechanisms, enabled strong performance in
language modeling, translation, vision, and multimodal tasks. Modern systems
balance model quality, latency, safety, and operating cost, and they are often
deployed with caching, batching, and memory management techniques to scale.

Software engineering emphasizes maintainability, correctness, and collaboration.
Version control, testing frameworks, code review, and continuous integration
help teams evolve large codebases without losing reliability. Security practices
such as threat modeling, authentication, encryption, and least privilege reduce
operational risk. Performance work addresses algorithmic complexity, memory
locality, and concurrency behavior across CPUs, GPUs, and networked services.

As computing systems expanded, ethics and governance became central concerns.
Researchers examine privacy, bias, transparency, accountability, and the social
effects of automation. Policy debates include data protection, critical
infrastructure resilience, competition in digital markets, and international
standards for emerging technologies. These questions are now treated as core
engineering constraints rather than optional considerations.
""".strip()


SYSTEMS_TEXT = """
Distributed systems coordinate computation across machines that can fail
independently. Replication improves availability, but maintaining consistency
requires explicit protocol design. Consensus algorithms such as Paxos and Raft
define how a cluster agrees on ordered updates even when messages are delayed,
duplicated, or dropped. Production services often combine leader election with
write-ahead logs, snapshots, and quorum reads to balance correctness and latency.

Large-scale data processing frameworks split jobs into stages and execute them
close to stored data to reduce network overhead. Operators monitor backpressure,
tail latency, and noisy-neighbor effects when colocating workloads. Capacity
planning includes failure domains, maintenance windows, and recovery objectives
that describe acceptable data loss and downtime. Reliability engineering treats
automation, observability, and incident response as first-class product features.
""".strip()


ARCHITECTURE_TEXT = """
Computer architecture studies how instruction sets, pipelines, and memory
hierarchies interact to deliver performance under power constraints. Out-of-order
execution improves throughput by exploiting instruction-level parallelism, while
branch predictors reduce control hazards. Cache coherence protocols keep shared
memory views consistent across cores, but coherence traffic can dominate runtime
for communication-heavy workloads.

Accelerators such as GPUs and TPUs trade control flexibility for data-parallel
throughput. Kernel performance depends on arithmetic intensity, memory coalescing,
and occupancy. Practical optimization requires profiling tools that expose stall
reasons and bandwidth utilization so engineers can separate algorithmic limits
from implementation bottlenecks.
""".strip()


SECURITY_TEXT = """
Modern security practice combines preventive controls with fast detection and
response. Threat models identify assets, attacker capabilities, and trust
boundaries, then map likely abuse paths. Defense-in-depth layers include strong
authentication, least-privilege authorization, encrypted communication, and
secure defaults in deployment pipelines.

Software supply chains introduce risk through transitive dependencies and build
infrastructure. Teams mitigate this with provenance metadata, reproducible builds,
artifact signing, and continuous vulnerability scanning. Incident handling depends
on telemetry quality: without structured logs, audit trails, and rapid rollback
mechanisms, containment becomes slow and expensive.
""".strip()


NETWORKING_TEXT = """
Internet protocols are organized in layers so independently developed systems can
interoperate. Routing decides packet paths between networks, while transport
protocols manage end-to-end delivery behavior. Congestion control adapts sending
rates to available capacity and aims to avoid persistent queue growth. Latency
sensitive applications also care about jitter, packet reordering, and head-of-line
blocking effects.

Datacenter networks increasingly use programmable switches and telemetry streams
to diagnose microbursts and path imbalance. Operators tune queue disciplines,
buffer sizing, and traffic shaping policies to prevent unfairness between flows.
At scale, small control-plane bugs can trigger global instability, so rollout
strategies rely on canaries and staged fault domains.
""".strip()


ML_TEXT = """
Machine learning systems translate statistical models into production pipelines.
Data quality determines an upper bound on model quality, so feature collection,
label consistency, and leakage checks are essential. During training, optimization
choices such as learning-rate schedules, regularization, and batch size affect both
convergence speed and generalization.

Inference serving adds constraints that are less visible in offline experiments:
tail latency budgets, memory fragmentation, and multi-tenant fairness. Teams often
combine quantization, batching, and cache-aware scheduling to sustain throughput.
Evaluation should include robustness and calibration, not only average accuracy, to
avoid brittle behavior on real traffic.
""".strip()


BUILTIN_CORPUS_TEXTS = [
    WIKI_BASE_TEXT,
    SYSTEMS_TEXT,
    ARCHITECTURE_TEXT,
    SECURITY_TEXT,
    NETWORKING_TEXT,
    ML_TEXT,
]


def apply_cachehit_prompt_logprob_patch() -> None:
    """
    vLLM v0 cache-hit + prompt_logprobs 길이 불일치 완화 패치.

    증상:
      cache hit 시 _get_next_prompt_tokens()가 seq_data.get_num_computed_tokens()
      기반으로 next token 범위를 계산하는데, 특정 경로에서 computed_len이
      cached_len보다 1 작아져 (indices=N, tokens=N+1) mismatch가 발생할 수 있음.

    완화:
      computed_len / cached_len 후보 중에서 prompt_logprob row 수와 가장 잘
      맞는 쪽을 선택해, 서버별 off-by-one 차이로 인한 경계 토큰 누락/과다를
      줄인다.
    """
    try:
        import vllm.model_executor.layers.sampler as sampler_mod
    except Exception:
        return

    if getattr(sampler_mod, "_progressiveserve_cachehit_patch_applied", False):
        return

    def _patched_get_next_prompt_tokens(seq_group):
        assert seq_group.is_prompt, (
            "Caller should ensure the sequence group is in a prefill stage."
        )
        seq_ids = seq_group.seq_ids
        query_len = seq_group.query_len
        assert query_len is not None
        assert len(seq_ids) == 1

        seq_data = seq_group.seq_data[seq_ids[0]]
        computed_len = seq_data.get_num_computed_tokens()
        prompt_tokens = seq_data.prompt_token_ids

        def _slice_next_prompt_tokens(base_len: int) -> list[int]:
            next_token_index_start = base_len + 1
            next_token_index_end = min(
                base_len + query_len + 1,
                len(prompt_tokens),
            )
            return prompt_tokens[next_token_index_start:next_token_index_end]

        # Cache-hit path can differ across servers: some runs report
        # `computed_len`, others need `cached_len` to align with
        # `prompt_logprob_indices`. Choose the candidate that best matches the
        # expected prompt-logprob row count instead of always forcing cached_len.
        next_prompt_tokens = _slice_next_prompt_tokens(computed_len)
        expected_len = None
        prompt_indices = getattr(seq_group, "prompt_logprob_indices", None)
        if prompt_indices is not None:
            expected_len = len(prompt_indices)

        if hasattr(seq_data, "get_num_cached_tokens"):
            try:
                cached_len = seq_data.get_num_cached_tokens()
            except Exception:
                cached_len = None
            if cached_len is not None and cached_len != computed_len:
                cached_prompt_tokens = _slice_next_prompt_tokens(cached_len)
                if expected_len is None:
                    if len(cached_prompt_tokens) > len(next_prompt_tokens):
                        next_prompt_tokens = cached_prompt_tokens
                else:
                    computed_diff = abs(len(next_prompt_tokens) - expected_len)
                    cached_diff = abs(len(cached_prompt_tokens) - expected_len)
                    if cached_diff < computed_diff:
                        next_prompt_tokens = cached_prompt_tokens

        return next_prompt_tokens

    sampler_mod._get_next_prompt_tokens = _patched_get_next_prompt_tokens
    sampler_mod._progressiveserve_cachehit_patch_applied = True
    print("  ✅ Applied cache-hit prompt_logprobs patch (_get_next_prompt_tokens)")


def reset_prefix_cache(llm: LLM) -> bool:
    if not hasattr(llm, "reset_prefix_cache"):
        return False
    try:
        return bool(llm.reset_prefix_cache())
    except Exception:
        return False


def clear_hidden_cache(model) -> bool:
    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "clear_hidden_cache"):
        return False
    try:
        inner_model.clear_hidden_cache()
        return True
    except Exception:
        return False


def clear_runtime_state(llm: LLM, model) -> dict[str, bool]:
    prefix_reset_ok = reset_prefix_cache(llm)
    hidden_cache_cleared = clear_hidden_cache(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "reset_prefix_cache_ok": prefix_reset_ok,
        "clear_hidden_cache_ok": hidden_cache_cleared,
    }


def build_dataset_metadata(
    dataset: str,
    dataset_jsonl: Optional[str],
    num_documents: int,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source": dataset,
        "num_documents": int(num_documents),
        "dataset_jsonl_requested": dataset_jsonl,
        "dataset_jsonl_resolved": None,
        "file_size_bytes": None,
        "file_mtime_s": None,
        "provenance_signature": stable_signature(
            {
                "source": dataset,
                "dataset_jsonl_requested": dataset_jsonl,
                "num_documents": int(num_documents),
            }
        ),
    }
    if dataset != "jsonl" or not dataset_jsonl:
        return meta

    resolved = os.path.abspath(dataset_jsonl)
    stat = os.stat(resolved)
    meta.update(
        {
            "dataset_jsonl_resolved": resolved,
            "file_size_bytes": int(stat.st_size),
            "file_mtime_s": float(stat.st_mtime),
            "provenance_signature": stable_signature(
                {
                    "source": dataset,
                    "dataset_jsonl_resolved": resolved,
                    "file_size_bytes": int(stat.st_size),
                    "file_mtime_ns": int(stat.st_mtime_ns),
                    "num_documents": int(num_documents),
                }
            ),
        }
    )
    return meta


def _dedupe_preserve_order(values: list[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_token_ids(encoded: Any) -> Optional[list[int]]:
    if encoded is None:
        return None

    if isinstance(encoded, dict):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        return None

    if torch.is_tensor(ids):
        ids = ids.tolist()
    elif hasattr(ids, "tolist") and not isinstance(ids, list):
        try:
            ids = ids.tolist()
        except Exception:
            pass

    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])

    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids is None:
        return None
    return [int(x) for x in ids]


def tokenize_text_to_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "__call__"):
        try:
            ids = _extract_token_ids(tokenizer(text, add_special_tokens=False))
            if ids is not None:
                return ids
        except Exception:
            pass

    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]

    raise TypeError(
        f"Tokenizer {type(tokenizer).__name__} does not support text -> token ids conversion."
    )


def find_tokenizer_fallbacks(
    model_path: str,
    baseline_path: Optional[str] = None,
) -> list[str]:
    candidates: list[Optional[str]] = []
    if model_path:
        candidates.append(os.path.join(model_path, "original_config"))
        manifest_path = os.path.join(model_path, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                original_cfg_dir = (
                    manifest.get("artifacts", {})
                    .get("original_config", {})
                    .get("dir")
                )
                if original_cfg_dir:
                    candidates.append(str(original_cfg_dir))
                base_model = manifest.get("base_model")
                if base_model:
                    candidates.append(str(base_model))
            except Exception:
                pass

    if baseline_path:
        candidates.append(str(baseline_path))
        candidates.append(os.path.join(str(baseline_path), "original_config"))

    return _dedupe_preserve_order(candidates)


def load_hf_tokenizer_with_fallbacks(
    model_path: str,
    fallback_paths: Optional[list[str]] = None,
) -> tuple[Any, str]:
    if AutoTokenizer is None:
        raise RuntimeError("transformers.AutoTokenizer is unavailable in this environment.")

    candidates = _dedupe_preserve_order([model_path, *(fallback_paths or [])])
    errors: list[tuple[str, str]] = []

    for path in candidates:
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            return tok, resolved
        except Exception as exc:
            errors.append((path, str(exc)))

    for path in candidates:
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(
                resolved,
                use_fast=False,
                trust_remote_code=True,
            )
            return tok, resolved
        except Exception:
            continue

    error_text = "\n".join(f"  - {path}: {err}" for path, err in errors)
    raise RuntimeError(f"Tokenizer load failed for all candidates:\n{error_text}")


def resolve_eval_tokenizer(
    config: dict[str, Any],
    runtime_tokenizer: Any,
) -> tuple[Any, dict[str, Any]]:
    model_name = str(config.get("canonical_name", "")).strip().lower()
    default_meta = {
        "kind": "runtime",
        "path": None,
        "fallback_paths": [],
        "runtime_probe_match": None,
    }
    if model_name != "falcon-7b":
        return runtime_tokenizer, default_meta

    model_path = str(config.get("progressive_path", "") or "")
    fallback_paths = find_tokenizer_fallbacks(
        model_path=model_path,
        baseline_path=config.get("baseline_path"),
    )
    try:
        eval_tokenizer, resolved_path = load_hf_tokenizer_with_fallbacks(
            model_path=model_path,
            fallback_paths=fallback_paths,
        )
        if getattr(eval_tokenizer, "pad_token", None) is None and getattr(
            eval_tokenizer, "eos_token", None
        ) is not None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token

        runtime_probe_match = None
        try:
            probe_text = "Falcon tokenizer alignment probe.\nSecond line."
            runtime_ids = tokenize_text_to_ids(runtime_tokenizer, probe_text)
            hf_ids = tokenize_text_to_ids(eval_tokenizer, probe_text)
            runtime_probe_match = runtime_ids == hf_ids
        except Exception:
            runtime_probe_match = None

        print(f"  [Tokenizer] Falcon eval tokenizer: {resolved_path}")
        if runtime_probe_match is False:
            print(
                "  [Tokenizer] Falcon runtime tokenizer ids differ from HF tokenizer ids; "
                "using HF tokenizer for eval sample construction."
            )
        elif runtime_probe_match is True:
            print(
                "  [Tokenizer] Falcon runtime tokenizer matches HF tokenizer; "
                "still using HF tokenizer to mirror merged-model PPL evaluation."
            )

        return eval_tokenizer, {
            "kind": "hf_auto",
            "path": resolved_path,
            "fallback_paths": fallback_paths,
            "runtime_probe_match": runtime_probe_match,
        }
    except Exception as exc:
        print(
            "  [Warn] Falcon HF tokenizer load failed; "
            "falling back to the runtime tokenizer."
        )
        print(f"  [Warn] tokenizer_error={exc}")
        return runtime_tokenizer, {
            "kind": "runtime_fallback",
            "path": model_path or None,
            "fallback_paths": fallback_paths,
            "runtime_probe_match": None,
            "error": str(exc),
          }


class FalconPassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(
        self,
        hidden_states,
        alibi=None,
        attention_mask=None,
        position_ids=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if not self.return_tuple:
            return hidden_states
        if use_cache:
            return (hidden_states, layer_past)
        return (hidden_states,)


def get_falcon_hf_layers(model) -> nn.ModuleList:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
            return inner.transformer.h
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "transformer") and hasattr(
            base.model.transformer, "h"
        ):
            return base.model.transformer.h
        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            return base.transformer.h
    raise RuntimeError("Cannot find Falcon decoder layers for HF sanity-check.")


def read_falcon_dropped_layers(model_path: str) -> list[int]:
    manifest_path = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return []

    stages = manifest.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])
    if not dropped:
        b_layers = stages.get("B", {}).get("removed_layers", [])
        c_layers = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(b_layers + c_layers))
    if not dropped:
        dropped = manifest.get("simdrop", {}).get("removed_layers", [])
    return sorted(set(int(x) for x in dropped))


def install_falcon_passlayers(model, dropped_indices: list[int], return_tuple: bool = True):
    if not dropped_indices:
        return model
    layers = get_falcon_hf_layers(model)
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            dev = (
                next(old.parameters()).device
                if sum(1 for _ in old.parameters()) > 0
                else torch.device("cpu")
            )
            layers[idx] = FalconPassLayer(return_tuple=return_tuple).to(dev)
            del old
    return model


def load_falcon_stage1_hf_model(model_path: str) -> tuple[Any, dict[str, Any]]:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers.AutoModelForCausalLM is unavailable in this environment.")

    resolved = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                resolved,
                dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

    dropped = read_falcon_dropped_layers(model_path)
    if dropped:
        model = install_falcon_passlayers(model, dropped_indices=dropped, return_tuple=True)

    model = model.to(device)
    model.eval()
    return model, {
        "model_path": resolved,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "dropped_layers": dropped,
    }


@torch.no_grad()
def compute_hf_masked_ppl_from_token_ids(
    model,
    prompt_token_ids: list[int],
    score_start_idx: int,
    score_end_idx: int,
) -> dict[str, Any]:
    if not (0 <= score_start_idx < score_end_idx <= len(prompt_token_ids)):
        raise ValueError(
            f"Invalid HF score span: [{score_start_idx}, {score_end_idx}) "
            f"for len={len(prompt_token_ids)}"
        )

    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    shift_logits = out.logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()

    shift_start = max(int(score_start_idx), 1) - 1
    shift_end = max(int(score_end_idx), 1) - 1
    score_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
    if shift_end > shift_start:
        score_mask[:, shift_start:shift_end] = 1.0

    n_tokens_used = int(score_mask.sum().item())
    if n_tokens_used <= 0:
        raise RuntimeError(
            f"HF sanity-check span has no scoreable tokens: "
            f"score_start={score_start_idx}, score_end={score_end_idx}, "
            f"prompt_len={len(prompt_token_ids)}"
        )

    vocab_size = shift_logits.size(-1)
    loss_tok = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    nll = float((loss_tok * score_mask).sum().item())
    nll_per_token = nll / n_tokens_used
    ppl = math.exp(min(nll_per_token, 100.0))
    return {
        "ppl": ppl,
        "nll": nll,
        "nll_per_token": nll_per_token,
        "n_tokens_used": n_tokens_used,
        "prompt_len": len(prompt_token_ids),
        "score_start_idx": int(score_start_idx),
        "score_end_idx": int(score_end_idx),
    }


def precompute_falcon_hf_stage1_sanity(
    model_name: str,
    config: dict[str, Any],
    eval_samples: list[dict[str, Any]],
    enabled: bool,
    max_samples: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    meta = {
        "enabled": bool(enabled),
        "requested_max_samples": int(max_samples),
        "completed_samples": 0,
        "model_path": None,
        "device": None,
        "dtype": None,
        "dropped_layers": [],
        "error": None,
    }
    if not enabled or str(model_name).strip().lower() != "falcon-7b":
        return {}, meta

    target_count = max(0, min(int(max_samples), len(eval_samples)))
    if target_count == 0:
        return {}, meta

    results: dict[str, dict[str, Any]] = {}
    model = None
    try:
        model, load_meta = load_falcon_stage1_hf_model(str(config["progressive_path"]))
        meta.update(load_meta)

        for sample in eval_samples[:target_count]:
            history_ids = list(sample.get("history_ids") or [])
            a_ids = list(sample["a_ids"])
            prompt_ids = history_ids + a_ids
            span_start_idx = len(history_ids)
            span_end_idx = len(prompt_ids)

            span_metric = compute_hf_masked_ppl_from_token_ids(
                model=model,
                prompt_token_ids=prompt_ids,
                score_start_idx=span_start_idx,
                score_end_idx=span_end_idx,
            )
            full_prompt_metric = compute_hf_masked_ppl_from_token_ids(
                model=model,
                prompt_token_ids=prompt_ids,
                score_start_idx=1,
                score_end_idx=len(prompt_ids),
            )
            results[sample["sample_id"]] = {
                "sample_id": sample["sample_id"],
                "prompt_len": len(prompt_ids),
                "history_tokens": len(history_ids),
                "a_tokens": len(a_ids),
                "same_prompt_span_mask": span_metric,
                "same_prompt_full_prompt": full_prompt_metric,
                "span_minus_full_prompt_nll_per_token": (
                    span_metric["nll_per_token"] - full_prompt_metric["nll_per_token"]
                ),
            }
            meta["completed_samples"] = int(meta["completed_samples"]) + 1
    except Exception as exc:
        meta["error"] = str(exc)
    finally:
        if model is not None:
            try:
                del model
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, meta


def get_model_handle(llm: LLM):
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):
        raise RuntimeError("V1 engine detected. Set VLLM_USE_V1=0.")
    try:
        return engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError("Could not resolve v0 model handle path.") from exc


def register_progressive_model(model_path: str, progressive_model_cls: Any) -> str:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        arch = json.load(f)["architectures"][0]
    ModelRegistry.register_model(arch, progressive_model_cls)
    return arch


def get_kv_block_size(model) -> int:
    # ProgressiveForCausalLM wrapper -> ProgressiveModelDualPath
    try:
        if hasattr(model, "model") and hasattr(model.model, "vllm_config"):
            return int(model.model.vllm_config.cache_config.block_size)
    except Exception:
        pass
    # fallback
    return 16


def get_llm_runtime_config(
    config: dict[str, Any],
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    effective_gpu_memory_utilization = (
        float(gpu_memory_utilization)
        if float(gpu_memory_utilization) > 0
        else float(config.get("gpu_memory_utilization", 0.4))
    )
    return {
        "model": config["progressive_path"],
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "gpu_memory_utilization": effective_gpu_memory_utilization,
        "max_model_len": int(config.get("max_model_len", 2048)),
        "tensor_parallel_size": max(1, int(config.get("tensor_parallel_size", 1))),
        "enforce_eager": False,
        "enable_prefix_caching": bool(config.get("enable_prefix_caching", True)),
        "disable_sliding_window": bool(config.get("disable_sliding_window", False)),
    }


def default_left_context_tokens(
    model_name: str,
    max_model_len: int,
    target_total_tokens: int,
) -> int:
    # Falcon perplexity is especially sensitive to missing left context when we
    # sample a span from the middle of a long document (e.g. PG19). Mirror
    # standard causal-LM evaluation more closely by reusing the preceding tokens
    # as non-scored history whenever the context window allows it.
    if str(model_name).strip().lower() != "falcon-7b":
        return 0
    return max(0, int(max_model_len) - int(target_total_tokens))


def build_abc_chunks(
    tokenizer,
    target_total_tokens: int = 1008,
    block_size: int = 16,
) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if target_total_tokens < block_size * 3:
        raise ValueError(
            f"target_total_tokens must be >= {block_size * 3} for 3 chunks."
        )

    base_ids = tokenize_text_to_ids(tokenizer, WIKI_BASE_TEXT)
    if not base_ids:
        raise RuntimeError("Failed to tokenize WIKI_BASE_TEXT")

    per_chunk_tokens = (target_total_tokens // 3 // block_size) * block_size
    if per_chunk_tokens < block_size:
        raise ValueError(
            f"per_chunk_tokens became too small: {per_chunk_tokens}. "
            f"Increase target_total_tokens or reduce block_size."
        )

    effective_total_tokens = per_chunk_tokens * 3

    full_ids: list[int] = []
    while len(full_ids) < effective_total_tokens:
        full_ids.extend(base_ids)
    full_ids = full_ids[:effective_total_tokens]

    cut1 = per_chunk_tokens
    cut2 = per_chunk_tokens * 2

    a_ids = full_ids[:cut1]
    b_ids = full_ids[cut1:cut2]
    c_ids = full_ids[cut2:]

    ab_ids = a_ids + b_ids
    abc_ids = ab_ids + c_ids

    return {
        "a_ids": a_ids,
        "b_ids": b_ids,
        "c_ids": c_ids,
        "ab_ids": ab_ids,
        "abc_ids": abc_ids,
        "block_size": block_size,
        "per_chunk_tokens": per_chunk_tokens,
        "effective_total_tokens": effective_total_tokens,
    }


def load_jsonl_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            text = ""
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        for key in ("text", "content", "body", "document"):
                            val = obj.get(key)
                            if isinstance(val, str) and val.strip():
                                text = val.strip()
                                break
                except Exception:
                    text = ""
            if not text:
                text = line
            if text:
                texts.append(text)
            if len(texts) == 0 and line_idx > 1000:
                break
    return texts


def load_eval_corpus_texts(dataset: str, dataset_jsonl: Optional[str]) -> list[str]:
    if dataset == "builtin":
        return [x for x in BUILTIN_CORPUS_TEXTS if x.strip()]

    if dataset == "jsonl":
        if not dataset_jsonl:
            raise ValueError("--dataset jsonl requires --dataset-jsonl PATH")
        if not os.path.exists(dataset_jsonl):
            raise FileNotFoundError(f"dataset_jsonl not found: {dataset_jsonl}")
        texts = load_jsonl_texts(dataset_jsonl)
        texts = [x for x in texts if x.strip()]
        if not texts:
            raise RuntimeError(f"No usable texts loaded from {dataset_jsonl}")
        return texts

    raise ValueError(f"Unsupported dataset source: {dataset}")


def build_abc_chunks_from_token_ids(
    token_ids: list[int],
    target_total_tokens: int,
    block_size: int,
    offset: int = 0,
    prefix_context_tokens: int = 0,
) -> dict[str, Any]:
    if not token_ids:
        raise RuntimeError("token_ids is empty")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if target_total_tokens < block_size * 3:
        raise ValueError(
            f"target_total_tokens must be >= {block_size * 3} for 3 chunks."
        )

    per_chunk_tokens = (target_total_tokens // 3 // block_size) * block_size
    if per_chunk_tokens < block_size:
        raise ValueError(
            f"per_chunk_tokens became too small: {per_chunk_tokens}. "
            f"Increase target_total_tokens or reduce block_size."
        )
    effective_total_tokens = per_chunk_tokens * 3

    source_repeated_to_target_length = False
    history_ids: list[int] = []
    if len(token_ids) >= effective_total_tokens:
        max_offset = len(token_ids) - effective_total_tokens
        clamped_offset = max(0, min(offset, max_offset))
        history_budget = max(0, int(prefix_context_tokens))
        if history_budget > 0 and clamped_offset > 0:
            history_start = max(0, clamped_offset - history_budget)
            history_ids = token_ids[history_start:clamped_offset]
        full_ids = token_ids[clamped_offset:clamped_offset + effective_total_tokens]
    else:
        source_repeated_to_target_length = True
        full_ids: list[int] = []
        while len(full_ids) < effective_total_tokens:
            full_ids.extend(token_ids)
        full_ids = full_ids[:effective_total_tokens]
        clamped_offset = 0

    cut1 = per_chunk_tokens
    cut2 = per_chunk_tokens * 2
    a_ids = full_ids[:cut1]
    b_ids = full_ids[cut1:cut2]
    c_ids = full_ids[cut2:]
    ab_ids = a_ids + b_ids
    abc_ids = ab_ids + c_ids

    return {
        "a_ids": a_ids,
        "b_ids": b_ids,
        "c_ids": c_ids,
        "ab_ids": ab_ids,
        "abc_ids": abc_ids,
        "block_size": block_size,
        "per_chunk_tokens": per_chunk_tokens,
        "effective_total_tokens": effective_total_tokens,
        "offset": clamped_offset,
        "history_ids": history_ids,
        "history_tokens": len(history_ids),
        "source_repeated_to_target_length": source_repeated_to_target_length,
    }


def build_eval_samples(
    tokenizer,
    texts: list[str],
    num_samples: int,
    target_total_tokens: int,
    block_size: int,
    seed: int,
    prefix_context_tokens: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    encoded_docs: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        ids = tokenize_text_to_ids(tokenizer, text)
        if len(ids) >= 16:
            encoded_docs.append(
                {
                    "doc_idx": idx,
                    "text": text,
                    "token_ids": ids,
                    "n_tokens": len(ids),
                }
            )

    if not encoded_docs:
        raise RuntimeError("No tokenized documents with enough tokens.")

    samples: list[dict[str, Any]] = []
    for sample_idx in range(num_samples):
        doc = encoded_docs[rng.randrange(len(encoded_docs))]
        ids = doc["token_ids"]
        per_chunk_tokens = (target_total_tokens // 3 // block_size) * block_size
        needed = per_chunk_tokens * 3
        max_offset = max(0, len(ids) - needed)
        offset = rng.randrange(max_offset + 1) if max_offset > 0 else 0

        chunk = build_abc_chunks_from_token_ids(
            token_ids=ids,
            target_total_tokens=target_total_tokens,
            block_size=block_size,
            offset=offset,
            prefix_context_tokens=prefix_context_tokens,
        )
        chunk["sample_id"] = f"s{sample_idx:03d}_doc{doc['doc_idx']}_off{chunk['offset']}"
        chunk["source_doc_idx"] = doc["doc_idx"]
        chunk["source_doc_tokens"] = doc["n_tokens"]
        samples.append(chunk)
    return samples


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95": 0.0,
        }
    n = len(values)
    mean = sum(values) / n
    var = 0.0
    if n > 1:
        var = sum((x - mean) * (x - mean) for x in values) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci95": ci95,
    }


def aggregate_turn(samples: list[dict[str, Any]], turn_key: str) -> dict[str, Any]:
    valid = [
        s for s in samples
        if "error" not in s and "turn_results" in s and turn_key in s["turn_results"]
    ]
    if not valid:
        return {
            "num_samples": len(samples),
            "num_valid_samples": 0,
            "mean_sample_ppl": 0.0,
            "std_sample_ppl": 0.0,
            "ci95_sample_ppl": 0.0,
            "min_sample_ppl": 0.0,
            "max_sample_ppl": 0.0,
            "corpus_nll_per_token": 0.0,
            "corpus_ppl": 0.0,
            "total_used_tokens": 0,
            "total_missing_positions": 0,
            "total_token_not_found": 0,
        }

    ppls = [s["turn_results"][turn_key]["ppl"] for s in valid]
    stats = summarize_numeric(ppls)
    total_nll = sum(s["turn_results"][turn_key]["nll"] for s in valid)
    total_used = sum(s["turn_results"][turn_key]["n_tokens_used"] for s in valid)
    total_missing = sum(s["turn_results"][turn_key]["n_missing_positions"] for s in valid)
    total_not_found = sum(s["turn_results"][turn_key]["n_token_not_found"] for s in valid)
    corpus_nll = total_nll / total_used if total_used > 0 else 0.0
    corpus_ppl = math.exp(corpus_nll) if total_used > 0 else 0.0
    return {
        "num_samples": len(samples),
        "num_valid_samples": len(valid),
        "mean_sample_ppl": stats["mean"],
        "std_sample_ppl": stats["std"],
        "ci95_sample_ppl": stats["ci95"],
        "min_sample_ppl": stats["min"],
        "max_sample_ppl": stats["max"],
        "corpus_nll_per_token": corpus_nll,
        "corpus_ppl": corpus_ppl,
        "total_used_tokens": total_used,
        "total_missing_positions": total_missing,
        "total_token_not_found": total_not_found,
    }


def extract_selected_logprob(
    lp_dict: dict[Any, Any],
    token_id: int,
) -> float | None:
    if token_id in lp_dict:
        return float(lp_dict[token_id].logprob)

    token_id_str = str(token_id)
    if token_id_str in lp_dict:
        return float(lp_dict[token_id_str].logprob)

    return None


def generate_with_token_ids(
    llm: LLM,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> Any:
    # 권장 API: prompts에 token prompt(dict) 전달
    return llm.generate(
        prompts=[{"prompt_token_ids": prompt_token_ids}],
        sampling_params=sampling_params,
      use_tqdm=False,
    )[0]


def probe_single_token_logprob(
    llm: LLM,
    prompt_token_ids: list[int],
    target_position: int,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    if not (0 < target_position < len(prompt_token_ids)):
        raise ValueError(
            f"target_position must satisfy 0 < pos < len(prompt), got pos={target_position}, "
            f"len={len(prompt_token_ids)}"
        )

    probe_prompt_ids = prompt_token_ids[:target_position + 1]

    def _build_sampling_params(prompt_logprobs: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=getattr(sampling_params, "max_tokens", 1),
            prompt_logprobs=prompt_logprobs,
            temperature=getattr(sampling_params, "temperature", 0.0),
        )

    def _force_target_generation_probe() -> dict[str, Any]:
        # Some cache-hit paths omit the first uncached token's prompt_logprobs
        # entry entirely. In that case, recover the exact token logprob by
        # forcing generation of the target token from its prefix.
        prefix_prompt_ids = prompt_token_ids[:target_position]
        target_token_id = prompt_token_ids[target_position]
        output_logprobs = getattr(sampling_params, "logprobs", None)
        forced_sp = SamplingParams(
            max_tokens=1,
            logprobs=1 if output_logprobs is None else max(1, int(output_logprobs)),
            temperature=0.0,
            allowed_token_ids=[int(target_token_id)],
        )
        out_local = generate_with_token_ids(
            llm=llm,
            prompt_token_ids=prefix_prompt_ids,
            sampling_params=forced_sp,
        )
        outputs = getattr(out_local, "outputs", None) or []
        if not outputs:
            raise RuntimeError("Forced-generation probe returned no outputs.")

        completion = outputs[0]
        generated_token_ids = list(getattr(completion, "token_ids", []) or [])
        if not generated_token_ids:
            raise RuntimeError("Forced-generation probe produced no token_ids.")
        if int(generated_token_ids[0]) != int(target_token_id):
            raise RuntimeError(
                "Forced-generation probe sampled an unexpected token: "
                f"got={generated_token_ids[0]} expected={target_token_id}"
            )

        logprob = None
        output_lp = getattr(completion, "logprobs", None)
        if output_lp and len(output_lp) > 0 and output_lp[0]:
            logprob = extract_selected_logprob(output_lp[0], target_token_id)
        if logprob is None:
            cumulative = getattr(completion, "cumulative_logprob", None)
            if cumulative is not None:
                logprob = float(cumulative)
        if logprob is None:
            raise RuntimeError("Forced-generation probe could not recover token logprob.")

        return {
            "target_position": target_position,
            "probe_prompt_len": len(prefix_prompt_ids),
            "logprob": float(logprob),
            "shift": None,
            "perfect_shift_count": None,
            "num_cached_tokens": getattr(out_local, "num_cached_tokens", None),
            "probe_prompt_logprobs": None,
            "repair_method": "forced_generation_allowed_token_ids",
        }

    def _run_probe(sp: SamplingParams) -> tuple[Any, dict[str, Any], int, list[int]]:
        out_local = generate_with_token_ids(
            llm=llm,
            prompt_token_ids=probe_prompt_ids,
            sampling_params=sp,
        )
        plp = out_local.prompt_logprobs
        if plp is None:
            raise RuntimeError("prompt_logprobs is None in single-token probe.")

        out_prompt_ids = out_local.prompt_token_ids or probe_prompt_ids
        probe_target_idx = len(probe_prompt_ids) - 1
        shift_hint = None
        cached = getattr(out_local, "num_cached_tokens", None)
        if cached is not None:
            shift_hint = int(cached) - 1

        shift_info = choose_plp_shift_from_output(
            output_obj=out_local,
            full_ids=out_prompt_ids,
            full_len=len(out_prompt_ids),
            plp_len=len(plp),
            target_positions=[probe_target_idx],
            shift_hint=shift_hint,
        )
        shift = int(shift_info["shift"])
        j = probe_target_idx - shift
        if j < 0 or j >= len(plp) or (j == 0 and probe_target_idx == 0):
            raise RuntimeError(
                f"Single-token probe shift produced invalid prompt_logprobs index: j={j}, "
                f"plp_len={len(plp)}, shift={shift}"
            )

        lp_dict = plp[j]
        if not lp_dict:
            raise RuntimeError("Single-token probe returned empty prompt_logprobs entry.")

        token_id = out_prompt_ids[probe_target_idx]
        lp = extract_selected_logprob(lp_dict, token_id)
        if lp is None:
            raise RuntimeError("Single-token probe could not find target token in prompt_logprobs.")

        return out_local, shift_info, shift, out_prompt_ids

    try:
        out, shift_info, shift, out_prompt_ids = _run_probe(sampling_params)
        probe_prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
    except Exception as first_exc:
        probe_prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
        retry_k = 20 if probe_prompt_logprobs is None or int(probe_prompt_logprobs) < 20 else None
        if retry_k is not None:
            try:
                retry_params = _build_sampling_params(retry_k)
                out, shift_info, shift, out_prompt_ids = _run_probe(retry_params)
                sampling_params = retry_params
                probe_prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
            except Exception:
                return _force_target_generation_probe()
        else:
            return _force_target_generation_probe()

    plp = out.prompt_logprobs
    probe_target_idx = len(probe_prompt_ids) - 1
    token_id = out_prompt_ids[probe_target_idx]
    lp_dict = plp[probe_target_idx - shift]
    lp = extract_selected_logprob(lp_dict, token_id)

    return {
        "target_position": target_position,
        "probe_prompt_len": len(probe_prompt_ids),
        "logprob": float(lp),
        "shift": shift,
        "perfect_shift_count": int(shift_info["perfect_shift_count"]),
        "num_cached_tokens": getattr(out, "num_cached_tokens", None),
        "probe_prompt_logprobs": getattr(sampling_params, "prompt_logprobs", None),
        "repair_method": "prompt_logprobs_probe",
    }


def choose_plp_shift_from_output(
    output_obj: Any,
    full_ids: list[int],
    full_len: int,
    plp_len: int,
    target_positions: list[int],
    shift_hint: Optional[int] = None,
) -> dict[str, Any]:
    # prompt_logprobs length mismatch가 있어도 token-id 매칭률 최대화로 shift 선택.
    cached = getattr(output_obj, "num_cached_tokens", None)
    base = full_len - plp_len
    expected_positions_with_logprob = sum(1 for i in target_positions if i > 0)

    candidates = {
        -1,
        base - 2,
        base - 1,
        base,
        base + 1,
        base + 2,
    }
    if cached is not None:
        c = int(cached) - 1
        candidates.update({c - 2, c - 1, c, c + 1, c + 2})
    if shift_hint is not None:
        candidates.update({shift_hint - 1, shift_hint, shift_hint + 1})

    best_shift = base
    best_found = -1
    best_covered = -1
    best_distance = 10**9
    plp = output_obj.prompt_logprobs
    perfect_shifts: list[int] = []

    for shift in sorted(candidates):
        found = 0
        covered = 0
        for i in target_positions:
            j = i - shift
            # Cache-hit prompt_logprobs may start at index 0 for the first uncached
            # token. Only the absolute first prompt token lacks a logprob by definition.
            if j < 0 or j >= plp_len or (j == 0 and i == 0):
                continue
            covered += 1
            lp_dict = plp[j]
            if not lp_dict:
                continue
            tid = full_ids[i]
            if tid in lp_dict or str(tid) in lp_dict:
                found += 1
        distance = abs(shift - base)
        if (
            found > best_found
            or (found == best_found and covered > best_covered)
            or (found == best_found and covered == best_covered and distance < best_distance)
        ):
            best_shift = shift
            best_found = found
            best_covered = covered
            best_distance = distance
        if (
            expected_positions_with_logprob > 0
            and found == expected_positions_with_logprob
            and covered == expected_positions_with_logprob
        ):
            perfect_shifts.append(shift)

    return {
        "shift": best_shift,
        "best_found": best_found,
        "best_covered": best_covered,
        "best_distance": best_distance,
        "candidate_count": len(candidates),
        "expected_positions_with_logprob": expected_positions_with_logprob,
        "perfect_shift_count": len(perfect_shifts),
        "perfect_shifts": perfect_shifts[:8],
    }


def compute_span_ppl_from_token_ids(
    llm: LLM,
    prompt_token_ids: list[int],
    span_start_idx: int,
    span_end_idx: int,
    sampling_params: SamplingParams,
    strict_logprob_matching: bool = True,
) -> dict[str, Any]:
    if not (0 <= span_start_idx < span_end_idx <= len(prompt_token_ids)):
        raise ValueError(
            f"Invalid span indices: [{span_start_idx}, {span_end_idx}) "
            f"for len={len(prompt_token_ids)}"
        )

    target_positions = list(range(span_start_idx, span_end_idx))

    out = generate_with_token_ids(
        llm=llm,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    plp = out.prompt_logprobs
    if plp is None:
        raise RuntimeError("prompt_logprobs is None. Check vLLM config.")

    out_prompt_ids = out.prompt_token_ids
    full_ids = out_prompt_ids or prompt_token_ids
    plp_len = len(plp)
    full_len = len(full_ids)
    prompt_token_ids_exact_match = True
    if out_prompt_ids is not None:
        prompt_token_ids_exact_match = list(out_prompt_ids) == list(prompt_token_ids)
    shift_hint = None
    cached = getattr(out, "num_cached_tokens", None)
    if cached is not None:
        shift_hint = int(cached) - 1
    shift_info = choose_plp_shift_from_output(
        output_obj=out,
        full_ids=full_ids,
        full_len=full_len,
        plp_len=plp_len,
        target_positions=target_positions,
        shift_hint=shift_hint,
    )
    shift = int(shift_info["shift"])

    nll = 0.0
    used = 0
    missing = 0
    token_not_found = 0
    expected_positions_with_logprob = sum(1 for i in target_positions if i > 0)
    positions_covered_by_plp = 0
    positions_with_nonempty_prompt_logprobs = 0
    repair_candidates: list[dict[str, Any]] = []
    successful_repairs: list[dict[str, Any]] = []
    failed_repairs: list[dict[str, Any]] = []

    for i in target_positions:
        j = i - shift
        if j < 0:
            missing += 1
            repair_candidates.append(
                {
                    "position": i,
                    "reason": "prompt_logprobs_index_before_start",
                }
            )
            continue
        if j == 0 and i == 0:
            continue  # 절대 첫 토큰 logprob 없음
        if j >= plp_len:
            missing += 1
            repair_candidates.append({"position": i, "reason": "prompt_logprobs_index_out_of_range"})
            continue
        positions_covered_by_plp += 1

        lp_dict = plp[j]
        if not lp_dict:
            missing += 1
            repair_candidates.append({"position": i, "reason": "empty_prompt_logprobs_entry"})
            continue
        positions_with_nonempty_prompt_logprobs += 1

        if i >= full_len:
            missing += 1
            repair_candidates.append({"position": i, "reason": "full_prompt_index_out_of_range"})
            continue

        tid = full_ids[i]
        lp = extract_selected_logprob(lp_dict, tid)
        if lp is None:
            token_not_found += 1
            repair_candidates.append({"position": i, "reason": "token_not_found_in_prompt_logprobs"})
            continue

        nll -= lp
        used += 1

    if used == 0:
        raise RuntimeError(
            "No usable token logprobs were collected. "
            f"(full_len={full_len}, plp_len={plp_len}, shift={shift})"
        )

    if strict_logprob_matching and repair_candidates:
        seen_positions: set[int] = set()
        for candidate in repair_candidates:
            position = int(candidate["position"])
            if position in seen_positions:
                continue
            seen_positions.add(position)
            try:
                repair = probe_single_token_logprob(
                    llm=llm,
                    prompt_token_ids=prompt_token_ids,
                    target_position=position,
                    sampling_params=sampling_params,
                )
                nll -= repair["logprob"]
                used += 1
                if candidate["reason"] == "token_not_found_in_prompt_logprobs":
                    token_not_found = max(0, token_not_found - 1)
                else:
                    missing = max(0, missing - 1)
                successful_repairs.append(
                    {
                        "position": position,
                        "reason": candidate["reason"],
                        **repair,
                    }
                )
            except Exception as exc:
                failed_repairs.append(
                    {
                        "position": position,
                        "reason": candidate["reason"],
                        "error": str(exc),
                    }
                )

    if strict_logprob_matching:
        strict_failures: list[str] = []
        if not prompt_token_ids_exact_match:
            strict_failures.append("prompt_token_ids_mismatch")
        if missing != 0:
            strict_failures.append(f"missing_positions={missing}")
        if token_not_found != 0:
            strict_failures.append(f"token_not_found={token_not_found}")
        if used != expected_positions_with_logprob:
            strict_failures.append(
                f"used_tokens={used} expected={expected_positions_with_logprob}"
            )
        unresolved_positions = missing + token_not_found
        if shift_info["best_found"] != expected_positions_with_logprob and unresolved_positions > 0:
            strict_failures.append("best_shift_selected_token_match_incomplete")
        if shift_info["best_covered"] != expected_positions_with_logprob and unresolved_positions > 0:
            strict_failures.append("best_shift_prompt_coverage_incomplete")
        if expected_positions_with_logprob > 0 and unresolved_positions > 0:
            if shift_info["perfect_shift_count"] == 0:
                strict_failures.append("no_perfect_shift_alignment")
            elif shift_info["perfect_shift_count"] > 1:
                strict_failures.append("ambiguous_perfect_shift_alignment")
        if failed_repairs:
            strict_failures.append(f"failed_single_token_repairs={len(failed_repairs)}")
        if strict_failures:
            raise RuntimeError(
                "Strict prompt_logprobs validation failed: "
                + ", ".join(strict_failures)
            )

    nll_per_token = nll / used
    ppl = math.exp(nll_per_token)
    return {
        "ppl": ppl,
        "nll": nll,
        "nll_per_token": nll_per_token,
        "n_tokens_used": used,
        "n_missing_positions": missing,
        "n_token_not_found": token_not_found,
        "n_target_positions": len(target_positions),
        "n_expected_positions_with_logprob": expected_positions_with_logprob,
        "n_positions_covered_by_prompt_logprobs": positions_covered_by_plp,
        "n_positions_with_nonempty_prompt_logprobs": positions_with_nonempty_prompt_logprobs,
        "full_prompt_len": full_len,
        "prompt_logprobs_len": plp_len,
        "plp_position_shift": shift,
        "prompt_token_ids_exact_match": prompt_token_ids_exact_match,
        "strict_logprob_matching": strict_logprob_matching,
        "selected_shift_found": shift_info["best_found"],
        "selected_shift_covered": shift_info["best_covered"],
        "perfect_shift_count": shift_info["perfect_shift_count"],
        "perfect_shifts": shift_info["perfect_shifts"],
        "num_single_token_repairs": len(successful_repairs),
        "num_failed_single_token_repairs": len(failed_repairs),
        "single_token_repairs": successful_repairs,
        "failed_single_token_repairs": failed_repairs,
        "num_cached_tokens": getattr(out, "num_cached_tokens", None),
    }


def transition_stage(
    llm: LLM,
    model,
    mode: str,
    config: dict[str, Any],
    target_stage: int,
    transition_context_len: Optional[int] = None,
    surgery_seq_len_override: Optional[int] = None,
) -> dict[str, Any]:
    if target_stage not in STAGE_CONFIG:
        raise ValueError(f"Unsupported target_stage={target_stage}")
    requested_mode = mode
    mode = canonical_eval_mode(mode)

    stage_cfg = STAGE_CONFIG[target_stage]
    ckpt = config[stage_cfg["checkpoint_key"]]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    prefetch_fn = getattr(model, stage_cfg["prefetch_fn"])
    advance_fn = getattr(model, stage_cfg["advance_fn"])

    print(f"\n[Stage {stage_cfg['name']}] prefetch + activation")
    t0 = time.time()
    prefetch_fn(ckpt)
    ready = model.wait_for_prefetch(timeout_s=120.0)
    if not ready:
        raise RuntimeError(f"Prefetch timed out for stage {stage_cfg['name']}")
    t_prefetch = time.time() - t0

    torch.cuda.synchronize()
    t0 = time.time()
    transitioned = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    if not transitioned:
        raise RuntimeError(f"Instant activation failed for stage {stage_cfg['name']}")
    t_activation = time.time() - t0
    inner_model = getattr(model, "model", None)

    info: dict[str, Any] = {
        "stage": stage_cfg["name"],
        "mode": mode,
        "t_prefetch_s": round(t_prefetch, 6),
        "t_activation_s": round(t_activation, 6),
        "t_cache_sync_s": 0.0,
        "policy_action": None,
        "boundary": None,
        "reconcile_ok": None,
        "reconcile_profile": None,
        "surgery_ok": None,
        "requested_mode": requested_mode,
        "transition_context_len": transition_context_len,
        "surgery_seq_len_override": surgery_seq_len_override,
        "reset_prefix_cache_ok": None,
        "clear_hidden_cache_ok": None,
    }

    if (
        mode == "surgery"
        and transition_context_len is not None
        and transition_context_len > 0
        and inner_model is not None
        and hasattr(inner_model, "sync_persistent_cache")
    ):
        torch.cuda.synchronize()
        t0 = time.time()
        inner_model.sync_persistent_cache(int(transition_context_len))
        torch.cuda.synchronize()
        info["t_cache_sync_s"] = round(time.time() - t0, 6)
        print(
            "  [CacheSync] "
            f"sync_persistent_cache(seq_len={transition_context_len})"
        )

    if mode == "full_recompute":
        info["reset_prefix_cache_ok"] = reset_prefix_cache(llm)
        info["clear_hidden_cache_ok"] = clear_hidden_cache(model)
        torch.cuda.synchronize()
        info["policy_action"] = "reset_prefix_cache+clear_hidden_cache"
        print("  [Policy] full_recompute -> reset_prefix_cache() + clear_hidden_cache()")
        return info

    if mode == "naive":
        info["policy_action"] = "keep_cache_untouched"
        print("  [Policy] naive -> KV cache untouched")
        return info

    if mode == "surgery":
        if inner_model is None or not hasattr(inner_model, "inject_upper_layer_kv"):
            raise RuntimeError(
                "Mode 'surgery' requires a progressive implementation with "
                "inject_upper_layer_kv()."
            )
        indices = getattr(model, stage_cfg["indices_fn"])()
        boundary = model.get_recompute_boundary(indices)
        info["boundary"] = boundary
        if boundary is None:
            raise RuntimeError("Surgery boundary is None.")

        if surgery_seq_len_override is not None:
            surgery_ok = bool(
                inner_model.inject_upper_layer_kv(
                    boundary,
                    seq_len=surgery_seq_len_override,
                )
            )
        else:
            surgery_ok = bool(inner_model.inject_upper_layer_kv(boundary))
        info["reconcile_ok"] = surgery_ok
        info["surgery_ok"] = surgery_ok
        if hasattr(inner_model, "get_last_surgery_profile"):
            try:
                info["reconcile_profile"] = inner_model.get_last_surgery_profile()
            except Exception:
                info["reconcile_profile"] = None
        if surgery_seq_len_override is not None:
            info["policy_action"] = (
                f"inject_upper_layer_kv(boundary={boundary}, "
                f"seq_len={surgery_seq_len_override})"
            )
            print(
                f"  [Policy] surgery -> inject_upper_layer_kv(boundary={boundary}, "
                f"seq_len={surgery_seq_len_override})"
            )
        else:
            info["policy_action"] = f"inject_upper_layer_kv(boundary={boundary})"
            print(f"  [Policy] surgery -> inject_upper_layer_kv(boundary={boundary})")
        if not surgery_ok:
            raise RuntimeError(
                f"inject_upper_layer_kv(boundary={boundary}) failed at stage "
                f"{stage_cfg['name']}."
            )
        return info

    raise ValueError(f"Unsupported mode: {mode}")


def print_turn_result(stage: int, turn: int, span_name: str, result: dict[str, Any]) -> None:
    print(f"\n[Stage {stage} | Turn {turn}] PPL on {span_name}")
    print(
        f"  ppl={result['ppl']:.6f} | nll/tok={result['nll_per_token']:.6f} "
        f"| used={result['n_tokens_used']} | missing={result['n_missing_positions']} "
        f"| token_not_found={result['n_token_not_found']}"
    )
    print(
        f"  prompt_len={result['full_prompt_len']} | plp_len={result['prompt_logprobs_len']} "
        f"| shift={result['plp_position_shift']}"
    )
    print(f"  num_cached_tokens={result['num_cached_tokens']}")


def load_llm_runtime(
    config: dict[str, Any],
    gpu_memory_utilization: float,
) -> tuple[LLM, Any, Any, int, dict[str, Any]]:
    llm_config = get_llm_runtime_config(
        config=config,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    llm = LLM(**llm_config)
    model = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    kv_block_size = get_kv_block_size(model)
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print("  ✅ Persistent GPU buffers cleared")
    clear_runtime_state(llm, model)
    return llm, model, tokenizer, kv_block_size, llm_config


def run_single_sample_eval(
    llm: LLM,
    model,
    mode: str,
    config: dict[str, str],
    history_ids: Optional[list[int]],
    a_ids: list[int],
    b_ids: list[int],
    c_ids: list[int],
    strict_logprob_matching: bool,
) -> dict[str, Any]:
    effective_mode = canonical_eval_mode(mode)
    history_ids = list(history_ids or [])
    n_history = len(history_ids)
    n_a = len(a_ids)
    n_b = len(b_ids)
    n_c = len(c_ids)
    stage1_prompt_ids = history_ids + a_ids
    ab_ids = stage1_prompt_ids + b_ids
    abc_ids = ab_ids + c_ids
    stage1_start = n_history
    stage1_end = stage1_start + n_a
    stage2_start = stage1_end
    stage2_end = stage2_start + n_b
    stage3_start = stage2_end
    stage3_end = stage3_start + n_c

    sp = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )

    stage1_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=stage1_prompt_ids,
        span_start_idx=stage1_start,
        span_end_idx=stage1_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    print_turn_result(stage=1, turn=1, span_name="A (warmup span)", result=stage1_res)

    stage2_transition = transition_stage(
        llm=llm,
        model=model,
        mode=effective_mode,
        config=config,
        target_stage=2,
        transition_context_len=stage1_end,
        surgery_seq_len_override=stage1_end if effective_mode == "surgery" else None,
    )
    stage2_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=ab_ids,
        span_start_idx=stage2_start,
        span_end_idx=stage2_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    print_turn_result(stage=2, turn=2, span_name="B (newly added tokens)", result=stage2_res)

    stage3_transition = transition_stage(
        llm=llm,
        model=model,
        mode=effective_mode,
        config=config,
        target_stage=3,
        transition_context_len=stage2_end,
        surgery_seq_len_override=stage2_end if effective_mode == "surgery" else None,
    )
    stage3_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=abc_ids,
        span_start_idx=stage3_start,
        span_end_idx=stage3_end,
        sampling_params=sp,
        strict_logprob_matching=strict_logprob_matching,
    )
    print_turn_result(stage=3, turn=3, span_name="C (newly added tokens)", result=stage3_res)

    return {
        "chunk_tokens": {
            "history": n_history,
            "A": n_a,
            "B": n_b,
            "C": n_c,
            "total": n_a + n_b + n_c,
            "prompt_total_with_history": n_history + n_a + n_b + n_c,
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


def default_output_path(model: str, mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SCRIPT_DIR, f"results_ppl_lossless_{model}_{mode}_{ts}.json")


def resolve_eval_modes(mode: Optional[str], modes: Optional[str]) -> list[str]:
    if mode and modes:
        raise ValueError("Use either --mode or --modes, not both.")

    if modes:
        parsed = [m.strip() for m in modes.split(",") if m.strip()]
    elif mode:
        parsed = [mode]
    else:
        raise ValueError("Either --mode or --modes must be provided.")

    if not parsed:
        raise ValueError("No modes selected.")

    normalized = [canonical_eval_mode(m) for m in parsed]
    unknown = [m for m in normalized if m not in tuple(canonical_eval_mode(x) for x in VALID_MODES)]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Valid: {list(VALID_MODES)}")

    ordered_unique: list[str] = []
    for m in normalized:
        if m not in ordered_unique:
            ordered_unique.append(m)
    return ordered_unique


def resolve_output_path_for_mode(
    model: str,
    mode: str,
    output: Optional[str],
    multi_mode: bool,
) -> str:
    if output is None:
        return default_output_path(model, mode)
    if not multi_mode:
        return output

    root, ext = os.path.splitext(output)
    if ext:
        return f"{root}_{mode}{ext}"
    return f"{output}_{mode}.json"


def assess_mode_result_paper_readiness(
    result: dict[str, Any],
    min_paper_samples: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    dataset = result.get("dataset", {})
    eval_config = result.get("eval_config", {})
    selected_modes = tuple(result.get("comparison_group", {}).get("selected_modes_in_invocation", []))

    if dataset.get("source") != "jsonl":
        reasons.append("dataset_source_must_be_jsonl")
    if not dataset.get("dataset_jsonl_resolved"):
        reasons.append("dataset_jsonl_resolved_missing")
    if not dataset.get("dataset_provenance_signature"):
        reasons.append("dataset_provenance_signature_missing")
    if not bool(eval_config.get("strict_logprob_matching", False)):
        reasons.append("strict_logprob_matching_disabled")
    if tuple(sorted(selected_modes)) != tuple(sorted(PAPER_REQUIRED_MODES)):
        reasons.append("all_correctness_modes_must_run_together")
    if int(dataset.get("num_samples_failed", 0)) != 0:
        reasons.append("failed_samples_present")
    if int(dataset.get("num_samples_valid", 0)) < int(min_paper_samples):
        reasons.append(
            f"num_samples_valid_below_minimum({dataset.get('num_samples_valid', 0)}<{min_paper_samples})"
        )

    repeated_samples = [
        sample.get("sample_id", f"sample-{idx}")
        for idx, sample in enumerate(result.get("samples", []))
        if "error" not in sample and bool(sample.get("source_repeated_to_target_length", False))
    ]
    if repeated_samples:
        reasons.append("repeated_source_tokens_used_to_reach_target_length")

    for turn_key in ("stage2_turn2_B", "stage3_turn3_C"):
        agg = result.get("aggregate", {}).get(turn_key, {})
        if int(agg.get("total_missing_positions", 0)) != 0:
            reasons.append(f"{turn_key}_has_missing_positions")
        if int(agg.get("total_token_not_found", 0)) != 0:
            reasons.append(f"{turn_key}_has_token_not_found")

    if result.get("mode") == "full_recompute":
        valid_samples = [sample for sample in result.get("samples", []) if "error" not in sample]
        for sample in valid_samples:
            transitions = sample.get("transitions", {})
            for transition_key in ("1_to_2", "2_to_3"):
                transition = transitions.get(transition_key, {})
                if transition.get("policy_action") != "reset_prefix_cache+clear_hidden_cache":
                    reasons.append(f"{transition_key}_policy_not_origin_style_invalidation")
                if not bool(transition.get("clear_hidden_cache_ok", False)):
                    reasons.append(f"{transition_key}_clear_hidden_cache_failed")

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "minimum_valid_samples": int(min_paper_samples),
        "repeated_sample_count": len(repeated_samples),
        "required_modes": list(PAPER_REQUIRED_MODES),
        "strict_logprob_matching_required": True,
    }


def run_mode_evaluation(
    model_name: str,
    mode: str,
    config: dict[str, Any],
    gpu_memory_utilization: float,
    progressive_impl: str,
    stop_on_error: bool,
    eval_samples: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    seed: int,
    num_samples_requested: int,
    target_total_tokens: int,
    kv_block_size: int,
    strict_logprob_matching: bool,
    min_paper_samples: int,
    comparison_group_id: str,
    comparison_signature: str,
    run_started_at: str,
    selected_modes: list[str],
) -> dict[str, Any]:
    print("\n" + "#" * 72)
    print(f"[Mode] {mode}")
    print("#" * 72)

    llm, model, _, mode_kv_block_size, llm_runtime_config = load_llm_runtime(
        config=config,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if mode_kv_block_size != kv_block_size:
        print(
            f"  [Warn] kv_block_size mismatch (dataset={kv_block_size}, runtime={mode_kv_block_size})."
        )

    sample_results: list[dict[str, Any]] = []
    for i, sample in enumerate(eval_samples):
        if i > 0:
            del llm
            del model
            gc.collect()
            torch.cuda.empty_cache()
            llm, model, _, kv_block_size_next, _ = load_llm_runtime(
                config=config,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            if kv_block_size_next != kv_block_size:
                print(
                    f"  [Warn] kv_block_size changed {kv_block_size} -> {kv_block_size_next}. "
                    f"Using {kv_block_size_next} for this sample runtime."
                )

        print("\n" + "-" * 72)
        print(
            f"[Sample {i + 1}/{len(eval_samples)}] id={sample['sample_id']} "
            f"(doc={sample['source_doc_idx']}, off={sample['offset']}, "
            f"doc_tokens={sample['source_doc_tokens']}, "
            f"history={int(sample.get('history_tokens', 0))})"
        )
        print("-" * 72)

        try:
            one = run_single_sample_eval(
                llm=llm,
                model=model,
                mode=mode,
                config=config,
                history_ids=sample.get("history_ids"),
                a_ids=sample["a_ids"],
                b_ids=sample["b_ids"],
                c_ids=sample["c_ids"],
                strict_logprob_matching=strict_logprob_matching,
            )
            row = {
                "sample_index": i,
                "sample_id": sample["sample_id"],
                "source_doc_idx": sample["source_doc_idx"],
                "source_doc_tokens": sample["source_doc_tokens"],
                "offset": sample["offset"],
                "history_tokens": int(sample.get("history_tokens", 0)),
                "source_repeated_to_target_length": bool(
                    sample.get("source_repeated_to_target_length", False)
                ),
                **one,
            }
            sample_results.append(row)
            print(
                f"[Sample {i + 1}] ppl(A/B/C)=("
                f"{row['turn_results']['stage1_turn1_A']['ppl']:.4f}, "
                f"{row['turn_results']['stage2_turn2_B']['ppl']:.4f}, "
                f"{row['turn_results']['stage3_turn3_C']['ppl']:.4f})"
            )
        except Exception as exc:
            row = {
                "sample_index": i,
                "sample_id": sample["sample_id"],
                "source_doc_idx": sample["source_doc_idx"],
                "source_doc_tokens": sample["source_doc_tokens"],
                "offset": sample["offset"],
                "history_tokens": int(sample.get("history_tokens", 0)),
                "error": str(exc),
            }
            sample_results.append(row)
            print(f"[Sample {i + 1}] ERROR: {exc}")
            if stop_on_error:
                raise

    del llm
    del model
    gc.collect()
    torch.cuda.empty_cache()

    agg_stage1 = aggregate_turn(sample_results, "stage1_turn1_A")
    agg_stage2 = aggregate_turn(sample_results, "stage2_turn2_B")
    agg_stage3 = aggregate_turn(sample_results, "stage3_turn3_C")

    print("\n" + "=" * 72)
    print(f"Final PPL Summary (Across Samples) | mode={mode}")
    print(
        "  Stage 1 / Turn 1 / A : "
        f"corpus_ppl={agg_stage1['corpus_ppl']:.6f} | "
        f"mean={agg_stage1['mean_sample_ppl']:.6f} ± {agg_stage1['ci95_sample_ppl']:.6f} (95% CI)"
    )
    print(
        "  Stage 2 / Turn 2 / B : "
        f"corpus_ppl={agg_stage2['corpus_ppl']:.6f} | "
        f"mean={agg_stage2['mean_sample_ppl']:.6f} ± {agg_stage2['ci95_sample_ppl']:.6f} (95% CI)"
    )
    print(
        "  Stage 3 / Turn 3 / C : "
        f"corpus_ppl={agg_stage3['corpus_ppl']:.6f} | "
        f"mean={agg_stage3['mean_sample_ppl']:.6f} ± {agg_stage3['ci95_sample_ppl']:.6f} (95% CI)"
    )
    print("=" * 72)

    valid_samples = [s for s in sample_results if "error" not in s]
    failed_samples = [s for s in sample_results if "error" in s]
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "model": model_name,
        "mode": mode,
        "progressive_impl": progressive_impl,
        "vllm_use_v1": os.environ["VLLM_USE_V1"],
        "created_at": datetime.now().isoformat(),
        "run_metadata": {
            "group_id": comparison_group_id,
            "run_started_at": run_started_at,
            "run_completed_at": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "argv": list(sys.argv),
        },
        "comparison_group": {
            "group_id": comparison_group_id,
            "config_signature": comparison_signature,
            "selected_modes_in_invocation": list(selected_modes),
            "required_modes_for_paper": list(PAPER_REQUIRED_MODES),
        },
        "dataset": {
            "source": dataset_meta["source"],
            "dataset_jsonl": dataset_meta.get("dataset_jsonl_requested"),
            "dataset_jsonl_resolved": dataset_meta.get("dataset_jsonl_resolved"),
            "dataset_provenance_signature": dataset_meta.get("provenance_signature"),
            "file_size_bytes": dataset_meta.get("file_size_bytes"),
            "file_mtime_s": dataset_meta.get("file_mtime_s"),
            "num_documents": dataset_meta["num_documents"],
            "tokenizer": dataset_meta.get("tokenizer"),
            "left_context_tokens": int(dataset_meta.get("left_context_tokens", 0)),
            "seed": seed,
            "num_samples_requested": num_samples_requested,
            "num_samples_valid": len(valid_samples),
            "num_samples_failed": len(failed_samples),
        },
        "llm_config": {
            **llm_runtime_config,
            "sampling": {
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "temperature": 0.0,
            },
        },
        "eval_config": {
            "target_total_tokens": target_total_tokens,
            "gpu_memory_utilization": llm_runtime_config["gpu_memory_utilization"],
            "kv_block_size": mode_kv_block_size,
            "strict_logprob_matching": strict_logprob_matching,
            "minimum_valid_samples_for_paper": int(min_paper_samples),
        },
        "aggregate": {
            "stage1_turn1_A": agg_stage1,
            "stage2_turn2_B": agg_stage2,
            "stage3_turn3_C": agg_stage3,
        },
        "samples": sample_results,
    }

    # Backward-compatible keys when a single sample is valid.
    if len(valid_samples) == 1:
        only = valid_samples[0]
        result["chunk_tokens"] = only["chunk_tokens"]
        result["transitions"] = only["transitions"]
        result["turn_results"] = only["turn_results"]

    result["paper_readiness"] = assess_mode_result_paper_readiness(
        result=result,
        min_paper_samples=min_paper_samples,
    )
    if result["paper_readiness"]["eligible"]:
        print("  [PaperReady] eligible")
    else:
        print(
            "  [PaperReady] not eligible: "
            + ", ".join(result["paper_readiness"]["reasons"])
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lossless PPL evaluation for stage transitions "
            "(full_recompute vs naive vs surgery)."
        )
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=list(VALID_MODES),
        default=None,
        help="Single mode run.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated multi-mode run. Example: full_recompute,naive,surgery",
    )
    parser.add_argument("--target-total-tokens", type=int, default=1008)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.0,
        help="Override gpu_memory_utilization. Use <=0 to follow model config.",
    )
    parser.add_argument(
        "--progressive-impl",
        choices=["auto", *SUPPORTED_PROGRESSIVE_IMPLS],
        default="auto",
        help=(
            "Progressive implementation directory to use. "
            f"auto: ASPLOS_PROGRESSIVE_IMPL/{DEFAULT_PROGRESSIVE_IMPL}."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--dataset",
        choices=["builtin", "jsonl"],
        default="builtin",
        help="builtin: script 내장 코퍼스, jsonl: 외부 문서 집합",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        default=None,
        help="--dataset jsonl일 때 사용할 JSONL/TXT 경로",
    )
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument(
        "--strict-logprob-matching",
        dest="strict_logprob_matching",
        action="store_true",
        default=True,
        help="Fail a sample when any prompt_logprob position is missing, ambiguous, or token-not-found.",
    )
    parser.add_argument(
        "--allow-logprob-gaps",
        dest="strict_logprob_matching",
        action="store_false",
        help="Developer-only mode: keep running even when prompt_logprob coverage is incomplete.",
    )
    parser.add_argument(
        "--min-paper-samples",
        type=int,
        default=DEFAULT_MIN_PAPER_SAMPLES,
        help="Minimum number of valid samples required for paper-ready eligibility.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    log_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_terminal_log(DEFAULT_LOG_DIR, log_tag)

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    eval_modes = resolve_eval_modes(mode=args.mode, modes=args.modes)
    progressive_impl = resolve_progressive_impl(args.progressive_impl)

    print("=" * 72)
    print("ProgressiveServe Stage-Transition PPL Evaluation (Lossless Check)")
    print(
        f"  model={args.model} | modes={eval_modes} "
        f"| progressive_impl={progressive_impl} "
        f"| VLLM_USE_V1={os.environ['VLLM_USE_V1']}"
    )
    print(f"  gpu={torch.cuda.get_device_name(0)}")
    print(f"  log_path={log_path}")
    print("=" * 72)

    apply_cachehit_prompt_logprob_patch()

    model_name = canonical_model_name(args.model)
    config = {**MODEL_CONFIGS[model_name], "canonical_name": model_name}
    model_path = config["progressive_path"]
    progressive_model_cls = load_progressive_model_class(progressive_impl)
    arch = register_progressive_model(model_path, progressive_model_cls)
    print(f"  Registered ProgressiveForCausalLM as: {arch}")

    llm, model, runtime_tokenizer, kv_block_size, initial_llm_runtime_config = load_llm_runtime(
        config=config,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer, tokenizer_meta = resolve_eval_tokenizer(
        config=config,
        runtime_tokenizer=runtime_tokenizer,
    )

    texts = load_eval_corpus_texts(args.dataset, args.dataset_jsonl)
    dataset_meta = build_dataset_metadata(
        dataset=args.dataset,
        dataset_jsonl=args.dataset_jsonl,
        num_documents=len(texts),
    )
    left_context_tokens = default_left_context_tokens(
        model_name=model_name,
        max_model_len=int(initial_llm_runtime_config["max_model_len"]),
        target_total_tokens=int(args.target_total_tokens),
    )
    dataset_meta["tokenizer"] = tokenizer_meta
    dataset_meta["left_context_tokens"] = int(left_context_tokens)
    eval_samples = build_eval_samples(
        tokenizer=tokenizer,
        texts=texts,
        num_samples=args.num_samples,
        target_total_tokens=args.target_total_tokens,
        block_size=kv_block_size,
        seed=args.seed,
        prefix_context_tokens=left_context_tokens,
    )

    print(
        f"\n[Dataset] source={args.dataset} | docs={len(texts)} | "
        f"samples={len(eval_samples)} | seed={args.seed}"
    )
    print(
        f"[Data] block_size={kv_block_size} | target_total_tokens={args.target_total_tokens} "
        f"(effective total is block-aligned per sample)"
    )
    if left_context_tokens > 0:
        print(
            f"[Data] left_context_tokens={left_context_tokens} "
            "(document prefix context is prepended and excluded from loss)"
        )
    if any(bool(sample.get("source_repeated_to_target_length", False)) for sample in eval_samples):
        print(
            "[Warn] Some sampled documents were shorter than the requested token budget "
            "and were repeated to reach target length. These runs will not be paper-ready."
        )

    del llm
    del model
    gc.collect()
    torch.cuda.empty_cache()

    multi_mode = len(eval_modes) > 1
    run_started_at = datetime.now().isoformat()
    comparison_signature = stable_signature(
        {
            "model": model_name,
            "progressive_impl": progressive_impl,
            "dataset_provenance_signature": dataset_meta["provenance_signature"],
            "seed": args.seed,
            "num_samples_requested": args.num_samples,
            "target_total_tokens": args.target_total_tokens,
            "kv_block_size": kv_block_size,
            "strict_logprob_matching": bool(args.strict_logprob_matching),
            "min_paper_samples": int(args.min_paper_samples),
            "selected_modes": sorted(eval_modes),
        }
    )
    run_group_id = (
        f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{comparison_signature[:12]}"
    )
    results_by_mode: dict[str, dict[str, Any]] = {}
    for mode in eval_modes:
        result = run_mode_evaluation(
            model_name=model_name,
            mode=mode,
            config=config,
            gpu_memory_utilization=args.gpu_memory_utilization,
            progressive_impl=progressive_impl,
            stop_on_error=args.stop_on_error,
            eval_samples=eval_samples,
            dataset_meta=dataset_meta,
            seed=args.seed,
            num_samples_requested=args.num_samples,
            target_total_tokens=args.target_total_tokens,
            kv_block_size=kv_block_size,
            strict_logprob_matching=bool(args.strict_logprob_matching),
            min_paper_samples=int(args.min_paper_samples),
            comparison_group_id=run_group_id,
            comparison_signature=comparison_signature,
            run_started_at=run_started_at,
            selected_modes=eval_modes,
        )
        results_by_mode[mode] = result
        out_path = resolve_output_path_for_mode(
            model=model_name,
            mode=mode,
            output=args.output,
            multi_mode=multi_mode,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved result JSON ({mode}): {out_path}")

    if multi_mode:
        print("\n" + "=" * 72)
        print("Multi-Mode Comparison (corpus_ppl)")
        print(f"{'Mode':<16} {'Stage1(A)':>12} {'Stage2(B)':>12} {'Stage3(C)':>12}")
        print("-" * 72)
        for mode in eval_modes:
            agg = results_by_mode[mode]["aggregate"]
            print(
                f"{mode:<16} "
                f"{agg['stage1_turn1_A']['corpus_ppl']:>12.6f} "
                f"{agg['stage2_turn2_B']['corpus_ppl']:>12.6f} "
                f"{agg['stage3_turn3_C']['corpus_ppl']:>12.6f}"
            )
        print("=" * 72)

        reference_mode = (
            "full_recompute" if "full_recompute" in results_by_mode else eval_modes[0]
        )
        ref_agg = results_by_mode[reference_mode]["aggregate"]

        def _pct_delta(curr: float, base: float) -> float:
            if base == 0.0:
                return 0.0
            return ((curr - base) / base) * 100.0

        print("\n" + "=" * 72)
        print(f"Relative Comparison vs {reference_mode} (corpus_ppl)")
        print(f"{'Mode':<16} {'Stage1(A)':>12} {'Stage2(B)':>12} {'Stage3(C)':>12}")
        print("-" * 72)
        for mode in eval_modes:
            agg = results_by_mode[mode]["aggregate"]
            d1 = _pct_delta(
                agg["stage1_turn1_A"]["corpus_ppl"],
                ref_agg["stage1_turn1_A"]["corpus_ppl"],
            )
            d2 = _pct_delta(
                agg["stage2_turn2_B"]["corpus_ppl"],
                ref_agg["stage2_turn2_B"]["corpus_ppl"],
            )
            d3 = _pct_delta(
                agg["stage3_turn3_C"]["corpus_ppl"],
                ref_agg["stage3_turn3_C"]["corpus_ppl"],
            )
            print(
                f"{mode:<16} "
                f"{d1:+11.2f}% "
                f"{d2:+11.2f}% "
                f"{d3:+11.2f}%"
            )
        print("=" * 72)


if __name__ == "__main__":
    main()
