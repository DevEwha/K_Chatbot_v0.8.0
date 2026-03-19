#!/usr/bin/env python3
"""
eval_ppl_lossless.py
====================

목적:
  ProgressiveServe Stage 전환(1->2->3)에서 KV cache 처리 방식에 따른
  PPL 변화를 측정한다.

모드:
  - full_recompute: 전환 직후 reset_prefix_cache()로 full prefill 유도
  - naive         : 전환 직후 cache 무처리(stale KV 재사용)
  - skbi          : 전환 직후 apply_skbi(boundary) 수행

평가 방식:
  - Turn 1 (Stage 1): prompt=A
  - Turn 2 (Stage 2): prompt=A+B, B 토큰만 PPL 계산
  - Turn 3 (Stage 3): prompt=A+B+C, C 토큰만 PPL 계산
  - SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)

사용 예시:
  python eval_ppl_lossless.py --model llama --mode full_recompute
  python eval_ppl_lossless.py --model llama --mode naive
  python eval_ppl_lossless.py --model llama --mode skbi
  python eval_ppl_lossless.py --model llama --modes full_recompute,naive,skbi
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Optional

# vLLM v0 엔진 강제
os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# vLLM v0.8.0 workaround: custom 모델 멀티모달 오인 방지(prefix caching 유지)
import vllm.config

vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)

# Progressive model
sys.path.insert(0, os.path.join(SCRIPT_DIR, "progressive_serve"))
from progressive_for_causal_lm import ProgressiveForCausalLM  # noqa: E402


MODELS = {
    "llama": {
        "progressive_path": "/acpl-ssd30/7b_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    },
    "mistral": {
        "progressive_path": "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    },
}


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

VALID_MODES = ("full_recompute", "naive", "skbi")


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
      computed_len < cached_len 이면 cached_len으로 보정.
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

        if hasattr(seq_data, "get_num_cached_tokens"):
            try:
                cached_len = seq_data.get_num_cached_tokens()
                if cached_len > computed_len:
                    computed_len = cached_len
            except Exception:
                pass

        prompt_tokens = seq_data.prompt_token_ids
        next_token_index_start = computed_len + 1
        next_token_index_end = min(
            computed_len + query_len + 1,
            len(prompt_tokens),
        )
        return prompt_tokens[next_token_index_start:next_token_index_end]

    sampler_mod._get_next_prompt_tokens = _patched_get_next_prompt_tokens
    sampler_mod._progressiveserve_cachehit_patch_applied = True
    print("  ✅ Applied cache-hit prompt_logprobs patch (_get_next_prompt_tokens)")


def get_model_handle(llm: LLM):
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):
        raise RuntimeError("V1 engine detected. Set VLLM_USE_V1=0.")
    try:
        return engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError("Could not resolve v0 model handle path.") from exc


def register_progressive_model(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        arch = json.load(f)["architectures"][0]
    ModelRegistry.register_model(arch, ProgressiveForCausalLM)
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

    base_ids = tokenizer.encode(WIKI_BASE_TEXT, add_special_tokens=False)
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

    if len(token_ids) >= effective_total_tokens:
        max_offset = len(token_ids) - effective_total_tokens
        clamped_offset = max(0, min(offset, max_offset))
        full_ids = token_ids[clamped_offset:clamped_offset + effective_total_tokens]
    else:
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
    }


def build_eval_samples(
    tokenizer,
    texts: list[str],
    num_samples: int,
    target_total_tokens: int,
    block_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    encoded_docs: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
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


def choose_plp_shift_from_output(
    output_obj: Any,
    full_ids: list[int],
    full_len: int,
    plp_len: int,
    target_positions: list[int],
    shift_hint: Optional[int] = None,
) -> int:
    # prompt_logprobs length mismatch가 있어도 token-id 매칭률 최대화로 shift 선택.
    cached = getattr(output_obj, "num_cached_tokens", None)
    base = full_len - plp_len

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

    for shift in sorted(candidates):
        found = 0
        covered = 0
        for i in target_positions:
            j = i - shift
            if j <= 0 or j >= plp_len:
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

    return best_shift


def compute_span_ppl_from_token_ids(
    llm: LLM,
    prompt_token_ids: list[int],
    span_start_idx: int,
    span_end_idx: int,
    sampling_params: SamplingParams,
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

    full_ids = out.prompt_token_ids or prompt_token_ids
    plp_len = len(plp)
    full_len = len(full_ids)
    shift_hint = None
    cached = getattr(out, "num_cached_tokens", None)
    if cached is not None:
        shift_hint = int(cached) - 1
    shift = choose_plp_shift_from_output(
        output_obj=out,
        full_ids=full_ids,
        full_len=full_len,
        plp_len=plp_len,
        target_positions=target_positions,
        shift_hint=shift_hint,
    )

    nll = 0.0
    used = 0
    missing = 0
    token_not_found = 0

    for i in target_positions:
        j = i - shift
        if j <= 0:
            continue  # 첫 토큰 logprob 없음
        if j >= plp_len:
            missing += 1
            continue

        lp_dict = plp[j]
        if not lp_dict:
            missing += 1
            continue

        if i >= full_len:
            missing += 1
            continue

        tid = full_ids[i]
        lp = extract_selected_logprob(lp_dict, tid)
        if lp is None:
            token_not_found += 1
            continue

        nll -= lp
        used += 1

    if used == 0:
        raise RuntimeError(
            "No usable token logprobs were collected. "
            f"(full_len={full_len}, plp_len={plp_len}, shift={shift})"
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
        "full_prompt_len": full_len,
        "prompt_logprobs_len": plp_len,
        "plp_position_shift": shift,
        "num_cached_tokens": getattr(out, "num_cached_tokens", None),
    }


def transition_stage(
    llm: LLM,
    model,
    mode: str,
    config: dict[str, str],
    target_stage: int,
    skbi_seq_len_override: Optional[int] = None,
) -> dict[str, Any]:
    if target_stage not in STAGE_CONFIG:
        raise ValueError(f"Unsupported target_stage={target_stage}")

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

    info: dict[str, Any] = {
        "stage": stage_cfg["name"],
        "mode": mode,
        "t_prefetch_s": round(t_prefetch, 6),
        "t_activation_s": round(t_activation, 6),
        "policy_action": None,
        "boundary": None,
        "skbi_ok": None,
        "skbi_seq_len_override": skbi_seq_len_override,
    }

    if mode == "full_recompute":
        ok = llm.reset_prefix_cache()
        info["policy_action"] = f"reset_prefix_cache({ok})"
        print("  [Policy] full_recompute -> reset_prefix_cache()")
        return info

    if mode == "naive":
        info["policy_action"] = "keep_cache_untouched"
        print("  [Policy] naive -> KV cache untouched")
        return info

    if mode == "skbi":
        inner_model = model.model
        indices = getattr(model, stage_cfg["indices_fn"])()
        boundary = model.get_recompute_boundary(indices)
        info["boundary"] = boundary
        if boundary is None:
            raise RuntimeError("SKBI boundary is None.")

        if skbi_seq_len_override is not None:
            skbi_ok = inner_model.apply_skbi(boundary=boundary, seq_len=skbi_seq_len_override)
        else:
            skbi_ok = inner_model.apply_skbi(boundary=boundary)
        info["skbi_ok"] = bool(skbi_ok)
        if not skbi_ok:
            raise RuntimeError(
                f"apply_skbi(boundary={boundary}) failed at stage {stage_cfg['name']}."
            )
        if skbi_seq_len_override is not None:
            info["policy_action"] = (
                f"apply_skbi(boundary={boundary}, seq_len={skbi_seq_len_override})"
            )
            print(
                f"  [Policy] skbi -> apply_skbi(boundary={boundary}, "
                f"seq_len={skbi_seq_len_override})"
            )
        else:
            info["policy_action"] = f"apply_skbi(boundary={boundary})"
            print(f"  [Policy] skbi -> apply_skbi(boundary={boundary})")
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
    model_path: str,
    gpu_memory_utilization: float,
) -> tuple[LLM, Any, Any, int]:
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    model = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    kv_block_size = get_kv_block_size(model)
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print("  ✅ Persistent GPU buffers cleared")
    return llm, model, tokenizer, kv_block_size


def run_single_sample_eval(
    llm: LLM,
    model,
    mode: str,
    config: dict[str, str],
    a_ids: list[int],
    b_ids: list[int],
    c_ids: list[int],
) -> dict[str, Any]:
    n_a = len(a_ids)
    n_b = len(b_ids)
    n_c = len(c_ids)
    ab_ids = a_ids + b_ids
    abc_ids = ab_ids + c_ids

    sp = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )

    stage1_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=a_ids,
        span_start_idx=0,
        span_end_idx=n_a,
        sampling_params=sp,
    )
    print_turn_result(stage=1, turn=1, span_name="A (warmup span)", result=stage1_res)

    stage2_transition = transition_stage(
        llm=llm,
        model=model,
        mode=mode,
        config=config,
        target_stage=2,
        skbi_seq_len_override=n_a if mode == "skbi" else None,
    )
    stage2_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=ab_ids,
        span_start_idx=n_a,
        span_end_idx=n_a + n_b,
        sampling_params=sp,
    )
    print_turn_result(stage=2, turn=2, span_name="B (newly added tokens)", result=stage2_res)

    stage3_transition = transition_stage(
        llm=llm,
        model=model,
        mode=mode,
        config=config,
        target_stage=3,
        skbi_seq_len_override=(n_a + n_b) if mode == "skbi" else None,
    )
    stage3_res = compute_span_ppl_from_token_ids(
        llm=llm,
        prompt_token_ids=abc_ids,
        span_start_idx=n_a + n_b,
        span_end_idx=n_a + n_b + n_c,
        sampling_params=sp,
    )
    print_turn_result(stage=3, turn=3, span_name="C (newly added tokens)", result=stage3_res)

    return {
        "chunk_tokens": {"A": n_a, "B": n_b, "C": n_c, "total": n_a + n_b + n_c},
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

    unknown = [m for m in parsed if m not in VALID_MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Valid: {list(VALID_MODES)}")

    # 중복은 순서를 유지한 채 제거.
    ordered_unique: list[str] = []
    for m in parsed:
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


def run_mode_evaluation(
    model_name: str,
    mode: str,
    config: dict[str, str],
    model_path: str,
    gpu_memory_utilization: float,
    stop_on_error: bool,
    eval_samples: list[dict[str, Any]],
    dataset: str,
    dataset_jsonl: Optional[str],
    num_documents: int,
    seed: int,
    num_samples_requested: int,
    target_total_tokens: int,
    kv_block_size: int,
) -> dict[str, Any]:
    print("\n" + "#" * 72)
    print(f"[Mode] {mode}")
    print("#" * 72)

    llm, model, _, mode_kv_block_size = load_llm_runtime(
        model_path=model_path,
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
            llm, model, _, kv_block_size_next = load_llm_runtime(
                model_path=model_path,
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
            f"doc_tokens={sample['source_doc_tokens']})"
        )
        print("-" * 72)

        try:
            one = run_single_sample_eval(
                llm=llm,
                model=model,
                mode=mode,
                config=config,
                a_ids=sample["a_ids"],
                b_ids=sample["b_ids"],
                c_ids=sample["c_ids"],
            )
            row = {
                "sample_index": i,
                "sample_id": sample["sample_id"],
                "source_doc_idx": sample["source_doc_idx"],
                "source_doc_tokens": sample["source_doc_tokens"],
                "offset": sample["offset"],
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
        "model": model_name,
        "mode": mode,
        "vllm_use_v1": os.environ["VLLM_USE_V1"],
        "dataset": {
            "source": dataset,
            "dataset_jsonl": dataset_jsonl,
            "num_documents": num_documents,
            "seed": seed,
            "num_samples_requested": num_samples_requested,
            "num_samples_valid": len(valid_samples),
            "num_samples_failed": len(failed_samples),
        },
        "llm_config": {
            "enable_prefix_caching": True,
            "max_model_len": 2048,
            "enforce_eager": False,
            "sampling": {
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "temperature": 0.0,
            },
        },
        "eval_config": {
            "target_total_tokens": target_total_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "kv_block_size": mode_kv_block_size,
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

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lossless PPL evaluation for stage transitions (full_recompute vs naive vs skbi)."
    )
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama")
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
        help="Comma-separated multi-mode run. Example: full_recompute,naive,skbi",
    )
    parser.add_argument("--target-total-tokens", type=int, default=1008)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
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
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    eval_modes = resolve_eval_modes(mode=args.mode, modes=args.modes)

    print("=" * 72)
    print("ProgressiveServe Stage-Transition PPL Evaluation (Lossless Check)")
    print(
        f"  model={args.model} | modes={eval_modes} "
        f"| VLLM_USE_V1={os.environ['VLLM_USE_V1']}"
    )
    print(f"  gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)

    apply_cachehit_prompt_logprob_patch()

    config = MODELS[args.model]
    model_path = config["progressive_path"]
    arch = register_progressive_model(model_path)
    print(f"  Registered ProgressiveForCausalLM as: {arch}")

    llm, model, tokenizer, kv_block_size = load_llm_runtime(
        model_path=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    texts = load_eval_corpus_texts(args.dataset, args.dataset_jsonl)
    eval_samples = build_eval_samples(
        tokenizer=tokenizer,
        texts=texts,
        num_samples=args.num_samples,
        target_total_tokens=args.target_total_tokens,
        block_size=kv_block_size,
        seed=args.seed,
    )

    print(
        f"\n[Dataset] source={args.dataset} | docs={len(texts)} | "
        f"samples={len(eval_samples)} | seed={args.seed}"
    )
    print(
        f"[Data] block_size={kv_block_size} | target_total_tokens={args.target_total_tokens} "
        f"(effective total is block-aligned per sample)"
    )

    del llm
    del model
    gc.collect()
    torch.cuda.empty_cache()

    multi_mode = len(eval_modes) > 1
    results_by_mode: dict[str, dict[str, Any]] = {}
    for mode in eval_modes:
        result = run_mode_evaluation(
            model_name=args.model,
            mode=mode,
            config=config,
            model_path=model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            stop_on_error=args.stop_on_error,
            eval_samples=eval_samples,
            dataset=args.dataset,
            dataset_jsonl=args.dataset_jsonl,
            num_documents=len(texts),
            seed=args.seed,
            num_samples_requested=args.num_samples,
            target_total_tokens=args.target_total_tokens,
            kv_block_size=kv_block_size,
        )
        results_by_mode[mode] = result
        out_path = resolve_output_path_for_mode(
            model=args.model,
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
