#!/usr/bin/env python3
"""
Universal Progressive Serving - Baseline vs Progressive Benchmark
=================================================================

Baseline (전체 모델 한번에 로드) vs Progressive (Stage 1→2→3 전환) 비교
nsys 프로파일링, 시간/메모리 측정, JSON 결과 저장

Usage:
  # Baseline만
  python 02_universal.py --mode baseline --model llama

  # Progressive만
  python 02_universal.py --mode progressive --model llama

  # 둘 다 (baseline → progressive)
  python 02_universal.py --mode both --model llama

  # nsys 프로파일링
  nsys profile -o report python 02_universal.py --mode both --model llama
  
  
  sync
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
"""

import os
import sys
import json
import time
import uuid
import socket
import logging
import re
import argparse
import gc
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, Optional, List

# vLLM v0 엔진 강제 사용 (모델 직접 접근 필요)
os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams, __version__ as vllm_version
from vllm.model_executor.models.registry import ModelRegistry

# psutil for network monitoring
try:
    import psutil
except ImportError:
    print("psutil not installed. Installing...")
    os.system("pip install psutil")
    import psutil

# Progressive model
sys.path.insert(0, "/home/devewha/v08/Juwon/01_universal/progressive_serve")
from progressive_for_causal_lm import ProgressiveForCausalLM

# ============================================================================
# NVTX helpers (nsys 프로파일링용)
# ============================================================================

def nvtx_push(name: str):
    """nsys NVTX range 시작"""
    torch.cuda.nvtx.range_push(name)

def nvtx_pop():
    """nsys NVTX range 종료"""
    torch.cuda.nvtx.range_pop()


# ============================================================================
# 모델 설정
# ============================================================================

MODELS = {
    "llama": {
        "baseline_path": "/acpl-ssd30/meta-llama/Llama-2-7b-chat-hf",
        "progressive_path": "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/A",
        "stage_b_checkpoint": "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    },
    "mistral": {
        "baseline_path": "mistralai/Mistral-7B-v0.1",
        "progressive_path": "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    },
}

# NFS 인터페이스 (네트워크 모니터링)
NFS_INTERFACE = "bond0.88"

# 결과 저장 디렉토리
RESULT_DIR = "/home/devewha/v08/results"

# Stage 전환 직후 강제 shape warmup 실행 여부
# 기본값 False: 불필요한 graph 재캡처 유발 방지
POST_TRANSITION_SHAPE_WARMUP_DEFAULT = False

# 테스트 프롬프트
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "In a galaxy far, far away",
]

# 실험 전 구간 max_tokens 고정 (그래프 shape 변동 최소화)
FIXED_MAX_TOKENS = 50


# ============================================================================
# vLLM Log Parser
# ============================================================================

class VLLMLogParser(logging.Handler):
    """vLLM 내부 로그 파싱"""

    def __init__(self, max_raw_logs: int = 100):
        super().__init__()
        self.logs = {
            "weight_loading_time": None,
            "model_loading_gb": None,
            "model_loading_time": None,
            "memory_profiling_time": None,
            "total_gpu_memory_gb": None,
            "gpu_memory_utilization": None,
            "model_weights_gb": None,
            "kv_cache_gb": None,
            "cuda_blocks": None,
            "cpu_blocks": None,
            "graph_capturing_time": None,
            "graph_capturing_memory_gb": None,
            "init_engine_time": None,
            "raw_logs": [],
        }
        self.max_raw_logs = max_raw_logs

    def emit(self, record):
        msg = record.getMessage()

        if len(self.logs["raw_logs"]) < self.max_raw_logs:
            self.logs["raw_logs"].append(msg)

        patterns = [
            (r"Loading weights took ([\d.]+) seconds",
             lambda m: self._set("weight_loading_time", float(m.group(1)))),
            (r"Model loading took ([\d.]+) GB and ([\d.]+) seconds",
             lambda m: (self._set("model_loading_gb", float(m.group(1))),
                        self._set("model_loading_time", float(m.group(2))))),
            (r"Memory profiling takes ([\d.]+) seconds",
             lambda m: self._set("memory_profiling_time", float(m.group(1)))),
            (r"total_gpu_memory \(([\d.]+)GiB\) x gpu_memory_utilization \(([\d.]+)\)",
             lambda m: (self._set("total_gpu_memory_gb", float(m.group(1))),
                        self._set("gpu_memory_utilization", float(m.group(2))))),
            (r"model weights take ([\d.]+)GiB",
             lambda m: self._set("model_weights_gb", float(m.group(1)))),
            (r"KV Cache is ([\d.]+)GiB",
             lambda m: self._set("kv_cache_gb", float(m.group(1)))),
            (r"# cuda blocks: (\d+), # CPU blocks: (\d+)",
             lambda m: (self._set("cuda_blocks", int(m.group(1))),
                        self._set("cpu_blocks", int(m.group(2))))),
            (r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB",
             lambda m: (self._set("graph_capturing_time", float(m.group(1))),
                        self._set("graph_capturing_memory_gb", float(m.group(2))))),
            (r"init engine.*?took ([\d.]+) seconds",
             lambda m: self._set("init_engine_time", float(m.group(1)))),
        ]

        for pattern, action in patterns:
            match = re.search(pattern, msg)
            if match:
                action(match)

    def _set(self, key, value):
        self.logs[key] = value


# ============================================================================
# Network Monitor
# ============================================================================

class NetworkMonitor:
    """네트워크 I/O 측정 (NFS 등)"""

    def __init__(self, interface: str = NFS_INTERFACE):
        self.interface = interface
        self.start_time = None
        self.start_counters = None
        self.error_message = None

    def start(self):
        try:
            all_if = psutil.net_io_counters(pernic=True)
            if self.interface not in all_if:
                self.error_message = f"Interface '{self.interface}' not found."
                self.start_counters = None
                return
            self.start_counters = all_if[self.interface]
            self.start_time = time.time()
        except Exception as e:
            self.error_message = str(e)
            self.start_counters = None

    def stop(self) -> Dict:
        if self.start_counters is None:
            return {"error": True, "error_message": self.error_message}
        try:
            end = psutil.net_io_counters(pernic=True)[self.interface]
            dur = time.time() - self.start_time
            recv = end.bytes_recv - self.start_counters.bytes_recv
            sent = end.bytes_sent - self.start_counters.bytes_sent
            return {
                "interface": self.interface,
                "duration_seconds": dur,
                "bytes_received_gb": recv / (1024 ** 3),
                "bytes_sent_gb": sent / (1024 ** 3),
                "throughput_mbs": recv / (dur * 1024 * 1024) if dur > 0 else 0,
                "error": False,
            }
        except Exception as e:
            return {"error": True, "error_message": str(e)}


# ============================================================================
# GPU Memory Snapshot
# ============================================================================

def gpu_memory_snapshot() -> Dict:
    """현재 GPU 메모리 상태"""
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def reset_gpu_memory_stats():
    """GPU 메모리 통계 초기화"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Benchmark Runner
# ============================================================================

class UniversalBenchmark:
    """Baseline vs Progressive 벤치마크"""

    def __init__(
        self,
        model_name: str,
        interface: str = NFS_INTERFACE,
        post_transition_shape_warmup: bool = POST_TRANSITION_SHAPE_WARMUP_DEFAULT,
    ):
        self.model_name = model_name
        self.model_config = MODELS[model_name]
        self.interface = interface
        self.post_transition_shape_warmup = post_transition_shape_warmup
        self.results = {}

    def setup_logging(self) -> VLLMLogParser:
        """vLLM 로그 파서 설정"""
        logger = logging.getLogger("vllm")
        logger.setLevel(logging.INFO)
        # 이전 파서 제거
        for h in logger.handlers[:]:
            if isinstance(h, VLLMLogParser):
                logger.removeHandler(h)
        parser = VLLMLogParser()
        logger.addHandler(parser)
        return parser

    def cleanup_logging(self, parser: VLLMLogParser):
        logger = logging.getLogger("vllm")
        logger.removeHandler(parser)

    def _timeline_start(self, name: str) -> Dict:
        """상대시간 기반 타임라인 시작"""
        return {
            "name": name,
            "start_unix": time.time(),
            "_t0": time.perf_counter(),
            "events": [],
        }

    def _timeline_mark(self, timeline: Dict, event: str, **meta):
        """타임라인 이벤트 기록"""
        now_perf = time.perf_counter()
        item = {
            "event": event,
            "t_rel_s": now_perf - timeline["_t0"],
            "t_unix": time.time(),
        }
        if meta:
            item["meta"] = meta
        timeline["events"].append(item)

    def _timeline_finalize(self, timeline: Dict) -> Dict:
        """타임라인 종료"""
        timeline["total_duration_s"] = time.perf_counter() - timeline["_t0"]
        del timeline["_t0"]
        return timeline

    def _get_progressive_model_handle(self, llm: LLM):
        """
        v0 엔진에서 progressive model 객체를 안전하게 가져온다.
        v1 엔진으로 구동되면 즉시 실패시켜 원인 추적을 명확히 한다.
        """
        engine = llm.llm_engine
        if hasattr(engine, "engine_core"):
            raise RuntimeError(
                "V1 engine detected. This script is v0-only. "
                "Use a v0 build/environment and keep VLLM_USE_V1=0."
            )

        try:
            return engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError as exc:
            raise RuntimeError(
                "Could not resolve v0 model handle path: "
                "llm.llm_engine.model_executor.driver_worker.worker.model_runner.model"
            ) from exc

    def _warmup_fixed_shapes(self, llm: LLM, tag: str) -> float:
        """
        측정 전에 고정된 요청 shape를 미리 실행해 CUDA graph 재캡처를 방지한다.
        baseline/progressive 모두 동일하게 적용한다.
        """
        nvtx_push(f"{tag}_shape_warmup")
        start_t = time.time()
        llm.generate(["shape_ttft"], SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0))
        llm.generate(
            TEST_PROMPTS,
            SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
        )
        dur = time.time() - start_t
        nvtx_pop()
        return dur

    def _run_prefetch_with_overlap(
        self,
        llm: LLM,
        model,
        stage_name: str,
        prefetch_fn,
        checkpoint_path: str,
        max_overlap_rounds: int = 4,
    ) -> Dict:
        """
        prefetch 시작 후, 완료될 때까지 유효 서빙을 겹쳐 수행한다.
        """
        nvtx_push(f"{stage_name}_prefetch")
        prefetch_start = time.time()
        prefetch_fn(checkpoint_path)
        prefetch_launch_time = time.time() - prefetch_start

        overlap_rounds = 0
        overlap_serving_time = 0.0
        while (not model.is_prefetch_ready()) and overlap_rounds < max_overlap_rounds:
            overlap_rounds += 1
            nvtx_push(f"{stage_name}_overlap_infer_round{overlap_rounds}")
            t0 = time.time()
            llm.generate(
                TEST_PROMPTS,
                SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
            )
            overlap_serving_time += (time.time() - t0)
            nvtx_pop()

        ready_before_wait = model.is_prefetch_ready()
        wait_start = time.time()
        ready_after_wait = model.wait_for_prefetch(timeout_s=120.0)
        prefetch_wait_time = time.time() - wait_start
        prefetch_total_time = time.time() - prefetch_start
        status = model.get_prefetch_status()
        nvtx_pop()

        return {
            "checkpoint_path": checkpoint_path,
            "prefetch_launch_time": prefetch_launch_time,
            "overlap_rounds": overlap_rounds,
            "overlap_serving_time": overlap_serving_time,
            "ready_before_wait": ready_before_wait,
            "ready_after_wait": ready_after_wait,
            "wait_time": prefetch_wait_time,
            "total_time": prefetch_total_time,
            "status_before_activation": status,
        }

    # ----------------------------------------------------------------
    # Baseline 측정
    # ----------------------------------------------------------------
    def measure_baseline(self) -> Dict:
        """Baseline: 전체 모델을 한번에 로드하고 추론"""
        print("\n" + "=" * 80)
        print(f"BASELINE MEASUREMENT ({self.model_name.upper()})")
        print(f"Path: {self.model_config['baseline_path']}")
        print("=" * 80)

        log_parser = self.setup_logging()
        net_monitor = NetworkMonitor(self.interface)
        reset_gpu_memory_stats()

        result = {
            "type": "baseline",
            "model_name": self.model_name,
            "model_path": self.model_config["baseline_path"],
            "timestamp": datetime.now().isoformat(),
        }
        timeline = self._timeline_start("baseline")

        # [1] Cold Start (모델 로딩)
        print("\n[1] Cold Start - Loading full model...")
        self._timeline_mark(timeline, "cold_start_begin")
        nvtx_push("baseline_cold_start")
        net_monitor.start()
        cold_start_t = time.time()

        llm = LLM(
            model=self.model_config["baseline_path"],
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            max_model_len=2048,
            enforce_eager=False,  # SIGSEGV 디버깅: CUDA Graph 비활성화
        )

        result["cold_start_time"] = time.time() - cold_start_t
        result["network"] = net_monitor.stop()
        self._timeline_mark(timeline, "model_loaded", cold_start_time=result["cold_start_time"])
        nvtx_pop()

        result["gpu_after_load"] = gpu_memory_snapshot()
        print(f"  Cold Start: {result['cold_start_time']:.2f}s")
        print(f"  GPU Memory: {result['gpu_after_load']['allocated_gb']:.2f} GB")

        # [2] TTFT (End-to-End: cold start + first request)
        print("\n[2] Measuring TTFT (E2E: fetch/load/graph + first token)...")
        nvtx_push("baseline_ttft_e2e")
        ttft_start = time.time()
        llm.generate(
            ["What is the capital of France?"],
            SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0),
        )
        ttft_request_only = time.time() - ttft_start
        result["ttft_request_only"] = ttft_request_only
        result["ttft"] = result["cold_start_time"] + ttft_request_only
        result["ttft_breakdown"] = {
            "e2e_total": result["ttft"],
            "fetch_and_model_init": result["cold_start_time"],
            "request_to_first_token": ttft_request_only,
            "network_fetch_window": result["network"].get("duration_seconds")
                if not result["network"].get("error")
                else None,
            "weight_loading_time": log_parser.logs.get("weight_loading_time"),
            "model_loading_time": log_parser.logs.get("model_loading_time"),
            "memory_profiling_time": log_parser.logs.get("memory_profiling_time"),
            "graph_capturing_time": log_parser.logs.get("graph_capturing_time"),
        }
        self._timeline_mark(
            timeline,
            "first_token_returned",
            ttft_e2e=result["ttft"],
            ttft_request_only=ttft_request_only,
        )
        nvtx_pop()
        print(f"  TTFT(E2E): {result['ttft']:.4f}s (request-only: {ttft_request_only:.4f}s)")

        # [3] Warmup (TTFT 측정 이후)
        print("\n[3] Warmup inference...")
        nvtx_push("baseline_warmup")
        llm.generate(["warmup"], SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0))
        nvtx_pop()

        print("[3-1] Fixed-shape warmup (baseline/progressive 동일 조건)...")
        baseline_shape_warmup = self._warmup_fixed_shapes(llm, "baseline")
        result["shape_warmup_time"] = baseline_shape_warmup
        self._timeline_mark(timeline, "warmup_done", warmup_time=baseline_shape_warmup)
        print(f"  Shape warmup time: {baseline_shape_warmup:.4f}s")

        # [4] Throughput
        print("\n[4] Measuring throughput...")
        nvtx_push("baseline_throughput")
        tp_start = time.time()
        outputs = llm.generate(
            TEST_PROMPTS,
            SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
        )
        tp_dur = time.time() - tp_start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        result["throughput_tokens"] = total_tokens
        result["throughput_duration"] = tp_dur
        result["throughput_tok_per_sec"] = total_tokens / tp_dur if tp_dur > 0 else 0
        self._timeline_mark(
            timeline,
            "throughput_done",
            duration=tp_dur,
            tokens=total_tokens,
            tok_per_sec=result["throughput_tok_per_sec"],
        )
        nvtx_pop()
        print(f"  Throughput: {result['throughput_tok_per_sec']:.2f} tok/s ({total_tokens} tokens in {tp_dur:.2f}s)")

        # [5] Final memory
        result["gpu_final"] = gpu_memory_snapshot()
        self._timeline_mark(timeline, "measurement_done")
        result["timeline"] = self._timeline_finalize(timeline)
        print(f"  Timeline Total: {result['timeline']['total_duration_s']:.4f}s")
        result["vllm_logs"] = log_parser.logs
        self.cleanup_logging(log_parser)

        # Cleanup
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

        print(f"\n  Baseline measurement complete.")
        return result

    # ----------------------------------------------------------------
    # Progressive 측정
    # ----------------------------------------------------------------
    def measure_progressive(self) -> Dict:
        """Progressive: Stage 1 로드 → prefetch → Stage 2 → prefetch → Stage 3"""
        print("\n" + "=" * 80)
        print(f"PROGRESSIVE MEASUREMENT ({self.model_name.upper()})")
        print(f"Path: {self.model_config['progressive_path']}")
        print("=" * 80)

        log_parser = self.setup_logging()
        net_monitor = NetworkMonitor(self.interface)
        reset_gpu_memory_stats()

        # config.json에서 아키텍처 읽기 → 등록
        model_path = self.model_config["progressive_path"]
        with open(os.path.join(model_path, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        ModelRegistry.register_model(arch, ProgressiveForCausalLM)
        print(f"  Registered ProgressiveForCausalLM as: {arch}")

        result = {
            "type": "progressive",
            "model_name": self.model_name,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "stage_transition_times": {},
        }
        timeline = self._timeline_start("progressive")

        # ============================================================
        # Stage 1: 초기 로딩
        # ============================================================
        print("\n--- STAGE 1: Initial Load ---")
        self._timeline_mark(timeline, "stage1_load_begin")
        nvtx_push("progressive_stage1_cold_start")
        net_monitor.start()
        cold_start_t = time.time()

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            max_model_len=2048,
            enforce_eager=False,  # SIGSEGV 디버깅: CUDA Graph 비활성화
        )

        stage1_cold_start = time.time() - cold_start_t
        stage1_network = net_monitor.stop()
        nvtx_pop()

        result["cold_start_time"] = stage1_cold_start
        result["network"] = stage1_network
        result["gpu_after_load"] = gpu_memory_snapshot()
        self._timeline_mark(timeline, "stage1_loaded", cold_start_time=stage1_cold_start)

        print(f"  Stage 1 Cold Start: {stage1_cold_start:.2f}s")
        print(f"  GPU Memory: {result['gpu_after_load']['allocated_gb']:.2f} GB")

        # vLLM v0 엔진 모델 직접 접근
        model = self._get_progressive_model_handle(llm)

        # Stage 1 TTFT (End-to-End: stage1 load + first request)
        print("\n  Measuring Stage 1 TTFT (E2E)...")
        nvtx_push("progressive_stage1_ttft_e2e")
        ttft_start = time.time()
        llm.generate(
            ["What is the capital of France?"],
            SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0),
        )
        stage1_ttft_request_only = time.time() - ttft_start
        stage1_ttft_e2e = stage1_cold_start + stage1_ttft_request_only
        self._timeline_mark(
            timeline,
            "stage1_first_token_returned",
            ttft_e2e=stage1_ttft_e2e,
            ttft_request_only=stage1_ttft_request_only,
        )
        nvtx_pop()
        print(
            f"  Stage 1 TTFT(E2E): {stage1_ttft_e2e:.4f}s "
            f"(request-only: {stage1_ttft_request_only:.4f}s)"
        )

        # Warmup (TTFT 이후)
        print("\n  Warmup inference...")
        nvtx_push("progressive_warmup")
        llm.generate(["warmup"], SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0))
        nvtx_pop()

        print("  Fixed-shape warmup (baseline/progressive 동일 조건)...")
        stage1_shape_warmup = self._warmup_fixed_shapes(llm, "progressive_stage1")
        self._timeline_mark(timeline, "stage1_warmup_done", warmup_time=stage1_shape_warmup)
        print(f"  Shape warmup time: {stage1_shape_warmup:.4f}s")

        # Stage 1 Throughput
        print("\n  Measuring Stage 1 throughput...")
        nvtx_push("progressive_stage1_throughput")
        tp_start = time.time()
        outputs = llm.generate(
            TEST_PROMPTS,
            SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
        )
        tp_dur = time.time() - tp_start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        nvtx_pop()

        stage_info = model.get_stage_info()
        result["stages"]["stage1"] = {
            "cold_start_time": stage1_cold_start,
            "shape_warmup_time": stage1_shape_warmup,
            "ttft": stage1_ttft_e2e,
            "ttft_request_only": stage1_ttft_request_only,
            "ttft_breakdown": {
                "e2e_total": stage1_ttft_e2e,
                "fetch_and_model_init": stage1_cold_start,
                "request_to_first_token": stage1_ttft_request_only,
                "network_fetch_window": stage1_network.get("duration_seconds")
                    if not stage1_network.get("error")
                    else None,
                "weight_loading_time": log_parser.logs.get("weight_loading_time"),
                "model_loading_time": log_parser.logs.get("model_loading_time"),
                "memory_profiling_time": log_parser.logs.get("memory_profiling_time"),
                "graph_capturing_time": log_parser.logs.get("graph_capturing_time"),
            },
            "throughput_tok_per_sec": total_tokens / tp_dur if tp_dur > 0 else 0,
            "throughput_tokens": total_tokens,
            "throughput_duration": tp_dur,
            "gpu_memory": gpu_memory_snapshot(),
            "active_layers": len(stage_info["active_layers"]),
            "inactive_layers": len(stage_info["inactive_layers"]),
            "activation_progress": stage_info["activation_progress"],
        }
        self._timeline_mark(
            timeline,
            "stage1_throughput_done",
            duration=tp_dur,
            tokens=total_tokens,
            tok_per_sec=result["stages"]["stage1"]["throughput_tok_per_sec"],
        )
        print(f"  Stage 1 Throughput: {result['stages']['stage1']['throughput_tok_per_sec']:.2f} tok/s")
        print(f"  Active Layers: {len(stage_info['active_layers'])}")

        # ============================================================
        # Stage 2: Prefetch → Instant Transition
        # ============================================================
        stage_b_path = self.model_config.get("stage_b_checkpoint")
        if stage_b_path and os.path.exists(stage_b_path):
            print("\n--- STAGE 2: Prefetch + Transition ---")
            prefetch_metrics = self._run_prefetch_with_overlap(
                llm=llm,
                model=model,
                stage_name="progressive_stage2",
                prefetch_fn=model.prefetch_stage2,
                checkpoint_path=stage_b_path,
            )
            if not prefetch_metrics["ready_after_wait"]:
                raise RuntimeError("Stage 2 prefetch failed or timed out.")

            print(
                "  Prefetch: "
                f"total={prefetch_metrics['total_time']:.4f}s, "
                f"overlap={prefetch_metrics['overlap_serving_time']:.4f}s "
                f"({prefetch_metrics['overlap_rounds']} rounds), "
                f"wait_before_transition={prefetch_metrics['wait_time']:.4f}s"
            )
            self._timeline_mark(
                timeline,
                "stage3_prefetch_ready",
                prefetch_total=prefetch_metrics["total_time"],
                prefetch_wait=prefetch_metrics["wait_time"],
                overlap_rounds=prefetch_metrics["overlap_rounds"],
            )
            self._timeline_mark(
                timeline,
                "stage2_prefetch_ready",
                prefetch_total=prefetch_metrics["total_time"],
                prefetch_wait=prefetch_metrics["wait_time"],
                overlap_rounds=prefetch_metrics["overlap_rounds"],
            )

            # Instant 전환 (순수 GPU copy + alpha only)
            nvtx_push("progressive_stage2_transition_instant")
            instant_start = time.time()
            transitioned = model.advance_to_stage2_instant(wait_if_needed=False)
            instant_transition_time = time.time() - instant_start
            nvtx_pop()
            if not transitioned:
                raise RuntimeError("Stage 2 instant transition failed (prefetch not ready).")

            stage2_transition_time = prefetch_metrics["wait_time"] + instant_transition_time
            print(
                "  Stage 1→2 transition: "
                f"total={stage2_transition_time:.4f}s "
                f"(wait={prefetch_metrics['wait_time']:.4f}s + "
                f"instant={instant_transition_time:.4f}s)"
            )
            self._timeline_mark(
                timeline,
                "stage2_transition_done",
                transition_total=stage2_transition_time,
                transition_instant=instant_transition_time,
            )

            stage2_shape_warmup = 0.0
            if self.post_transition_shape_warmup:
                # 옵션: 전환 직후 강제 warmup
                stage2_shape_warmup = self._warmup_fixed_shapes(llm, "progressive_stage2")
                print(f"  Post-transition shape warmup: {stage2_shape_warmup:.4f}s")
            else:
                print("  Post-transition shape warmup: skipped (graph recapture 방지)")

            # Stage 2 TTFT
            print("\n  Measuring Stage 2 TTFT...")
            nvtx_push("progressive_stage2_ttft")
            ttft_start = time.time()
            llm.generate(
                ["What is the capital of France?"],
                SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0),
            )
            stage2_ttft_request_only = time.time() - ttft_start
            stage2_ttft_e2e = (
                prefetch_metrics["total_time"]
                + instant_transition_time
                + stage2_ttft_request_only
            )
            self._timeline_mark(
                timeline,
                "stage2_first_token_returned",
                ttft_e2e=stage2_ttft_e2e,
                ttft_request_only=stage2_ttft_request_only,
            )
            nvtx_pop()
            print(
                f"  Stage 2 TTFT(E2E from prefetch start): {stage2_ttft_e2e:.4f}s "
                f"(request-only: {stage2_ttft_request_only:.4f}s)"
            )

            # Stage 2 Throughput
            print("\n  Measuring Stage 2 throughput...")
            nvtx_push("progressive_stage2_throughput")
            tp_start = time.time()
            outputs = llm.generate(
                TEST_PROMPTS,
                SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
            )
            tp_dur = time.time() - tp_start
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            nvtx_pop()

            stage_info = model.get_stage_info()
            result["stages"]["stage2"] = {
                "transition_from": "stage1_to_stage2",
                "prefetch_duration": prefetch_metrics["total_time"],  # backward compatibility
                "prefetch_launch_time": prefetch_metrics["prefetch_launch_time"],
                "prefetch_overlap_serving_time": prefetch_metrics["overlap_serving_time"],
                "prefetch_overlap_rounds": prefetch_metrics["overlap_rounds"],
                "prefetch_wait_time": prefetch_metrics["wait_time"],
                "prefetch_ready_before_wait": prefetch_metrics["ready_before_wait"],
                "prefetch_ready_after_wait": prefetch_metrics["ready_after_wait"],
                "prefetch_status_before_activation": prefetch_metrics["status_before_activation"],
                "transition_time": stage2_transition_time,
                "instant_transition_time": instant_transition_time,
                "post_transition_shape_warmup_enabled": self.post_transition_shape_warmup,
                "shape_warmup_time": stage2_shape_warmup,
                "ttft": stage2_ttft_e2e,
                "ttft_request_only": stage2_ttft_request_only,
                "ttft_breakdown": {
                    "e2e_total_from_prefetch_start": stage2_ttft_e2e,
                    "prefetch_total_time": prefetch_metrics["total_time"],
                    "transition_instant_time": instant_transition_time,
                    "request_to_first_token": stage2_ttft_request_only,
                    "post_transition_shape_warmup": stage2_shape_warmup,
                },
                "throughput_tok_per_sec": total_tokens / tp_dur if tp_dur > 0 else 0,
                "throughput_tokens": total_tokens,
                "throughput_duration": tp_dur,
                "gpu_memory": gpu_memory_snapshot(),
                "active_layers": len(stage_info["active_layers"]),
                "inactive_layers": len(stage_info["inactive_layers"]),
                "activation_progress": stage_info["activation_progress"],
            }
            self._timeline_mark(
                timeline,
                "stage2_throughput_done",
                duration=tp_dur,
                tokens=total_tokens,
                tok_per_sec=result["stages"]["stage2"]["throughput_tok_per_sec"],
            )
            result["stage_transition_times"]["stage1_to_stage2"] = stage2_transition_time
            print(f"  Stage 2 Throughput: {result['stages']['stage2']['throughput_tok_per_sec']:.2f} tok/s")
            print(f"  Active Layers: {len(stage_info['active_layers'])}")
        else:
            print(f"\n  Stage B checkpoint not found, skipping Stage 2")

        # ============================================================
        # Stage 3: Prefetch → Instant Transition
        # ============================================================
        stage_c_path = self.model_config.get("stage_c_checkpoint")
        if stage_c_path and os.path.exists(stage_c_path) and "stage2" in result["stages"]:
            print("\n--- STAGE 3: Prefetch + Transition ---")
            prefetch_metrics = self._run_prefetch_with_overlap(
                llm=llm,
                model=model,
                stage_name="progressive_stage3",
                prefetch_fn=model.prefetch_stage3,
                checkpoint_path=stage_c_path,
            )
            if not prefetch_metrics["ready_after_wait"]:
                raise RuntimeError("Stage 3 prefetch failed or timed out.")

            print(
                "  Prefetch: "
                f"total={prefetch_metrics['total_time']:.4f}s, "
                f"overlap={prefetch_metrics['overlap_serving_time']:.4f}s "
                f"({prefetch_metrics['overlap_rounds']} rounds), "
                f"wait_before_transition={prefetch_metrics['wait_time']:.4f}s"
            )

            # Instant 전환 (순수 GPU copy + alpha only)
            nvtx_push("progressive_stage3_transition_instant")
            instant_start = time.time()
            transitioned = model.advance_to_stage3_instant(wait_if_needed=False)
            instant_transition_time = time.time() - instant_start
            nvtx_pop()
            if not transitioned:
                raise RuntimeError("Stage 3 instant transition failed (prefetch not ready).")

            stage3_transition_time = prefetch_metrics["wait_time"] + instant_transition_time
            print(
                "  Stage 2→3 transition: "
                f"total={stage3_transition_time:.4f}s "
                f"(wait={prefetch_metrics['wait_time']:.4f}s + "
                f"instant={instant_transition_time:.4f}s)"
            )
            self._timeline_mark(
                timeline,
                "stage3_transition_done",
                transition_total=stage3_transition_time,
                transition_instant=instant_transition_time,
            )

            stage3_shape_warmup = 0.0
            if self.post_transition_shape_warmup:
                # 옵션: 전환 직후 강제 warmup
                stage3_shape_warmup = self._warmup_fixed_shapes(llm, "progressive_stage3")
                print(f"  Post-transition shape warmup: {stage3_shape_warmup:.4f}s")
            else:
                print("  Post-transition shape warmup: skipped (graph recapture 방지)")

            # Stage 3 TTFT
            print("\n  Measuring Stage 3 TTFT...")
            nvtx_push("progressive_stage3_ttft")
            ttft_start = time.time()
            llm.generate(
                ["What is the capital of France?"],
                SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0),
            )
            stage3_ttft_request_only = time.time() - ttft_start
            stage3_ttft_e2e = (
                prefetch_metrics["total_time"]
                + instant_transition_time
                + stage3_ttft_request_only
            )
            self._timeline_mark(
                timeline,
                "stage3_first_token_returned",
                ttft_e2e=stage3_ttft_e2e,
                ttft_request_only=stage3_ttft_request_only,
            )
            nvtx_pop()
            print(
                f"  Stage 3 TTFT(E2E from prefetch start): {stage3_ttft_e2e:.4f}s "
                f"(request-only: {stage3_ttft_request_only:.4f}s)"
            )

            # Stage 3 Throughput
            print("\n  Measuring Stage 3 throughput...")
            nvtx_push("progressive_stage3_throughput")
            tp_start = time.time()
            outputs = llm.generate(
                TEST_PROMPTS,
                SamplingParams(max_tokens=FIXED_MAX_TOKENS, temperature=0.8, top_p=0.95),
            )
            tp_dur = time.time() - tp_start
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            nvtx_pop()

            stage_info = model.get_stage_info()
            result["stages"]["stage3"] = {
                "transition_from": "stage2_to_stage3",
                "prefetch_duration": prefetch_metrics["total_time"],  # backward compatibility
                "prefetch_launch_time": prefetch_metrics["prefetch_launch_time"],
                "prefetch_overlap_serving_time": prefetch_metrics["overlap_serving_time"],
                "prefetch_overlap_rounds": prefetch_metrics["overlap_rounds"],
                "prefetch_wait_time": prefetch_metrics["wait_time"],
                "prefetch_ready_before_wait": prefetch_metrics["ready_before_wait"],
                "prefetch_ready_after_wait": prefetch_metrics["ready_after_wait"],
                "prefetch_status_before_activation": prefetch_metrics["status_before_activation"],
                "transition_time": stage3_transition_time,
                "instant_transition_time": instant_transition_time,
                "post_transition_shape_warmup_enabled": self.post_transition_shape_warmup,
                "shape_warmup_time": stage3_shape_warmup,
                "ttft": stage3_ttft_e2e,
                "ttft_request_only": stage3_ttft_request_only,
                "ttft_breakdown": {
                    "e2e_total_from_prefetch_start": stage3_ttft_e2e,
                    "prefetch_total_time": prefetch_metrics["total_time"],
                    "transition_instant_time": instant_transition_time,
                    "request_to_first_token": stage3_ttft_request_only,
                    "post_transition_shape_warmup": stage3_shape_warmup,
                },
                "throughput_tok_per_sec": total_tokens / tp_dur if tp_dur > 0 else 0,
                "throughput_tokens": total_tokens,
                "throughput_duration": tp_dur,
                "gpu_memory": gpu_memory_snapshot(),
                "active_layers": len(stage_info["active_layers"]),
                "inactive_layers": len(stage_info["inactive_layers"]),
                "activation_progress": stage_info["activation_progress"],
            }
            self._timeline_mark(
                timeline,
                "stage3_throughput_done",
                duration=tp_dur,
                tokens=total_tokens,
                tok_per_sec=result["stages"]["stage3"]["throughput_tok_per_sec"],
            )
            result["stage_transition_times"]["stage2_to_stage3"] = stage3_transition_time
            print(f"  Stage 3 Throughput: {result['stages']['stage3']['throughput_tok_per_sec']:.2f} tok/s")
            print(f"  Active Layers: {len(stage_info['active_layers'])}")
        else:
            print(f"\n  Stage C checkpoint not found, skipping Stage 3")

        # 최종 결과
        result["gpu_final"] = gpu_memory_snapshot()

        # 총 전환 시간 계산 (Stage 1 로드 ~ Stage 3 완료)
        total_transition = 0
        for s in ["stage2", "stage3"]:
            if s in result["stages"]:
                total_transition += result["stages"][s].get("transition_time", 0)
        result["total_transition_time"] = total_transition

        # TTFT: Stage 1 E2E 기준 (fetch/load/graph + first request)
        result["ttft"] = stage1_ttft_e2e
        # Throughput: 마지막 stage 기준
        last_stage = max(result["stages"].keys())
        result["throughput_tok_per_sec"] = result["stages"][last_stage]["throughput_tok_per_sec"]

        self._timeline_mark(timeline, "measurement_done")
        result["timeline"] = self._timeline_finalize(timeline)
        print(f"  Timeline Total: {result['timeline']['total_duration_s']:.4f}s")
        result["vllm_logs"] = log_parser.logs
        self.cleanup_logging(log_parser)

        # Cleanup
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

        print(f"\n  Progressive measurement complete.")
        return result

    # ----------------------------------------------------------------
    # 비교 테이블
    # ----------------------------------------------------------------
    def print_comparison(self, baseline: Dict, progressive: Dict):
        """비교 결과 출력"""
        print("\n" + "=" * 90)
        print("COMPARISON: BASELINE vs PROGRESSIVE")
        print("=" * 90)

        def pct(base, prog):
            if base == 0:
                return "N/A"
            return f"{(base - prog) / base * 100:.1f}%"

        rows = [
            ("Cold Start Time (s)",
             f"{baseline['cold_start_time']:.2f}",
             f"{progressive['cold_start_time']:.2f}",
             pct(baseline["cold_start_time"], progressive["cold_start_time"])),
            ("GPU Memory After Load (GB)",
             f"{baseline['gpu_after_load']['allocated_gb']:.2f}",
             f"{progressive['gpu_after_load']['allocated_gb']:.2f}",
             pct(baseline["gpu_after_load"]["allocated_gb"],
                 progressive["gpu_after_load"]["allocated_gb"])),
            ("TTFT E2E (s)",
             f"{baseline['ttft']:.4f}",
             f"{progressive['ttft']:.4f}",
             pct(baseline["ttft"], progressive["ttft"])),
        ]

        # Throughput (baseline vs last stage)
        b_tp = baseline.get("throughput_tok_per_sec", 0)
        p_tp = progressive.get("throughput_tok_per_sec", 0)
        rows.append((
            "Throughput (tok/s)",
            f"{b_tp:.2f}",
            f"{p_tp:.2f}",
            f"{'+'if p_tp > b_tp else ''}{(p_tp - b_tp) / b_tp * 100:.1f}%" if b_tp > 0 else "N/A",
        ))

        # Network
        b_net = baseline.get("network", {})
        p_net = progressive.get("network", {})
        if not b_net.get("error") and not p_net.get("error"):
            rows.append((
                "Network Transfer (GB)",
                f"{b_net.get('bytes_received_gb', 0):.2f}",
                f"{p_net.get('bytes_received_gb', 0):.2f}",
                pct(b_net.get("bytes_received_gb", 0),
                    p_net.get("bytes_received_gb", 0)),
            ))

        # Stage transition times
        if progressive.get("total_transition_time"):
            rows.append((
                "Total Transition Time (s)",
                "-",
                f"{progressive['total_transition_time']:.4f}",
                "-",
            ))

        print(f"\n{'Metric':<30} {'Baseline':<18} {'Progressive':<18} {'Improvement':<15}")
        print("-" * 81)
        for label, bval, pval, imp in rows:
            print(f"{label:<30} {bval:<18} {pval:<18} {imp:<15}")

        # Stage별 상세
        if progressive.get("stages"):
            print(f"\n{'='*90}")
            print("PROGRESSIVE STAGE DETAILS")
            print(f"{'='*90}")
            print(f"\n{'Stage':<12} {'Layers':<12} {'TTFT E2E (s)':<14} {'Throughput':<16} {'Transition':<14} {'GPU (GB)':<12}")
            print("-" * 80)
            for stage_name in sorted(progressive["stages"].keys()):
                s = progressive["stages"][stage_name]
                trans = f"{s.get('transition_time', 0):.4f}" if "transition_time" in s else "-"
                print(f"{stage_name:<12} "
                      f"{s['active_layers']:<12} "
                      f"{s['ttft']:.4f}{'':>8} "
                      f"{s['throughput_tok_per_sec']:.2f} tok/s{'':>2} "
                      f"{trans:<14} "
                      f"{s['gpu_memory']['allocated_gb']:.2f}")

            print(f"\n{'Stage Transition Breakdown (seconds)':<40}")
            print("-" * 81)
            for stage_name in sorted(progressive["stages"].keys()):
                s = progressive["stages"][stage_name]
                if "transition_time" not in s:
                    continue
                total_t = s.get("transition_time", 0.0)
                wait_t = s.get("prefetch_wait_time", 0.0)
                instant_t = s.get("instant_transition_time", total_t)
                shape_warmup_t = s.get("shape_warmup_time", 0.0)
                print(
                    f"  {s.get('transition_from', stage_name)}: "
                    f"total={total_t:.4f}s, wait={wait_t:.4f}s, "
                    f"instant={instant_t:.4f}s, shape_warmup={shape_warmup_t:.4f}s"
                )

        print("\n" + "=" * 90)
        print("KEY FINDINGS")
        print("=" * 90)
        cs_imp = (baseline["cold_start_time"] - progressive["cold_start_time"]) / baseline["cold_start_time"] * 100
        mem_imp = (baseline["gpu_after_load"]["allocated_gb"] - progressive["gpu_after_load"]["allocated_gb"]) / baseline["gpu_after_load"]["allocated_gb"] * 100
        print(f"  Cold Start Reduction:  {cs_imp:.1f}%")
        print(f"  Memory Reduction:      {mem_imp:.1f}%")
        if progressive.get("total_transition_time"):
            print(f"  Total Transition Cost: {progressive['total_transition_time']:.4f}s (wait + GPU copy + alpha)")
        for key, value in sorted(progressive.get("stage_transition_times", {}).items()):
            print(f"  {key}: {value:.4f}s")
        print("=" * 90)


# ============================================================================
# 결과 저장
# ============================================================================

def save_results(model_name: str, mode: str, data: Dict):
    """실험 결과 JSON 저장"""
    os.makedirs(RESULT_DIR, exist_ok=True)

    filename = f"02_{model_name}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(RESULT_DIR, filename)

    data["experiment_id"] = str(uuid.uuid4())
    data["hostname"] = socket.gethostname()
    data["model_name"] = model_name
    data["mode"] = mode

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def run_single_measurement(mode: str, model_name: str, interface: str,
                           baseline_path: str = None,
                           progressive_path: str = None,
                           post_transition_shape_warmup: bool = POST_TRANSITION_SHAPE_WARMUP_DEFAULT) -> Dict:
    """단일 측정 실행 (baseline 또는 progressive)"""
    if baseline_path:
        MODELS[model_name]["baseline_path"] = baseline_path
    if progressive_path:
        MODELS[model_name]["progressive_path"] = progressive_path

    bench = UniversalBenchmark(
        model_name,
        interface,
        post_transition_shape_warmup=post_transition_shape_warmup,
    )

    if mode == "baseline":
        nvtx_push("experiment_baseline")
        result = bench.measure_baseline()
        nvtx_pop()
    else:
        nvtx_push("experiment_progressive")
        result = bench.measure_progressive()
        nvtx_pop()

    return result


def run_in_subprocess(mode: str, model_name: str, interface: str,
                      baseline_path: str = None,
                      progressive_path: str = None,
                      post_transition_shape_warmup: bool = POST_TRANSITION_SHAPE_WARMUP_DEFAULT) -> Dict:
    """
    별도 subprocess에서 측정 실행.
    프로세스 종료 시 GPU 메모리가 완전히 해제됨.
    """
    # 결과를 받을 임시 파일
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        result_file = f.name

    # 같은 스크립트를 subprocess로 실행 (--_subprocess 플래그)
    cmd = [
        sys.executable, __file__,
        "--mode", mode,
        "--model", model_name,
        "--interface", interface,
        "--_subprocess", result_file,
    ]
    if baseline_path:
        cmd += ["--baseline-path", baseline_path]
    if progressive_path:
        cmd += ["--progressive-path", progressive_path]
    if post_transition_shape_warmup:
        cmd += ["--post-transition-shape-warmup"]

    # 현재 환경변수 전달 (CUDA_VISIBLE_DEVICES 등)
    env = os.environ.copy()

    print(f"\n>>> Launching subprocess for '{mode}' measurement...")
    proc = subprocess.run(cmd, env=env)

    if proc.returncode != 0:
        print(f"  Subprocess for '{mode}' exited with code {proc.returncode}")
        return {"error": True, "mode": mode}

    # 결과 읽기
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
    finally:
        os.unlink(result_file)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Universal Progressive Serving - Baseline vs Progressive Benchmark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "progressive", "both"],
        default="both",
        help="baseline: full model, progressive: staged, both: compare (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="llama",
        help="Model to benchmark (default: llama)",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=NFS_INTERFACE,
        help=f"Network interface for monitoring (default: {NFS_INTERFACE})",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Override baseline model path",
    )
    parser.add_argument(
        "--progressive-path",
        type=str,
        default=None,
        help="Override progressive model path",
    )
    parser.add_argument(
        "--post-transition-shape-warmup",
        action="store_true",
        help="Enable forced shape warmup right after stage transition (default: off)",
    )
    # 내부 subprocess 모드 (사용자가 직접 쓰지 않음)
    parser.add_argument(
        "--_subprocess",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # ── subprocess 모드: 측정만 하고 결과를 파일에 저장 후 종료 ──
    if args._subprocess:
        result = run_single_measurement(
            args.mode, args.model, args.interface,
            args.baseline_path, args.progressive_path,
            args.post_transition_shape_warmup,
        )
        with open(args._subprocess, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return

    # ── 메인 프로세스 ──
    print("\n" + "=" * 90)
    print("Universal Progressive Serving Benchmark")
    print(f"  Model:    {args.model}")
    print(f"  Mode:     {args.mode}")
    print(f"  vLLM:     {vllm_version} (expecting v0 engine)")
    print(f"  Post-transition warmup: {args.post_transition_shape_warmup}")
    print(f"  Date:     {datetime.now().isoformat()}")
    print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    experiment = {"timestamp": datetime.now().isoformat()}

    if args.mode == "both":
        # 각 측정을 별도 프로세스에서 실행 → GPU 메모리 완전 해제 보장
        experiment["baseline"] = run_in_subprocess(
            "baseline", args.model, args.interface,
            args.baseline_path, args.progressive_path,
            args.post_transition_shape_warmup,
        )
        experiment["progressive"] = run_in_subprocess(
            "progressive", args.model, args.interface,
            args.baseline_path, args.progressive_path,
            args.post_transition_shape_warmup,
        )

        # 비교 테이블
        if not experiment["baseline"].get("error") and not experiment["progressive"].get("error"):
            bench = UniversalBenchmark(args.model, args.interface)
            bench.print_comparison(experiment["baseline"], experiment["progressive"])
    else:
        # 단일 모드: 현재 프로세스에서 직접 실행
        result = run_single_measurement(
            args.mode, args.model, args.interface,
            args.baseline_path, args.progressive_path,
            args.post_transition_shape_warmup,
        )
        experiment[args.mode] = result

    # 결과 저장
    save_results(args.model, args.mode, experiment)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
