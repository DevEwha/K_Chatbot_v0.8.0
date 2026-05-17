#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${SCRIPT_DIR}/benchmark_scaling_true_e2e_origin_first.py"
ARTIFACT_PY="${SCRIPT_DIR}/build_kv_paper_artifacts.py"

MODELS_CSV="${MODELS_CSV:-llama2-7b,llama2-13b,falcon-7b,gemma-7b}"
STAGES_CSV="${STAGES_CSV:-1to2,2to3}"
ARTIFACT_STAGES_CSV="${ARTIFACT_STAGES_CSV:-${STAGES_CSV}}"
SUMMARY_MODELS_CSV="${SUMMARY_MODELS_CSV:-${MODELS_CSV}}"
REPRESENTATIVE_MODEL="${REPRESENTATIVE_MODEL:-llama2-7b}"
PROGRESSIVE_IMPL="${PROGRESSIVE_IMPL:-progressive_serve5}"
CACHE_CONDITION="${CACHE_CONDITION:-cold}"
NUM_RUNS="${NUM_RUNS:-5}"
STAGE1_WARMUP_RUNS="${STAGE1_WARMUP_RUNS:-3}"
STAGE_TRANSITION_TIMEOUT_S="${STAGE_TRANSITION_TIMEOUT_S:-300}"
DROP_CACHES_TIMEOUT_S="${DROP_CACHES_TIMEOUT_S:-60}"
TOKEN_COUNTS_CSV="${TOKEN_COUNTS_CSV:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ALLOW_PARTIAL_FALLBACK="${ALLOW_PARTIAL_FALLBACK:-0}"
NO_PLOT="${NO_PLOT:-0}"
BUILD_ARTIFACTS="${BUILD_ARTIFACTS:-1}"
CACHE_DROP_PAUSE_S="${CACHE_DROP_PAUSE_S:-2}"
SLEEP_BETWEEN_JOBS_S="${SLEEP_BETWEEN_JOBS_S:-5}"
SUDO_KEEPALIVE_ENABLED="${SUDO_KEEPALIVE_ENABLED:-1}"
SUDO_KEEPALIVE_INTERVAL_S="${SUDO_KEEPALIVE_INTERVAL_S:-60}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

LOG_ROOT="${SCRIPT_DIR}/results/asplos_truee2e_originfirst_logs/${RUN_TAG}"
MASTER_LOG="${LOG_ROOT}/run.log"
MANIFEST_TSV="${LOG_ROOT}/manifest.tsv"
SUDO_KEEPALIVE_PID=""

if [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

mkdir -p "${LOG_ROOT}"

log() {
  local ts
  ts="$(date '+%F %T')"
  echo "[${ts}] $*" | tee -a "${MASTER_LOG}"
}

die() {
  log "ERROR: $*"
  exit 1
}

cleanup() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]]; then
    kill "${SUDO_KEEPALIVE_PID}" >/dev/null 2>&1 || true
    wait "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true
    SUDO_KEEPALIVE_PID=""
  fi
}

trap cleanup EXIT INT TERM

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  local raw=()
  local item
  IFS=',' read -r -a raw <<< "${csv}"
  out_ref=()
  for item in "${raw[@]}"; do
    item="$(trim "${item}")"
    [[ -n "${item}" ]] && out_ref+=("${item}")
  done
}

run_and_log() {
  local log_path="$1"
  shift
  log "CMD: $*"
  "$@" 2>&1 | tee "${log_path}"
}

log_gpu_snapshot() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  log "GPU snapshot:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
    | sed 's/^/  /' \
    | tee -a "${MASTER_LOG}" >/dev/null
}

drop_os_page_cache() {
  local reason="${1:-manual}"
  local rc=0
  if [[ "${CACHE_CONDITION}" != "cold" ]]; then
    log "[cache] skip drop (${reason}) because CACHE_CONDITION=${CACHE_CONDITION}"
    return 0
  fi

  log "[cache] dropping OS page cache (${reason})"
  if [[ "$(id -u)" -eq 0 ]]; then
    sync
    sh -c 'echo 3 > /proc/sys/vm/drop_caches' || rc=$?
  else
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || rc=$?
  fi
  if [[ "${rc}" -ne 0 ]]; then
    log "[cache] drop_caches failed (${reason}, rc=${rc}). sudo credential may have expired; refresh with 'sudo -v' or let the launcher acquire it up front."
    return "${rc}"
  fi
  sleep "${CACHE_DROP_PAUSE_S}"
}

start_sudo_keepalive() {
  local parent_pid="$$"
  if [[ "${CACHE_CONDITION}" != "cold" ]]; then
    return 0
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    return 0
  fi
  if [[ "${SUDO_KEEPALIVE_ENABLED}" != "1" ]]; then
    log "[cache] sudo keepalive disabled (SUDO_KEEPALIVE_ENABLED=${SUDO_KEEPALIVE_ENABLED})"
    return 0
  fi
  command -v sudo >/dev/null 2>&1 || die "sudo is required for cold-cache execution."

  if ! sudo -n -v >/dev/null 2>&1; then
    log "[cache] acquiring sudo credential for cold-cache drops; you may be prompted once."
    sudo -v || return 1
  fi

  (
    while kill -0 "${parent_pid}" >/dev/null 2>&1; do
      if ! sudo -n -v >/dev/null 2>&1; then
        printf '[%s] [cache] sudo keepalive refresh failed; later cold-cache drops may fail.\n' \
          "$(date '+%F %T')" >> "${MASTER_LOG}"
        exit 0
      fi
      sleep "${SUDO_KEEPALIVE_INTERVAL_S}" || exit 0
    done
  ) &
  SUDO_KEEPALIVE_PID=$!
  log "[cache] sudo keepalive active (interval=${SUDO_KEEPALIVE_INTERVAL_S}s, pid=${SUDO_KEEPALIVE_PID})"
}

append_manifest_header() {
  {
    echo -e "status\tmodel\tstage\tjson\tpng\tlog"
  } > "${MANIFEST_TSV}"
}

append_manifest_row() {
  local status="$1"
  local model="$2"
  local stage="$3"
  local json_out="$4"
  local png_out="$5"
  local job_log="$6"
  {
    echo -e "${status}\t${model}\t${stage}\t${json_out}\t${png_out}\t${job_log}"
  } >> "${MANIFEST_TSV}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Missing required file: ${path}"
}

preflight() {
  require_file "${BENCHMARK_PY}"
  if [[ "${BUILD_ARTIFACTS}" == "1" ]]; then
    require_file "${ARTIFACT_PY}"
  fi

  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

  if [[ "${CACHE_CONDITION}" == "cold" ]]; then
    if ! start_sudo_keepalive; then
      die "Cold-cache execution requires sudo permission. Run from an interactive terminal or pre-authorize with 'sudo -v'."
    fi
    if ! drop_os_page_cache "preflight"; then
      die "Cold-cache execution requires working drop_caches permission. If the sudo credential expired, refresh with 'sudo -v' and rerun."
    fi
  fi
}

build_job_command() {
  local model="$1"
  local stage="$2"
  local output_json="$3"
  local -n cmd_ref="$4"

  cmd_ref=(
    "${PYTHON_BIN}"
    "${BENCHMARK_PY}"
      "--model" "${model}"
      "--stage" "${stage}"
      "--num-runs" "${NUM_RUNS}"
      "--cache-condition" "${CACHE_CONDITION}"
      "--drop-caches-timeout-s" "${DROP_CACHES_TIMEOUT_S}"
      "--stage1-warmup-runs" "${STAGE1_WARMUP_RUNS}"
      "--stage-transition-timeout-s" "${STAGE_TRANSITION_TIMEOUT_S}"
      "--progressive-impl" "${PROGRESSIVE_IMPL}"
      "--output" "${output_json}"
  )

  if [[ -n "${TOKEN_COUNTS_CSV}" ]]; then
    cmd_ref+=("--token-counts" "${TOKEN_COUNTS_CSV}")
  fi
  if [[ "${GPU_MEMORY_UTILIZATION}" != "0" ]]; then
    cmd_ref+=("--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}")
  fi
  if [[ "${TENSOR_PARALLEL_SIZE}" != "0" ]]; then
    cmd_ref+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
  fi
  if [[ "${MAX_MODEL_LEN}" != "0" ]]; then
    cmd_ref+=("--max-model-len" "${MAX_MODEL_LEN}")
  fi
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cmd_ref+=("--enforce-eager")
  fi
  if [[ "${ALLOW_PARTIAL_FALLBACK}" == "1" ]]; then
    cmd_ref+=("--allow-partial-fallback")
  fi
  if [[ "${NO_PLOT}" == "1" ]]; then
    cmd_ref+=("--no-plot")
  fi
}

run_benchmark_job() {
  local model="$1"
  local stage="$2"
  local index="$3"
  local output_json="${SCRIPT_DIR}/results_scaling_truee2e_originfirst_${model}_${stage}_${RUN_TAG}.json"
  local output_png="${output_json%.json}.png"
  local job_log="${LOG_ROOT}/$(printf '%02d' "${index}")_${model}_${stage}.log"
  local cmd=()

  build_job_command "${model}" "${stage}" "${output_json}" cmd

  drop_os_page_cache "before ${model}/${stage}"
  log "===== [${index}] ${model} ${stage} start ====="
  log_gpu_snapshot

  if run_and_log "${job_log}" "${cmd[@]}"; then
    [[ -f "${output_json}" ]] || die "Benchmark finished without JSON output: ${output_json}"
    if [[ "${NO_PLOT}" != "1" && ! -f "${output_png}" ]]; then
      die "Benchmark finished without PNG output: ${output_png}"
    fi
    append_manifest_row "ok" "${model}" "${stage}" "${output_json}" "${output_png}" "${job_log}"
    log "===== [${index}] ${model} ${stage} done ====="
  else
    append_manifest_row "failed" "${model}" "${stage}" "${output_json}" "${output_png}" "${job_log}"
    die "Benchmark failed for ${model} ${stage}. Check ${job_log}"
  fi

  drop_os_page_cache "after ${model}/${stage}"
  sleep "${SLEEP_BETWEEN_JOBS_S}"
}

run_artifact_builds() {
  local artifact_stages=()
  csv_to_array "${ARTIFACT_STAGES_CSV}" artifact_stages
  if [[ "${BUILD_ARTIFACTS}" != "1" || ${#artifact_stages[@]} -eq 0 ]]; then
    return 0
  fi

  local stage
  for stage in "${artifact_stages[@]}"; do
    local out_dir="${SCRIPT_DIR}/figures/asplos_truee2e_originfirst_${stage}_${RUN_TAG}"
    local art_log="${LOG_ROOT}/artifacts_${stage}.log"
    mkdir -p "${out_dir}"
    log "===== artifact build start (stage=${stage}) ====="
    if ! run_and_log "${art_log}" \
      "${PYTHON_BIN}" "${ARTIFACT_PY}" \
      "--results-dir" "${SCRIPT_DIR}" \
      "--output-dir" "${out_dir}" \
      "--representative-model" "${REPRESENTATIVE_MODEL}" \
      "--stage" "${stage}" \
      "--summary-models" "${SUMMARY_MODELS_CSV}"; then
      die "Artifact build failed for stage=${stage}. Check ${art_log}"
    fi
    log "===== artifact build done (stage=${stage}) ====="
  done
}

main() {
  local models=()
  local stages=()
  local model
  local stage
  local job_index=1

  csv_to_array "${MODELS_CSV}" models
  csv_to_array "${STAGES_CSV}" stages

  [[ ${#models[@]} -gt 0 ]] || die "No models resolved from MODELS_CSV=${MODELS_CSV}"
  [[ ${#stages[@]} -gt 0 ]] || die "No stages resolved from STAGES_CSV=${STAGES_CSV}"

  append_manifest_header
  preflight

  log "Overnight ASPLOS true-E2E origin-first run started"
  log "RUN_TAG=${RUN_TAG}"
  log "PYTHON_BIN=${PYTHON_BIN}"
  log "BENCHMARK_PY=${BENCHMARK_PY}"
  log "MODELS=${MODELS_CSV}"
  log "STAGES=${STAGES_CSV}"
  log "CACHE_CONDITION=${CACHE_CONDITION}"
  log "DROP_CACHES_TIMEOUT_S=${DROP_CACHES_TIMEOUT_S}"
  log "NUM_RUNS=${NUM_RUNS}"
  log "PROGRESSIVE_IMPL=${PROGRESSIVE_IMPL}"
  log "LOG_ROOT=${LOG_ROOT}"

  for model in "${models[@]}"; do
    for stage in "${stages[@]}"; do
      run_benchmark_job "${model}" "${stage}" "${job_index}"
      job_index=$((job_index + 1))
    done
  done

  run_artifact_builds

  log "All jobs completed successfully."
  log "Manifest: ${MANIFEST_TSV}"
}

main "$@"
