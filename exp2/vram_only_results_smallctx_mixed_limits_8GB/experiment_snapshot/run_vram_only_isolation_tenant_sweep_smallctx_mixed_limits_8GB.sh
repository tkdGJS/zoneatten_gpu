#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/home/noslab-gpu/tkdgjs/tkdgjs"

SOURCE_DATASET_PATH="${ROOT_DIR}/data/ShareGPT_V3_unfiltered_cleaned_split.json"
GENERATED_DATASET_DIR="${ROOT_DIR}/data/generated"
RESULT_DIR="${ROOT_DIR}/vram_only_results_smallctx_mixed_limits_8GB"
LOG_DIR="${ROOT_DIR}/vram_only_logs_smallctx_mixed_limits_8GB"
IO_LOG_DIR="${ROOT_DIR}/vram_only_artifacts_smallctx_mixed_limits_8GB/io_logs"
METRICS_DIR="${ROOT_DIR}/vram_only_artifacts_smallctx_mixed_limits_8GB/request_metrics"
RAW_CSV="${RESULT_DIR}/result_raw.csv"
SUMMARY_CSV="${RESULT_DIR}/result_summary.csv"
VENDOR_DIR="${ROOT_DIR}/vendor"
VLLM_EXECUTABLE="${VENV_DIR}/bin/vllm"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
REUSE_GENERATED_DATASET="${REUSE_GENERATED_DATASET:-1}"
PREBUILT_DATASET_PATH="${PREBUILT_DATASET_PATH:-}"

BLOCK_VALUE="16384"
TENANT_VALUES=(32 16 8)
RUNS_PER_SETTING="1"
TURNS_PER_TENANT="10"
MIN_SESSION_USER_TURNS="10"
PRE_REQUEST_SLEEP_SEC="30"
POST_REQUEST_SLEEP_SEC="30"
INTER_TURN_SLEEP_SEC="30"
REQUEST_TIMEOUT_SEC="1500"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-24576}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SHORT_MAX_TOKENS="${SHORT_MAX_TOKENS:-512}"
LONG_MAX_TOKENS="${LONG_MAX_TOKENS:-2048}"
MIN_TOKENS="${MIN_TOKENS:-0}"
SHORT_MIN_TOKENS="${SHORT_MIN_TOKENS:-${SHORT_MAX_TOKENS}}"
LONG_MIN_TOKENS="${LONG_MIN_TOKENS:-${LONG_MAX_TOKENS}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TARGET_OUTPUT_BUDGET_TOKENS="${TARGET_OUTPUT_BUDGET_TOKENS:-${MAX_TOKENS}}"
SAFETY_MARGIN_TOKENS="${SAFETY_MARGIN_TOKENS:-64}"
SHORT_LIMIT_TOKENS="${SHORT_LIMIT_TOKENS:-12288}"
LONG_LIMIT_TOKENS="${LONG_LIMIT_TOKENS:-24576}"
SHORT_TARGET_FINAL_PROMPT_TOKENS="${SHORT_TARGET_FINAL_PROMPT_TOKENS:-11200}"
SHORT_FINAL_PROMPT_TOLERANCE_TOKENS="${SHORT_FINAL_PROMPT_TOLERANCE_TOKENS:-1000}"
LONG_TARGET_FINAL_PROMPT_TOKENS="${LONG_TARGET_FINAL_PROMPT_TOKENS:-18000}"
LONG_FINAL_PROMPT_TOLERANCE_TOKENS="${LONG_FINAL_PROMPT_TOLERANCE_TOKENS:-1000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((LONG_LIMIT_TOKENS + LONG_MAX_TOKENS))}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
PORT="8000"

source "${VENV_DIR}/bin/activate"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${IO_LOG_DIR}" "${METRICS_DIR}" "${GENERATED_DATASET_DIR}"

snapshot_experiment_context() {
  local snapshot_dir="${RESULT_DIR}/experiment_snapshot"
  mkdir -p "${snapshot_dir}"

  cp "${ROOT_DIR}/run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh" "${snapshot_dir}/"
  cp "${ROOT_DIR}/generate_mixed_history_limit_dataset.py" "${snapshot_dir}/"
  cp "${ROOT_DIR}/build_mixed_history_limit_dataset_offline.py" "${snapshot_dir}/"
  cp "${ROOT_DIR}/inspect_mixed_history_dataset.py" "${snapshot_dir}/"
  cp "${ROOT_DIR}/measure_vram_only_isolation.py" "${snapshot_dir}/"
  cp "${ROOT_DIR}/start_vllm.sh" "${snapshot_dir}/"

  python3 - <<PY
from pathlib import Path

snapshot_path = Path("${snapshot_dir}") / "settings.txt"
snapshot_path.write_text(
    "\n".join(
        [
            "source_dataset_path=${SOURCE_DATASET_PATH}",
            "generated_dataset_dir=${GENERATED_DATASET_DIR}",
            "result_dir=${RESULT_DIR}",
            "log_dir=${LOG_DIR}",
            "io_log_dir=${IO_LOG_DIR}",
            "metrics_dir=${METRICS_DIR}",
            "block_value=${BLOCK_VALUE}",
            "tenant_values=${TENANT_VALUES[*]}",
            "runs_per_setting=${RUNS_PER_SETTING}",
            "turns_per_tenant=${TURNS_PER_TENANT}",
            "min_session_user_turns=${MIN_SESSION_USER_TURNS}",
            "pre_request_sleep_sec=${PRE_REQUEST_SLEEP_SEC}",
            "post_request_sleep_sec=${POST_REQUEST_SLEEP_SEC}",
            "inter_turn_sleep_sec=${INTER_TURN_SLEEP_SEC}",
            "request_timeout_sec=${REQUEST_TIMEOUT_SEC}",
            "max_prompt_tokens=${MAX_PROMPT_TOKENS}",
            "min_tokens=${MIN_TOKENS}",
            "max_tokens=${MAX_TOKENS}",
            "short_min_tokens=${SHORT_MIN_TOKENS}",
            "short_max_tokens=${SHORT_MAX_TOKENS}",
            "long_min_tokens=${LONG_MIN_TOKENS}",
            "long_max_tokens=${LONG_MAX_TOKENS}",
            "max_num_seqs=${MAX_NUM_SEQS}",
            "target_output_budget_tokens=${TARGET_OUTPUT_BUDGET_TOKENS}",
            "safety_margin_tokens=${SAFETY_MARGIN_TOKENS}",
            "short_limit_tokens=${SHORT_LIMIT_TOKENS}",
            "long_limit_tokens=${LONG_LIMIT_TOKENS}",
            "short_target_final_prompt_tokens=${SHORT_TARGET_FINAL_PROMPT_TOKENS}",
            "short_final_prompt_tolerance_tokens=${SHORT_FINAL_PROMPT_TOLERANCE_TOKENS}",
            "long_target_final_prompt_tokens=${LONG_TARGET_FINAL_PROMPT_TOKENS}",
            "long_final_prompt_tolerance_tokens=${LONG_FINAL_PROMPT_TOLERANCE_TOKENS}",
            "vllm_max_model_len=${VLLM_MAX_MODEL_LEN}",
            "vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}",
            "port=${PORT}",
            "vllm_executable=${VLLM_EXECUTABLE}",
            "vendor_dir=${VENDOR_DIR}",
            "model_name=${MODEL_NAME}",
            "reuse_generated_dataset=${REUSE_GENERATED_DATASET}",
            "prebuilt_dataset_path=${PREBUILT_DATASET_PATH}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
PY
}

snapshot_experiment_context

is_port_open() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
rc = sock.connect_ex(("127.0.0.1", port))
sock.close()
sys.exit(0 if rc == 0 else 1)
PY
}

wait_for_server_ready() {
  local base_url="$1"
  local timeout_sec="${2:-180}"
  python3 - "${base_url}" "${timeout_sec}" <<'PY'
import sys
import time

import requests

base_url = sys.argv[1]
timeout_sec = int(sys.argv[2])
deadline = time.time() + timeout_sec
while time.time() < deadline:
    try:
        response = requests.get(base_url + "/v1/models", timeout=5)
        if response.ok:
            sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
raise SystemExit("Timed out waiting for vLLM API server readiness")
PY
}

cleanup_server() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 5
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

kill_port_owners() {
  local port="$1"
  local pids

  pids="$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -z "${pids// /}" ]]; then
    return
  fi

  echo "[INFO] killing listeners on port=${port}: ${pids}" >&2
  kill ${pids} 2>/dev/null || true
  sleep 5

  pids="$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -n "${pids// /}" ]]; then
    echo "[INFO] force killing listeners on port=${port}: ${pids}" >&2
    kill -9 ${pids} 2>/dev/null || true
  fi
}

wait_for_instance_shutdown() {
  local pid="$1"
  local port="$2"
  local timeout_sec="${3:-180}"
  local deadline=$((SECONDS + timeout_sec))

  while ((SECONDS < deadline)); do
    local pid_alive=1
    local port_open=1

    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      pid_alive=0
    fi

    if ! is_port_open "${port}"; then
      port_open=0
    fi

    if [[ "${pid_alive}" -eq 0 && "${port_open}" -eq 0 ]]; then
      return 0
    fi

    sleep 2
  done

  echo "[WARN] timed out waiting for full shutdown on port=${port}" >&2
  kill_port_owners "${port}"
  if ! is_port_open "${port}"; then
    return 0
  fi
  return 1
}

for tenant_count in "${TENANT_VALUES[@]}"; do
  for run_index in $(seq 1 "${RUNS_PER_SETTING}"); do
    base_url="http://127.0.0.1:${PORT}"
    run_log="${LOG_DIR}/block_${BLOCK_VALUE}_tenant_${tenant_count}_run_${run_index}.log"
    metrics_jsonl="${METRICS_DIR}/block_${BLOCK_VALUE}_tenant_${tenant_count}_run_${run_index}.jsonl"
    generated_dataset_path="${GENERATED_DATASET_DIR}/sharegpt_mixed_history_limits_tenant_${tenant_count}.json"

    if [[ -n "${PREBUILT_DATASET_PATH}" ]]; then
      if [[ "${PREBUILT_DATASET_PATH}" = /* ]]; then
        generated_dataset_path="${PREBUILT_DATASET_PATH}"
      else
        generated_dataset_path="${ROOT_DIR}/${PREBUILT_DATASET_PATH}"
      fi
      if [[ ! -f "${generated_dataset_path}" ]]; then
        echo "[ERROR] prebuilt dataset not found: ${generated_dataset_path}" >&2
        exit 1
      fi
      echo "[INFO] using prebuilt dataset: ${generated_dataset_path}"
    elif [[ "${REUSE_GENERATED_DATASET}" == "1" && -f "${generated_dataset_path}" ]]; then
      echo "[INFO] reusing generated dataset: ${generated_dataset_path}"
    else
      rm -f "${generated_dataset_path}"
      python "${ROOT_DIR}/build_mixed_history_limit_dataset_offline.py" \
        --model "${MODEL_NAME}" \
        --vendor-dir "${VENDOR_DIR}" \
        --dataset-path "${SOURCE_DATASET_PATH}" \
        --output-path "${generated_dataset_path}" \
        --tenant-count "${tenant_count}" \
        --turns-per-tenant "${TURNS_PER_TENANT}" \
        --short-limit-tokens "${SHORT_LIMIT_TOKENS}" \
        --long-limit-tokens "${LONG_LIMIT_TOKENS}" \
        --target-output-budget-tokens "${TARGET_OUTPUT_BUDGET_TOKENS}" \
        --safety-margin-tokens "${SAFETY_MARGIN_TOKENS}" \
        --short-target-final-prompt-tokens "${SHORT_TARGET_FINAL_PROMPT_TOKENS}" \
        --short-final-prompt-tolerance-tokens "${SHORT_FINAL_PROMPT_TOLERANCE_TOKENS}" \
        --long-target-final-prompt-tokens "${LONG_TARGET_FINAL_PROMPT_TOKENS}" \
        --long-final-prompt-tolerance-tokens "${LONG_FINAL_PROMPT_TOLERANCE_TOKENS}"
    fi

    python "${ROOT_DIR}/inspect_mixed_history_dataset.py" \
      --dataset-path "${generated_dataset_path}" \
      --show-first "${tenant_count}" | tee "${run_log}"

    echo "[RUN] num_gpu_blocks_override=${BLOCK_VALUE} run=${run_index} tenants=${tenant_count} port=${PORT}"
    rm -f "${metrics_jsonl}"
    kill_port_owners "${PORT}"

    server_pid=""
    cleanup_needed=1
    trap 'if [[ "${cleanup_needed}" -eq 1 ]]; then cleanup_server "${server_pid}"; fi' EXIT

    (
      cd "${ROOT_DIR}"
      VLLM_EXECUTABLE="${VLLM_EXECUTABLE}" \
        VLLM_VENDOR_DIR="${VENDOR_DIR}" \
        VLLM_REQUEST_METRICS_JSONL="${metrics_jsonl}" \
        VLLM_NUM_GPU_BLOCKS_OVERRIDE="${BLOCK_VALUE}" \
        VLLM_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        VLLM_DISABLE_PREFIX_CACHING=0 \
        VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
        VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN" \
        "${ROOT_DIR}/start_vllm.sh" --port "${PORT}"
    ) >>"${run_log}" 2>&1 &
    server_pid="$!"

    wait_for_server_ready "${base_url}"

    cp "${generated_dataset_path}" "${RESULT_DIR}/experiment_snapshot/tenant_${tenant_count}_run_${run_index}_dataset.json"

    if ! python "${ROOT_DIR}/measure_vram_only_isolation.py" \
      --base-url "${base_url}" \
      --dataset-path "${generated_dataset_path}" \
      --raw-csv "${RAW_CSV}" \
      --summary-csv "${SUMMARY_CSV}" \
      --run-log "${run_log}" \
      --io-log-dir "${IO_LOG_DIR}" \
      --metrics-jsonl "${metrics_jsonl}" \
      --tenant-count "${tenant_count}" \
      --run-index "${run_index}" \
      --num-gpu-blocks-override "${BLOCK_VALUE}" \
      --turns-per-tenant "${TURNS_PER_TENANT}" \
      --min-session-user-turns "${MIN_SESSION_USER_TURNS}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
      --min-tokens "${MIN_TOKENS}" \
      --max-tokens "${MAX_TOKENS}" \
      --short-limit-tokens "${SHORT_LIMIT_TOKENS}" \
      --long-limit-tokens "${LONG_LIMIT_TOKENS}" \
      --short-min-tokens "${SHORT_MIN_TOKENS}" \
      --short-max-tokens "${SHORT_MAX_TOKENS}" \
      --long-min-tokens "${LONG_MIN_TOKENS}" \
      --long-max-tokens "${LONG_MAX_TOKENS}" \
      --pre-request-sleep-sec "${PRE_REQUEST_SLEEP_SEC}" \
      --inter-turn-sleep-sec "${INTER_TURN_SLEEP_SEC}" \
      --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"; then
      echo "[FAIL] num_gpu_blocks_override=${BLOCK_VALUE} run=${run_index} tenants=${tenant_count}" >&2
    fi

    sleep "${POST_REQUEST_SLEEP_SEC}"
    cleanup_server "${server_pid}"
    echo "[INFO] num_gpu_blocks_override=${BLOCK_VALUE} run=${run_index}: waiting for full shutdown on port=${PORT}"
    wait_for_instance_shutdown "${server_pid}" "${PORT}"
    cleanup_needed=0
    trap - EXIT
  done
done

echo "[DONE] block=${BLOCK_VALUE} tenant_sweep=${TENANT_VALUES[*]} raw_csv=${RAW_CSV} summary_csv=${SUMMARY_CSV}"
