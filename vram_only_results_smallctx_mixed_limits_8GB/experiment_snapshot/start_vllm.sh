#!/usr/bin/env bash
set -euo pipefail

OPTIONS=$(getopt -o m:p:h --long model:,port:,help -- "$@") || exit 1
eval set -- "$OPTIONS"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT="${VLLM_PORT:-8000}"

while true; do
  case "$1" in
  -m | --model)
    MODEL="$2"
    shift 2
    ;;
  -p | --port)
    PORT="$2"
    shift 2
    ;;
  -h | --help)
    cat <<EOF
Usage: $0 [--model MODEL] [--port PORT]
EOF
    exit 0
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Unknown option: $1" >&2
    exit 1
    ;;
  esac
done

# ---- env overrides ----
DTYPE="${VLLM_DTYPE:-half}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
NUM_GPU_BLOCKS_OVERRIDE="${VLLM_NUM_GPU_BLOCKS_OVERRIDE:-512}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY:-fcfs}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
DISABLE_PREFIX_CACHING="${VLLM_DISABLE_PREFIX_CACHING:-1}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VENDOR_DIR="${VLLM_VENDOR_DIR:-}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"

CHUNK_FLAG=""
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  CHUNK_FLAG="--enable-chunked-prefill"
fi

PREFIX_CACHE_FLAG=""
if [[ "${DISABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
fi

BLOCK_OVERRIDE_FLAG=""
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  BLOCK_OVERRIDE_FLAG="--num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
fi

echo "[start_vllm] model=${MODEL} port=${PORT}"
echo "[start_vllm] dtype=${DTYPE}"
echo "[start_vllm] kv_cache_dtype=${KV_CACHE_DTYPE}"
echo "[start_vllm] block_size=${BLOCK_SIZE}"
echo "[start_vllm] max_model_len=${MAX_MODEL_LEN}"
echo "[start_vllm] max_num_seqs=${MAX_NUM_SEQS}"
echo "[start_vllm] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[start_vllm] num_gpu_blocks_override=${NUM_GPU_BLOCKS_OVERRIDE}"
echo "[start_vllm] cpu_offload_gb=${CPU_OFFLOAD_GB}"
echo "[start_vllm] scheduling_policy=${SCHEDULING_POLICY}"
echo "[start_vllm] chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
echo "[start_vllm] disable_prefix_caching=${DISABLE_PREFIX_CACHING}"
echo "[start_vllm] extra_args=${EXTRA_ARGS}"
echo "[start_vllm] vendor_dir=${VENDOR_DIR}"
echo "[start_vllm] executable=${VLLM_EXECUTABLE}"

if [[ -n "${VENDOR_DIR}" ]]; then
  export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "${VLLM_EXECUTABLE}" serve "${MODEL}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
  --scheduling-policy "${SCHEDULING_POLICY}" \
  ${CHUNK_FLAG} \
  ${PREFIX_CACHE_FLAG} \
  ${BLOCK_OVERRIDE_FLAG} \
  ${EXTRA_ARGS}
