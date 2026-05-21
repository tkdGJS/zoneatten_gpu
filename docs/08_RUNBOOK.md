# 08. Runbook

## 1. Environment Setup

vLLM upstream 개발 문서 기준으로 Python-only 개발은 editable install을 사용할 수 있다.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

CUDA/C++ 코드까지 수정하는 경우:

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

## 2. Linting

```bash
uv pip install 'pre-commit>=4.5.1'
pre-commit install

pre-commit run
pre-commit run -a
```

## 3. Tests

```bash
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto
uv pip install pytest pytest-asyncio

pytest tests/
pytest -s -v tests/test_logger.py
```

프로젝트 관련 테스트 예시:

```bash
pytest -q tests/v1/core/sched
pytest -q tests -k "scheduler or kv_cache or metrics"
python -m compileall vllm
```

## 4. Smoke Test

```bash
python - <<'PY'
import vllm
print("vllm", getattr(vllm, "__version__", "unknown"))
PY
```

## 5. Run vLLM Server

예시:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64
```

큰 모델/Codex backend 예시는 환경에 맞게 조정한다.

```bash
vllm serve Qwen/Qwen3.6-27B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

## 6. Test API

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "metadata": {"tenant_id": "tenant_a"}
  }'
```

## 7. Codex with vLLM

`~/.codex/config.toml` 예시:

```toml
model = "my-model"
model_provider = "vllm"

[model_providers.vllm]
name = "vLLM"
env_key = "VLLM_API_KEY"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

```bash
export VLLM_API_KEY=dummy
codex
```

## 8. Before Every Patch

```bash
git status --short
git branch --show-current
rg "class Scheduler|def schedule|KVCacheManager|SchedulerConfig" vllm tests
```

## 9. After Every Patch

```bash
python -m compileall vllm
pytest -q <relevant-tests>
pre-commit run --files <changed-files>
```

Then update:

```text
docs/06_PATCH_NOTES.md
docs/07_TROUBLESHOOTING.md, if new failure pattern exists
docs/15_DECISIONS.md, if design choice was made
```

## 10. Result Directory Convention

```text
results/
  YYYYMMDD_experiment_name/
    config.json
    workload.json
    raw_logs.jsonl
    metrics.csv
    summary.md
    figures/
```

Each result must include:

- git commit hash
- vLLM version
- GPU name/count
- model name
- vLLM CLI flags
- workload seed
- policy flags
