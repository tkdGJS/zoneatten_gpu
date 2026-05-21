# vLLM Multi-Tenant / Multi-Turn KV Cache Pressure Experiment Report

생성일: 2026-05-18  
결과 디렉토리: `vram_only_results_smallctx_mixed_limits_8GB`

## 1. 실험 목적

이 실험의 목적은 vLLM 기반 LLM serving 환경에서 multi-tenant, multi-turn workload가 누적될 때 KV cache budget pressure가 어떤 형태로 성능에 나타나는지 측정하는 것이다.

핵심 질문은 다음과 같다.

- tenant 수가 `8 -> 16 -> 32`로 증가할 때, multi-turn history 누적으로 인해 TTFT, blocking time, prefill time, prefix hit rate가 어떻게 변하는가?
- 동일한 8GiB 수준의 강제 KV capacity에서 Group1(long history)과 Group2(short history)가 서로 다른 KV history 압력을 만드는가?
- KV budget pressure가 단순 prefill 증가로 나타나는지, scheduler queue blocking으로 나타나는지, 또는 prefix cache hit 저하로 나타나는지 확인한다.
- turn이 진행될수록 tenant별 history가 누적될 때 32 tenants 구간에서 지연이 급격히 증가하는지 확인한다.

## 2. 실험 환경

| 항목 | 설정 |
|---|---:|
| GPU | NVIDIA Tesla T4 |
| GPU VRAM | 16GB |
| LLM serving engine | vLLM |
| vLLM source | `vendor/vllm` patched source 우선 사용 |
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| API server | OpenAI-compatible vLLM server |
| Base URL | `http://127.0.0.1:8000` |
| Attention backend | `TRITON_ATTN` |
| FlashInfer 사용 여부 | 사용하지 않음 |
| 결과 반복 수 | 3회, exp1/exp2/exp3 통합 |

FlashInfer backend는 서버 시작 시 JIT compilation 과정에서 `nvcc`를 요구했고, 해당 환경에 CUDA toolkit 경로가 없어 실패했다. 따라서 실행 스크립트에서 `--attention-backend TRITON_ATTN`을 사용하도록 설정했다.

## 3. 강제 KV Capacity 설정

vLLM의 KV cache block 수를 강제로 제한하여 실제 GPU 16GB 전체가 아니라 약 8GiB 수준의 KV capacity를 사용하는 상황을 만들었다.

| 항목 | 값 |
|---|---:|
| `num_gpu_blocks_override` | 16384 |
| `block_size` | 16 tokens |
| 총 KV token capacity | 16384 x 16 = 262,144 tokens |
| KV capacity 환산 가정 | 32,768 tokens ~= 1GiB |
| 강제 KV capacity | 약 8GiB |

즉, 실험의 핵심 resource limit은 다음과 같다.

```text
KV capacity tokens = 16384 blocks * 16 tokens/block = 262144 tokens
262144 tokens / 32768 tokens per GiB = 8 GiB
```

이 값은 vLLM allocator가 사용할 수 있는 KV block budget을 제한하기 위한 실험적 설정이다. 실제 GPU VRAM 16GB에는 모델 weight, runtime buffer, activation, CUDA context 등이 함께 존재하므로, 이 실험에서 말하는 8GiB는 전체 VRAM 사용량이 아니라 강제로 제한한 KV cache capacity 기준이다.

## 4. vLLM 실행 옵션

현재 실험 스크립트는 `run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh`이다.

주요 옵션은 다음과 같다.

| 옵션 | 값 |
|---|---:|
| `BLOCK_VALUE` | 16384 |
| `TENANT_VALUES` | 32, 16, 8 |
| `TURNS_PER_TENANT` | 10 |
| `MAX_NUM_SEQS` | 32 |
| `VLLM_MAX_MODEL_LEN` | 26624 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 16384 |
| `INTER_TURN_SLEEP_SEC` | 30 |
| `REQUEST_TIMEOUT_SEC` | 1500 |
| `VLLM_EXTRA_ARGS` | `--attention-backend TRITON_ATTN` |
| patched metrics | `VLLM_REQUEST_METRICS_JSONL` 사용 |

`VLLM_MAX_MODEL_LEN=26624`는 long group의 history limit과 output budget을 기준으로 계산된다.

```text
LONG_LIMIT_TOKENS + LONG_MAX_TOKENS = 24576 + 2048 = 26624
```

prefix caching은 비활성화하지 않았으며, patched vLLM이 request별 JSONL metrics를 생성하도록 `VLLM_REQUEST_METRICS_JSONL` 환경변수를 전달한다.

## 5. Workload 및 데이터셋 구성

원본 데이터셋은 ShareGPT 기반 데이터셋이다.

```text
data/ShareGPT_V3_unfiltered_cleaned_split.json
```

실험용 workload는 multi-turn history 누적을 의도적으로 만들기 위해 tenant별 10 turns로 구성했다. 각 turn은 tenant들이 동기적으로 요청을 보내고, 모든 요청이 완료된 뒤 30초 대기 후 다음 turn으로 진행한다.

### Tenant sweep

| tenant count | 목적 |
|---:|---|
| 8 | 낮은 동시성 기준선 |
| 16 | 중간 동시성 및 초기 pressure 확인 |
| 32 | 8GiB KV capacity에서 강한 queue/KV pressure 유도 |

### Group 구성

Group1과 Group2는 history limit과 output token budget을 다르게 설정했다. 입력 prompt만으로는 원본 ShareGPT 데이터에서 충분한 group separation을 만들기 어려웠기 때문에, output token을 의도적으로 다르게 설정해 turn이 진행될수록 KV history가 서로 다르게 증가하도록 설계했다.

| 그룹 | 의미 | `history_limit_tokens` | output tokens |
|---|---|---:|---:|
| Group1 | long history group | 24576 | min=max 2048 |
| Group2 | short history group | 12288 | min=max 512 |

각 group의 output token은 최소값과 최대값을 동일하게 두었다. 이 설정은 모델이 너무 짧게 답변하면서 KV history 증가량이 기대보다 작아지는 문제를 방지하기 위한 것이다.

```text
SHORT_MAX_TOKENS=512
SHORT_MIN_TOKENS=512
LONG_MAX_TOKENS=2048
LONG_MIN_TOKENS=2048
```

## 6. Turn 진행 방식

초기에는 fixed period 방식으로 turn을 진행했으나, 일부 turn에서 요청 완료 시간이 240초를 넘으면서 timeout처럼 보이는 문제가 있었다. 현재 방식은 다음과 같이 수정되어 있다.

1. 현재 turn의 모든 tenant request를 동시에 발행한다.
2. 모든 request가 완료될 때까지 기다린다.
3. 완료 후 30초 대기한다.
4. 다음 turn을 시작한다.

이 방식은 실제 workload pressure는 유지하면서도, 느린 turn 때문에 다음 turn이 겹쳐 들어가거나 measurement timeout이 발생하는 문제를 줄인다.

## 7. 계측 방식

결과 CSV와 patched vLLM request metrics JSONL을 함께 사용한다.

### Raw result

```text
vram_only_results_smallctx_mixed_limits_8GB/result_raw.csv
```

이 파일에는 tenant, turn, input tokens, output tokens, TTFT, TTLT, TBT, blocking time, prefix hit 관련 값이 포함된다.

### Patched vLLM metrics

patched vLLM은 request마다 JSONL metrics를 생성한다. 주요 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `request_id` | request 식별자 |
| `queued_time_s` | scheduler queue에서 대기한 시간 |
| `prefill_time_s` | scheduling 이후 first token까지의 prefill 시간 |
| `decode_time_s` | decode 구간 시간 |
| `ttft_s` | server-side TTFT |
| `ttlt_s` | server-side TTLT |
| `prefix_hit_tokens` | prefix cache hit token 수 |
| `prompt_tokens` | prompt token 수 |
| `completion_tokens` | completion token 수 |

그래프에서는 TTFT를 다음 component로 분해한다.

```text
TTFT ~= blocking time + prefill time + remaining TTFT
```

여기서 remaining TTFT는 client 관측 TTFT에서 patched vLLM이 제공하는 blocking/prefill component를 제외한 나머지 시간이다. 이 값에는 network overhead, Python client overhead, streaming 처리 지연, measurement gap 등이 포함될 수 있다.

## 8. 실행 결과 요약

3회 반복 실험 결과를 모두 합쳐 분석했다.

| 항목 | 값 |
|---|---:|
| 총 request 수 | 1680 |
| 성공 request 수 | 1680 |
| 실패 request 수 | 0 |
| run index | 1, 2, 3 |
| tenant count | 8, 16, 32 |
| turn range | 1-10 |
| group limits | 12288, 24576 |

모든 tenant count에서 turn 10까지 도달했고, 전체 request가 success 상태로 기록되었다.

## 9. Group별 평균 결과

아래 표는 exp1/exp2/exp3 전체를 통합한 평균이다.

| tenants | group | limit | avg input tokens | avg output tokens | avg KV history tokens | avg TTFT ms | avg blocking ms | avg prefix hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | Group2 | 12288 | 3247.40 | 513.00 | 3095.57 | 11403.16 | 0.04 | 0.7608 |
| 8 | Group1 | 24576 | 10313.83 | 2048.94 | 10147.01 | 14014.58 | 0.05 | 0.7277 |
| 16 | Group2 | 12288 | 3119.40 | 513.00 | 2971.37 | 22649.71 | 180.90 | 0.7353 |
| 16 | Group1 | 24576 | 10236.57 | 2048.97 | 10077.20 | 25871.03 | 23.23 | 0.7193 |
| 32 | Group2 | 12288 | 3111.05 | 512.99 | 2964.49 | 83477.93 | 56838.98 | 0.3775 |
| 32 | Group1 | 24576 | 10133.82 | 2048.98 | 9980.12 | 103908.20 | 66560.61 | 0.5135 |

관찰 결과는 명확하다.

- 8 tenants에서는 blocking time이 사실상 0에 가깝다.
- 16 tenants에서는 TTFT가 증가하지만 blocking time은 아직 제한적이다.
- 32 tenants에서는 blocking time이 TTFT의 큰 비중을 차지하며, scheduler queue pressure가 지배적으로 나타난다.
- Group1은 output token을 2048로 고정했기 때문에 Group2보다 KV history가 빠르게 증가하고, 평균 input/KV history가 크게 분리된다.

## 10. Turn 10 결과

turn 10은 multi-turn history가 가장 많이 누적된 상태이므로 pressure를 확인하기에 가장 중요한 구간이다.

| tenants | group | limit | turn10 avg input tokens | turn10 avg KV history tokens | turn10 avg TTFT ms | turn10 avg blocking ms |
|---:|---|---:|---:|---:|---:|---:|
| 8 | Group2 | 12288 | 6142.67 | 6064.17 | 21220.04 | 0.06 |
| 8 | Group1 | 24576 | 20116.50 | 20076.83 | 28020.65 | 0.08 |
| 16 | Group2 | 12288 | 6104.92 | 6009.92 | 45474.14 | 174.14 |
| 16 | Group1 | 24576 | 20041.79 | 19982.67 | 51342.74 | 29.89 |
| 32 | Group2 | 12288 | 6090.94 | 5949.96 | 317604.21 | 277872.68 |
| 32 | Group1 | 24576 | 19984.96 | 19907.90 | 390309.72 | 296854.53 |

turn 10 기준으로 Group1과 Group2는 input/KV history 측면에서 분리된다.

- Group1 turn10 input은 약 20K tokens이다.
- Group2 turn10 input은 약 6.1K tokens이다.
- 32 tenants에서는 두 group 모두 매우 큰 blocking time을 보이며, long group은 TTFT가 약 390초까지 증가한다.

## 11. 32 Tenants TTFT Component 분석

32 tenants의 p99 component sum을 보면 blocking time이 turn 후반부에서 압도적인 비중을 차지한다.

| group | sum p99 blocking ms | sum p99 prefill ms | sum p99 remaining ms | sum p99 total component ms |
|---|---:|---:|---:|---:|
| Group1, limit=24576 | 2775840.62 | 1100531.34 | 94970.89 | 3971342.84 |
| Group2, limit=12288 | 2775549.02 | 433169.32 | 81133.58 | 3289851.92 |

해석은 다음과 같다.

- Group1은 긴 history와 2048 output budget 때문에 prefill component가 Group2보다 훨씬 크다.
- blocking component는 Group1/Group2 모두 매우 크다. 이는 32 tenants 조건에서 전체 shared serving system이 scheduler/KV budget pressure를 강하게 받고 있음을 의미한다.
- remaining TTFT는 존재하지만 blocking과 prefill에 비해 상대적으로 작다.

## 12. 결과 해석

### 12.1 Tenant 증가에 따른 변화

8 tenants에서는 KV budget pressure가 거의 드러나지 않는다. TTFT는 group별 input length와 prefill cost에 의해 주로 결정되고, blocking time은 거의 0이다.

16 tenants에서는 TTFT가 약 2배 수준으로 증가하지만, blocking time은 아직 수백 ms 수준이다. 이 구간은 GPU compute와 batching 영향이 커지기 시작하는 중간 부하 구간으로 볼 수 있다.

32 tenants에서는 동작 양상이 바뀐다. 평균 blocking time이 수십 초 수준으로 증가하고, turn 10에서는 blocking time이 수백 초까지 증가한다. 이는 단순히 긴 prompt prefill이 느린 것이 아니라, 요청이 scheduler queue에서 오래 대기하는 상태가 되었음을 보여준다.

### 12.2 Group1/Group2 분리

원본 ShareGPT prompt만으로는 10 turns에서 각 group이 history limit에 충분히 가깝게 도달하지 못했다. 따라서 output token budget을 다르게 주어 KV history 증가량을 통제했다.

현재 결과에서는 turn 10 기준으로 Group1과 Group2의 input/KV history가 뚜렷하게 분리된다.

- Group1: 약 20K input tokens
- Group2: 약 6K input tokens

이 구성은 Group1이 더 큰 KV history pressure를 만들고, Group2는 상대적으로 짧은 history를 유지하는 비교군 역할을 하도록 만든다.

### 12.3 Prefix hit rate 변화

평균 prefix hit rate는 tenant 수가 증가할수록 감소한다.

- 8 tenants: Group1 0.7277, Group2 0.7608
- 16 tenants: Group1 0.7193, Group2 0.7353
- 32 tenants: Group1 0.5135, Group2 0.3775

32 tenants에서 prefix hit rate가 크게 낮아지는 것은 KV cache pressure와 eviction/재사용 실패 가능성이 커졌음을 시사한다. 특히 Group2는 짧은 history group임에도 shared pressure의 영향을 받아 prefix hit rate가 크게 떨어졌다.

### 12.4 Blocking이 주요 병목인지 여부

32 tenants turn 후반부에서는 blocking time이 dominant component로 나타난다. 이는 요청이 이미 scheduling되어 decoding 중에 OOM으로 죽는 상황보다는, scheduler가 요청을 즉시 실행하지 못하고 queue에서 오래 대기시키는 현상에 가깝다.

다만 이 해석은 patched vLLM metrics의 `queued_time_s`를 blocking time으로 사용하는 방식에 기반한다. 실제 allocator 내부 block allocation 실패, preemption, eviction, recomputation이 어느 정도 발생했는지는 vLLM 내부 trace를 추가로 계측해야 더 정확히 확인할 수 있다.

## 13. 생성된 그래프

그래프는 다음 디렉토리에 생성되어 있다.

```text
vram_only_results_smallctx_mixed_limits_8GB/graphs
```

현재 그래프 산출물은 총 311개 파일이며, 그중 PNG 그래프는 256개이다.

주요 그래프는 다음과 같다.

| 그래프 | 경로 |
|---|---|
| 32 tenants TTFT breakdown by group/turn | `graphs/generate_selected_paper_graphs/paper_ttft_breakdown_group_p99_by_turn_32tenants.png` |
| 32 tenants blocking+prefill p99 sum by turn | `graphs/generate_selected_paper_graphs/paper_p99_blocking_prefill_sum_by_turn_32tenants.png` |
| Group1/2 component total stacked bar | `graphs/generate_selected_paper_graphs/paper_p99_component_totals_by_group_32tenants.png` |
| TTFT vs blocking/prefill vs compute ratio | `graphs/generate_selected_paper_graphs/paper_ttft_vs_blocking_and_prefill_vs_compute_ratio.png` |
| TTFT vs prefix hit rate | `graphs/generate_selected_paper_graphs/paper_ttft_vs_prefix_hit_rate_by_tenantcount.png` |
| KV usage by turn, 32 tenants | `graphs/generate_kv_usage_by_turn_same_timing/exp123_mean_kv_usage_by_turn_32tenants.png` |
| input KV usage by turn, 32 tenants | `graphs/generate_input_kv_usage_by_turn_same_timing/exp123_mean_input_kv_usage_by_turn_32tenants.png` |
| TTFT breakdown timeline, 32 tenants | `graphs/generate_ttft_breakdown_timeline_same_timing/exp123_ttft_breakdown_group_p99_by_turn_32tenants.png` |
| turn 5/7 root cause analysis | `graphs/generate_turn57_rootcause_analysis_same_timing/turn57_rootcause_analysis.md` |

## 14. 재현 방법

기본 실행 명령은 다음과 같다.

```bash
bash ./run_vram_only_isolation_tenant_sweep_smallctx_mixed_limits_8GB.sh
```

현재 스크립트는 `vendor/vllm`을 `PYTHONPATH` 앞에 두도록 설정되어 있으므로, patched vLLM source가 사용되어야 한다.

확인 명령:

```bash
python - <<'PY'
import vllm
import vllm.v1.engine.output_processor as op
print(vllm.__file__)
print(op.__file__)
PY
```

기대 경로:

```text
/home/yuhwa2323/zoneatten_gpu/vendor/vllm/__init__.py
/home/yuhwa2323/zoneatten_gpu/vendor/vllm/v1/engine/output_processor.py
```

## 15. 한계 및 주의점

- 이 실험의 8GiB는 실제 GPU VRAM 전체 사용량이 아니라 vLLM KV block 수를 제한해 만든 KV cache capacity 기준이다.
- physical GPU는 Tesla T4 16GB이지만, 모델 weight와 runtime overhead가 함께 존재한다.
- Group1/Group2 분리는 원본 ShareGPT prompt 길이만으로 만든 것이 아니라 output token budget을 조절해 만든 것이다.
- graph compatibility layer 때문에 일부 legacy graph script 내부에서는 limit을 `8192/2048`으로 매핑해 처리하지만, 최종 selected paper graph 표시는 `24576/12288`로 복원되어 있다.
- blocking time은 patched vLLM의 `queued_time_s`를 사용한다. vLLM 내부 allocator event, preemption, eviction까지 완전히 설명하려면 추가 계측이 필요하다.
- TBT가 tenant 8/16에서 거의 동일하게 보이는 현상은 낮은 pressure 구간에서 decode가 비슷한 batch cadence로 진행되기 때문일 가능성이 크다. 이 실험의 주요 병목 분석 대상은 TBT보다 TTFT, blocking, prefill, prefix hit 변화이다.

## 16. 결론

이번 결과는 8GiB 수준의 강제 KV capacity에서 tenant 수가 증가할수록 multi-turn KV history가 serving latency에 강한 영향을 준다는 점을 보여준다.

8 tenants에서는 blocking이 거의 없고, 16 tenants에서는 latency가 증가하지만 queue pressure는 제한적이다. 반면 32 tenants에서는 turn 후반부로 갈수록 blocking time이 급격히 증가하며, TTFT의 주요 component가 된다.

Group1은 2048 output tokens로 인해 turn 10에서 약 20K input/KV history에 도달했고, Group2는 512 output tokens로 약 6K 수준에 머물렀다. 따라서 현재 설정은 Group1/Group2의 KV history pressure를 분리하는 데 성공했다.

최종적으로, 이 실험은 multi-tenant multi-turn 환경에서 KV cache pressure가 단순 prefill latency 증가뿐 아니라 scheduler queue blocking과 prefix cache effectiveness 저하로 나타날 수 있음을 보여준다.
