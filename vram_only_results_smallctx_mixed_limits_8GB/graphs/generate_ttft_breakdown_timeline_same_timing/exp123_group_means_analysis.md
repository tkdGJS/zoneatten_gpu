# TTFT Breakdown Analysis

exp1, exp2, exp3를 합친 뒤 tenant 8/16/32 조건에서 Group1과 Group2의 mean breakdown을 비교한 결과입니다.

## Group Definition

- Group1: `history_limit_tokens = 8192`
- Group2: `history_limit_tokens = 2048`
- 각 요청의 TTFT를 `blocking time + prefill time + remaining TTFT`로 분해해 비교합니다.

## Experiment Goal

- tenant 수가 `8`, `16`, `32`로 증가할 때 TTFT breakdown이 어떻게 변하는지 확인합니다.
- 같은 tenant 조건에서 Group1과 Group2의 차이가 주로 `blocking`, `prefill`, `remaining TTFT` 중 어디에서 발생하는지 봅니다.
- `exp1`, `exp2`, `exp3`를 합쳐 개별 실험 편차보다 전체 경향을 우선 확인합니다.

## Method

- 입력 데이터는 `exp1`, `exp2`, `exp3`의 성공한 요청만 사용했습니다.
- tenant_count를 `8`, `16`, `32`로 나누고, 각 조건에서 Group1과 Group2를 분리했습니다.
- Breakdown 항목은 다음처럼 계산했습니다.
  - Blocking Time: request metrics의 `queued_time_s * 1000`
  - Prefill Time: request metrics의 `prefill_time_s * 1000`
  - Remaining TTFT: `ttft_ms - blocking_ms - prefill_ms`
- `ttft_breakdown_group_means_by_turn` 결과는 같은 turn에 속한 요청들의 평균입니다.
- `mean_breakdown_bar` 결과는 turn 구분 없이 전체 요청 샘플 평균입니다.

## KV History Control

- 각 tenant는 고정된 ShareGPT 세션 하나를 배정받고, 같은 세션의 user turn을 순서대로 사용합니다.
- prompt는 이전에 완료된 `(user, assistant)` 쌍을 누적해 구성합니다.
- 누적 history가 `history_limit_tokens`를 초과하면, 토큰 기준으로 뒤쪽 최근 내용만 남기도록 잘라냅니다.
- 따라서 KV history는 `Group1=8192`, `Group2=2048` 한도 안에서 최근 대화 이력만 유지되도록 통제됩니다.
- 데이터셋 생성 단계에서도 각 세션이 해당 history limit 안에 들어오도록 사전 필터링합니다.

## Request Schedule

- 각 tenant는 `10 turns` 동안 요청을 보냅니다.
- 첫 동기화 배치 전에는 `30초` 대기합니다 (`pre_request_sleep_sec=30`).
- 이후 모든 tenant가 같은 turn에서 동시에 요청을 보내도록 barrier로 동기화합니다.
- turn 간 주기는 `30초`입니다 (`inter_turn_sleep_sec=30`).
- 즉 한 turn의 동시 요청을 보낸 뒤, 다음 동시 요청 배치는 `30초` 뒤에 시작되도록 스케줄됩니다.
- 각 tenant 수 실험은 `1 run`씩 수행했습니다.

## vLLM Configuration

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- dtype: `half`
- kv-cache-dtype: `auto`
- block-size: `16`
- max-model-len: `12288`
- max-num-seqs: `32`
- max-num-batched-tokens: `8192`
- num-gpu-blocks-override: `2048`
- cpu-offload-gb: `0`
- scheduling-policy: `fcfs`
- chunked prefill: enabled
- prefix caching: enabled
- max output tokens per request: `128`

## Environment

- Source dataset: `ShareGPT_V3_unfiltered_cleaned_split.json`
- tenant별 dataset은 실험 시작 전에 생성된 mixed-history-limit dataset을 사용했습니다.
- GPU 스펙은 현재 실행 환경에서 `nvidia-smi`가 NVIDIA driver와 통신하지 못해 자동 수집하지 못했습니다.

## Tenant 8

- Group1 mean total TTFT: `14014.58 ms`
- Group2 mean total TTFT: `11403.16 ms`
- Group1 breakdown: blocking `0.05 ms`, prefill `12725.20 ms`, remaining `1289.33 ms`
- Group2 breakdown: blocking `0.04 ms`, prefill `10192.97 ms`, remaining `1210.15 ms`
- Prefill 차이가 가장 큰 구간은 `Group1` 쪽이며, 두 그룹 간 prefill 평균 차이는 `2532.24 ms`입니다.
- 평균 total TTFT는 `Group1`가 더 크고, 차이는 `2611.42 ms`입니다.

## Tenant 16

- Group1 mean total TTFT: `25871.03 ms`
- Group2 mean total TTFT: `22649.71 ms`
- Group1 breakdown: blocking `23.23 ms`, prefill `24611.61 ms`, remaining `1236.18 ms`
- Group2 breakdown: blocking `180.90 ms`, prefill `21268.31 ms`, remaining `1200.50 ms`
- Prefill 차이가 가장 큰 구간은 `Group1` 쪽이며, 두 그룹 간 prefill 평균 차이는 `3343.31 ms`입니다.
- 평균 total TTFT는 `Group1`가 더 크고, 차이는 `3221.32 ms`입니다.

## Tenant 32

- Group1 mean total TTFT: `103908.20 ms`
- Group2 mean total TTFT: `83477.93 ms`
- Group1 breakdown: blocking `66560.61 ms`, prefill `34015.88 ms`, remaining `3331.71 ms`
- Group2 breakdown: blocking `56838.98 ms`, prefill `23940.27 ms`, remaining `2698.68 ms`
- Prefill 차이가 가장 큰 구간은 `Group1` 쪽이며, 두 그룹 간 prefill 평균 차이는 `10075.61 ms`입니다.
- 평균 total TTFT는 `Group1`가 더 크고, 차이는 `20430.27 ms`입니다.
