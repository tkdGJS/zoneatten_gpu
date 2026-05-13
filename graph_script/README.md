# graph_script

`exp1/exp2/exp3 same-timing` 분석에 사용한 그래프 생성 스크립트 모음이다.

## Script Summary

- `generate_32tenant_experiment_comparison.py`
  - 32-tenant 실험 비교 그래프와 요약 산출물 생성

- `generate_computation_residual_analysis_same_timing.py`
  - computation tokens 대비 residual/prefill 관련 분석 그래프 생성

- `generate_input_kv_usage_by_turn_same_timing.py`
  - turn별 input KV usage 그래프 생성

- `generate_kv_usage_by_turn_same_timing.py`
  - turn별 resident KV usage 그래프 생성

- `generate_prefill_axis_search_same_timing.py`
  - prefill을 설명하는 후보 축들에 대한 탐색 그래프 생성

- `generate_prefill_batch_rank_analysis_same_timing.py`
  - batch 내부 rank와 prefill 관계 분석 그래프 생성

- `generate_prefill_computation_analysis_same_timing.py`
  - prefill vs computation tokens 분석 그래프 생성

- `generate_prefill_compute_ratio_blocking_compare_same_timing.py`
  - blocking slice별 prefill vs total compute/8192 비교 그래프 생성

- `generate_prefill_cost_driver_analysis_same_timing.py`
  - prefill 비용 증가 요인(batch pressure, residual 등) 분석 그래프 생성

- `generate_prefill_delay_group_panels_same_timing.py`
  - earliest first-token delay 대비 prefill을 Group1/2 패널로 시각화

- `generate_prefill_delay_group_scatter_same_timing.py`
  - earliest first-token delay 대비 prefill scatter 그래프 생성

- `generate_prefill_direct_cause_graph_same_timing.py`
  - batch-level direct cause 관점의 prefill 그래프 생성

- `generate_prefill_internal_analysis_same_timing.py`
  - blocking `< 100 ms` 구간의 내부 prefill 동작 분석 그래프 생성

- `generate_prefill_wave_diagnostics_same_timing.py`
  - batch 내부 wave/cluster/delta-gap 진단 그래프 생성

- `generate_prefix_hit_prefill_gain_same_timing.py`
  - prefix hit가 prefill gain에 미치는 영향 분석 그래프 생성

- `generate_prefix_hit_ttft_sufficient_analysis.py`
  - resource-sufficient 구간에서 prefix hit와 TTFT 관계를 분석하는 그래프 생성

- `generate_prefix_ttft_relationship_same_timing.py`
  - TTFT vs prefix hit rate/tokens/blocking time 그래프 생성

- `generate_ttft_breakdown_timeline_same_timing.py`
  - turn별 TTFT breakdown(mean/p95/p99) 그래프 생성

- `generate_turn57_rootcause_analysis_same_timing.py`
  - turn 5/6/7 root cause 분석용 표/그래프 생성

- `generate_selected_paper_graphs.py`
  - 논문용으로 선택한 핵심 그래프만 별도로 생성

## Notes

- 대부분 스크립트는 `/home/yuhwa2323/zoneatten` 기준 데이터 경로를 가정한다.
- 일부 스크립트는 특정 분석 디렉토리를 출력 경로로 하드코딩하고 있으므로, 재실행 전 출력 경로를 확인하는 것이 안전하다.
