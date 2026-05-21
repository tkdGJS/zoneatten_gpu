# Codex Task Board

## Milestone 0: Repo Mapping

- [ ] 현재 checkout의 vLLM 버전 확인
- [ ] Scheduler, KV cache, request, metrics 관련 실제 파일 경로 확인
- [ ] `docs/03_VLLM_CODE_MAP.md`를 현재 버전에 맞게 보정
- [ ] minimal smoke test 실행

## Milestone 1: Observability

- [ ] request metadata에서 tenant_id 전달 경로 확인
- [ ] tenant_id가 없다면 fallback tenant `"default"` 적용
- [ ] token timestamp 수집 위치 확인
- [ ] scheduler iteration id 추가 가능 여부 확인
- [ ] batch-level context/KV summary 로깅
- [ ] KV blocks used/free 로깅
- [ ] preemption event 로깅
- [ ] metrics를 default-disabled flag로 보호

## Milestone 2: TBT Decomposition

- [ ] observed TBT 계산
- [ ] scheduler gap 계산
- [ ] model execution step time 계산
- [ ] prefill chunk insertion 여부 로깅
- [ ] KV allocation delay 또는 allocation failure/stall 로깅
- [ ] preemption/recompute delay 로깅

## Milestone 3: Baselines

- [ ] vLLM default baseline
- [ ] length-aware grouping baseline
- [ ] static tenant partition baseline
- [ ] KV retention only baseline
- [ ] TBT debt only baseline

## Milestone 4: Policy Prototype

- [ ] TBT Debt Manager class 추가
- [ ] Decode Externality Meter class 추가
- [ ] TBT-aware admission hook 추가
- [ ] Elastic KV Reclaimer hook 추가
- [ ] 모든 policy는 config flag로 default off

## Milestone 5: Experiments

- [ ] Guarantee sweep
- [ ] Group1-only / Group2-only / Mixed
- [ ] TBT service guarantee evaluation
- [ ] ablation: retention only vs debt only vs externality only vs full policy

## Milestone 6: Documentation

- [ ] Patch notes 갱신
- [ ] Troubleshooting wiki 갱신
- [ ] Experiment report 작성
- [ ] Reviewer risk와 방어 논리 업데이트
