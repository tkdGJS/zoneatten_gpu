# vLLM TTFT/TBT Isolation Prototype Documentation Pack

Generated: 2026-05-21

이 문서 묶음은 vLLM 소스코드를 수정하여 **multi-turn multi-tenant LLM serving에서 TTFT/TBT isolation을 실험하는 프로젝트**를 시작하기 위한 기본 문서 세트다.

핵심 방향은 다음과 같다.

> KV cache는 단순한 재사용 cache가 아니라, prefill recomputation을 줄이면서 동시에 serving concurrency와 decode cadence를 소모하는 shared scheduling resource다.  
> 따라서 목표는 KV를 많이 보존하는 것이 아니라, tenant별 token emission service를 보장하면서 KV retention, admission, externality를 제어하는 것이다.

## 사용 방법

1. 이 묶음을 vLLM fork의 root에 복사한다.
2. `AGENTS.md`는 Codex/agent가 항상 먼저 읽을 프로젝트 규칙이다.
3. `codex/START_HERE.md`를 Codex 첫 프롬프트의 컨텍스트로 사용한다.
4. 실제 구현 전에는 `docs/03_VLLM_CODE_MAP.md`의 파일 경로가 현재 checkout된 vLLM 버전과 맞는지 확인한다.
5. 코드 수정 후에는 반드시 `docs/06_PATCH_NOTES.md`에 변경 위치, 의도, 테스트 결과를 추가한다.

## 문서 구조

| 파일 | 목적 |
|---|---|
| `AGENTS.md` | Codex/agent용 최상위 개발 규칙 |
| `codex/START_HERE.md` | Codex가 프로젝트를 시작할 때 읽을 요약 |
| `codex/TASKS.md` | 초기 작업 목록 |
| `codex/PROMPTS.md` | Codex에게 줄 수 있는 프롬프트 템플릿 |
| `docs/00_PROJECT_OVERVIEW.md` | 프로젝트 목표와 framing |
| `docs/01_RESEARCH_MODEL.md` | TTFT/TBT/KV retention 연구 모델 |
| `docs/02_ARCHITECTURE.md` | 제안 시스템 아키텍처 |
| `docs/03_VLLM_CODE_MAP.md` | 수정 후보 파일과 역할 |
| `docs/04_API_FLOW.md` | 요청/API/scheduler 흐름 |
| `docs/05_IMPLEMENTATION_PLAN.md` | 단계별 구현 계획 |
| `docs/06_PATCH_NOTES.md` | 패치노트/버전 관리 로그 |
| `docs/07_TROUBLESHOOTING.md` | 트러블슈팅 위키 |
| `docs/08_RUNBOOK.md` | 실행/개발/테스트 방법 |
| `docs/09_EXPERIMENT_DESIGN.md` | 실험 설계 |
| `docs/10_METRICS_LOGGING.md` | 로깅/메트릭 설계 |
| `docs/11_DEV_RULES.md` | 개발 규칙과 코드 수정 원칙 |
| `docs/12_RISKS_AND_REVIEWER_DEFENSE.md` | 연구 리스크와 방어 논리 |
| `docs/13_CONFIG_FLAGS.md` | 추가할 설정/CLI flag 설계 |
| `docs/14_TEST_PLAN.md` | 테스트 계획 |
| `docs/15_DECISIONS.md` | Architecture Decision Record 목록 |
| `docs/16_GLOSSARY.md` | 용어집 |
| `docs/17_OPEN_QUESTIONS.md` | 미해결 질문 |
| `docs/99_SOURCE_REFERENCES.md` | 참고 자료와 원문 출처 |

## 첫 번째 구현 목표

처음부터 정책을 모두 구현하지 말고, 아래 순서로 진행한다.

1. **관찰 가능성 확보**: tenant id, request phase, token timestamp, batch step time, KV usage, preemption 등을 로깅한다.
2. **TBT 분해**: observed TBT를 decode kernel time, scheduler gap, prefill interference, KV allocation delay, preemption delay로 나누어 측정한다.
3. **baseline 재현**: vLLM default, length-aware grouping, static partition 등을 비교한다.
4. **policy 최소 구현**: TBT debt priority 또는 externality metering 중 하나만 먼저 적용한다.
5. **KV retention과 결합**: KV keep/reclaim 정책을 TBT guarantee 조건에 종속시킨다.
