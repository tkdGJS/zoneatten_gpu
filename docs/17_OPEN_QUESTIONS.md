# 17. Open Questions

## Research Questions

1. TBT debt priority만으로 tenant-level TBT SLO를 얼마나 만족할 수 있는가?
2. Decode externality를 어떤 feature로 가장 잘 예측할 수 있는가?
3. Externality를 정확히 tenant별로 attribution해야 하는가, approximation으로 충분한가?
4. KV retention의 reuse benefit과 decode externality cost를 같은 scale로 어떻게 normalize할 것인가?
5. Static partition 대비 full policy의 TPS 이득은 충분히 큰가?
6. Long-context tenant를 제한할 때 fairness와 starvation을 어떻게 방지할 것인가?
7. Overload 상황에서 queueing, throttling, reject, scale-out 중 무엇을 기본 action으로 둘 것인가?
8. Client-visible TBT와 internal TBT 중 논문에서 무엇을 primary metric으로 둘 것인가?
9. Prefix caching, chunked prefill, speculative decoding이 TBT decomposition에 미치는 영향을 어떻게 분리할 것인가?
10. Multi-GPU/tensor parallel 환경에서 externality meter가 달라지는가?

## Implementation Questions

1. tenant_id는 OpenAI request metadata, header, extra field 중 어디로 받는 것이 가장 안전한가?
2. token timestamp를 어느 위치에서 기록해야 overhead와 정확성 균형이 좋은가?
3. SchedulerOutput에 metrics field를 추가해도 worker compatibility가 깨지지 않는가?
4. KV protected/elastic 구분을 block-level로 할 것인가 session/request-level로 할 것인가?
5. Existing vLLM metrics pipeline에 넣을 것인가 별도 JSONL trace로 둘 것인가?
6. async scheduling이 켜진 경우 scheduler gap 정의를 어떻게 할 것인가?
7. preemption이 발생한 request의 served token/debt를 어떻게 처리할 것인가?
8. CPU-only CI에서 GPU timing 관련 test를 어떻게 skip/mock할 것인가?

## Paper Framing Questions

1. Title에 isolation을 넣어도 충분히 strong한 guarantee를 보일 수 있는가?
2. Contribution을 KV retention policy로 둘 것인가, token service guarantee framework로 둘 것인가?
3. CascadeInfer류 length heterogeneity reduction과 차별화 실험을 어떻게 설계할 것인가?
4. Reviewer가 “engineering patch”라고 보지 않도록 abstraction을 어떻게 제시할 것인가?
