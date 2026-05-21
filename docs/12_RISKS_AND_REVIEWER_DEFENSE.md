# 12. Risks and Reviewer Defense

## Risk 1: “This is just length-aware batching.”

### Defense

This project is not merely grouping similar context lengths.

Difference:

| Length-aware batching | This project |
|---|---|
| Groups requests by sequence/context length | Enforces tenant-level token service guarantee |
| Optimizes batch efficiency | Controls TBT debt and externality |
| Heuristic | SLO/service-curve based |
| Focus on throughput/latency | Focus on isolation under shared decode externality |

Key statement:

> We do not merely group similar lengths. We meter and limit each tenant’s contribution to shared decode-step latency.

## Risk 2: “KV guarantee is throughput optimization, not isolation.”

### Defense

A pure KV retention policy may indeed be throughput/TTFT optimization.

Isolation framing requires explicit SLO constraints:

\[
P95(TBT_i) \le \tau_i
\]

and possibly:

\[
P99(TTFT_i) \le \theta_i
\]

Therefore the project must show:

- tenant-level SLO violation rate
- token service debt
- victim tenant TBT under mixed workload
- TPS cost to maintain isolation

## Risk 3: “Strong guarantee is not feasible under overload.”

### Defense

Correct. The system only guarantees within feasible capacity.

Overload handling is part of the design:

- queueing
- throttling
- admission rejection
- scale-out trigger
- best-effort degradation

Report feasible/infeasible regions explicitly.

## Risk 4: “TBT increase may be scheduler gap, not decode cost.”

### Defense

Observed TBT will be decomposed:

\[
ObservedTBT = DecodeKernelTime + SchedulerGap + PrefillInterference + KVAllocationDelay + PreemptionDelay
\]

Experiments must show which component dominates.

## Risk 5: “KV retention can improve one tenant by hurting another.”

### Defense

That is exactly the motivation. The policy keeps KV only if:

\[
ReuseBenefit > ConcurrencyCost + DecodeExternalityCost
\]

and victim tenant TBT debt remains below threshold.

## Risk 6: “Static partitioning can solve isolation.”

### Defense

Static partition is an important baseline.

Expected trade-off:

- strong isolation
- lower batching opportunity
- lower global TPS
- poor adaptation to changing tenant load

Our policy aims to preserve more batching opportunity while enforcing service guarantees.

## Risk 7: “Metrics overhead changes performance.”

### Defense

Metrics must be:

- default off
- measured for overhead
- separated into lightweight counters and heavy JSONL tracing
- used consistently across baselines

## Risk 8: “Policy is too complex.”

### Defense

Implementation is phased:

1. observability
2. decomposition
3. debt only
4. externality only
5. KV reclaim only
6. full policy

Ablation identifies which component is necessary.
