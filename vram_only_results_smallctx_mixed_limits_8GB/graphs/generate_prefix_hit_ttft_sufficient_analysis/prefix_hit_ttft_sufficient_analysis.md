# Prefix Hit vs TTFT in Resource-Sufficient Slice

- Data: `exp1+exp2+exp3`
- Slice: `tenant_count in {8,16}`, `blocking_time_ms < 100`
- Goal: test whether prefix hit can be tied to TTFT gain after reducing queue-pressure confounding

## tenants=8
- Input bin 50-2548: 0-0.25=>p50 386.4, 0.5-0.75=>p50 4296.9
- Input bin 2548-4866: 0-0.25=>p50 3617.1, 0.25-0.5=>p50 3635.7, 0.5-0.75=>p50 3053.6, 0.75-1.0=>p50 17971.9
- Input bin 4866-9813: 0.5-0.75=>p50 7455.3, 0.75-1.0=>p50 19911.5
- Input bin 9813-20159: 0.75-1.0=>p50 26382.2

## tenants=16
- Input bin 47-2238: 0-0.25=>p50 354.5, 0.5-0.75=>p50 10749.1
- Input bin 2238-4866: 0-0.25=>p50 6818.7, 0.25-0.5=>p50 6830.7, 0.5-0.75=>p50 10705.6, 0.75-1.0=>p50 27920.8
- Input bin 4866-11309: 0.5-0.75=>p50 18650.1, 0.75-1.0=>p50 27956.6
- Input bin 11309-20159: 0.75-1.0=>p50 46257.5

## tenants=8+16
- Input bin 47-2316: 0-0.25=>p50 364.4, 0.5-0.75=>p50 7985.4
- Input bin 2316-4866: 0-0.25=>p50 6815.2, 0.25-0.5=>p50 6829.8, 0.5-0.75=>p50 10674.5, 0.75-1.0=>p50 21764.1
- Input bin 4866-11223: 0.5-0.75=>p50 12667.3, 0.75-1.0=>p50 26393.7
- Input bin 11223-20159: 0.75-1.0=>p50 36178.1

