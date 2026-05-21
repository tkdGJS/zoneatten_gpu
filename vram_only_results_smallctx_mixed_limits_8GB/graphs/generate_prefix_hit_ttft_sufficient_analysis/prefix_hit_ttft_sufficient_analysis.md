# Prefix Hit vs TTFT in Resource-Sufficient Slice

- Data: `exp1+exp2+exp3`
- Slice: `tenant_count in {8,16}`, `blocking_time_ms < 100`
- Goal: test whether prefix hit can be tied to TTFT gain after reducing queue-pressure confounding

## tenants=8
- Input bin 50-2549: 0-0.25=>p50 390.1, 0.5-0.75=>p50 4294.6
- Input bin 2549-4866: 0-0.25=>p50 1929.9, 0.25-0.5=>p50 3612.3, 0.5-0.75=>p50 7391.0, 0.75-1.0=>p50 13466.7
- Input bin 4866-9813: 0.5-0.75=>p50 10600.6, 0.75-1.0=>p50 23567.8
- Input bin 9813-20159: 0.75-1.0=>p50 23672.5

## tenants=16
- Input bin 47-2233: 0-0.25=>p50 367.2, 0.5-0.75=>p50 14487.3
- Input bin 2233-4849: 0-0.25=>p50 6041.8, 0.25-0.5=>p50 6829.8, 0.5-0.75=>p50 10848.7, 0.75-1.0=>p50 25285.8
- Input bin 4849-11183: 0.5-0.75=>p50 18624.2, 0.75-1.0=>p50 36135.3
- Input bin 11183-20160: 0.75-1.0=>p50 42085.1

## tenants=8+16
- Input bin 47-2240: 0-0.25=>p50 385.1, 0.5-0.75=>p50 10698.1
- Input bin 2240-4850: 0-0.25=>p50 3657.8, 0.25-0.5=>p50 6292.3, 0.5-0.75=>p50 10733.9, 0.75-1.0=>p50 21754.8
- Input bin 4850-11182: 0.5-0.75=>p50 10662.8, 0.75-1.0=>p50 26378.2
- Input bin 11182-20160: 0.75-1.0=>p50 32043.4

