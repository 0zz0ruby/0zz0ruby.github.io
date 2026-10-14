---
layout: default
title: LOB Factor Calculation with Hadoop MapReduce
---

# LOB Factor Calculation with Hadoop MapReduce

**Java, Hadoop, HDFS, MapReduce**

[Back to portfolio](../) ·
[Source code and reproduction](https://github.com/0zz0ruby/lob-factor-mapreduce)

## From correctness to optimization

First implemented and validated a two-job calculation pipeline. After observing
its runtime cost, investigated the bottlenecks and redesigned the pipeline as
a single MapReduce job, reducing task-scheduling and intermediate-data overhead.

The workload covers approximately 1,500 stock snapshot files across five trading
days for the CSI300 universe. It calculates 20 limit order book factors and
their cross-sectional averages at each trading timestamp.

## Baseline

The first job groups and sorts snapshots by day and stock, computes
stock-level factors, and writes intermediate results to HDFS. The second job
calculates cross-sectional averages.

Hadoop logs and counters identified overhead from approximately 300 small Map
tasks and the intermediate results between the two jobs.

## Single-job design

1. Combine complete stock files into Map inputs while retaining ordered
   per-stock time series.
2. Calculate factors sequentially from the current and previous valid snapshots.
3. Accumulate per-factor `sum/count` by trading timestamp within each Mapper.
4. Merge additive statistics with a Combiner.
5. Produce sorted cross-sectional averages with one Reducer.

Selective field parsing, alternating reuse of two snapshot objects, and numeric
timestamp keys further reduce allocation and intermediate output.

## Validation

All 20 factors passed the **1% mean-relative-error threshold on each of five
days**: `0102`, `0103`, `0104`, `0105`, and `0108`, within 09:30:00–15:00:00.

The validator aligns rows by timestamp and checks headers, timestamp sets,
duplicates, and finite numeric values. A zero reference value is handled
conservatively: only an actual zero also receives zero error.

## Recorded runtime

| Implementation / day | Seconds |
|---|---:|
| Two-job baseline / 0102 | 195.036 |
| Memory-tuned two-job / 0102 | 151.078 |
| Single-job / 0102 | 16.553 |
| Single-job / 0103 | 16.714 |
| Single-job / 0104 | 16.764 |
| Single-job / 0105 | 17.222 |
| Single-job / 0108 | 16.759 |

The single-job pipeline averaged **16.802 seconds per day** across the five
days. Its timing includes previous-output cleanup, job submission and execution,
CSV export, and reference validation. The two-job timings cover submission and
execution of both jobs; the memory-tuning comparison was measured on one day.
The figures demonstrate the recorded runtime reduction but are not a strictly
like-for-like speedup measurement.

## Reproducibility

The recorded environment was Ubuntu 22.04, JDK 11, Hadoop 3.3.6, single-node
HDFS/YARN, 16 vCPUs, and 32 GiB memory. The repository retains both
implementations, build and timing scripts, validator regression tests,
and five-day validation records.

Raw snapshots and reference answers are not bundled. Each optimized run processes
one day, with one complete, time-ordered file per stock.

[Experiment notes](https://github.com/0zz0ruby/lob-factor-mapreduce/blob/main/docs/experiments.md)
