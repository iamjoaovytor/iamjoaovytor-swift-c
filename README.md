# swift-c-fraud-detector

A high-performance fraud detection API built for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — a Brazilian backend engineering challenge focused on real-time vector similarity search under extreme resource constraints.

## The Challenge

Given a transaction payload, classify it as **fraud** or **legit** in under 1ms at 900 requests/second, using only:

- **2 API instances** — 0.35 CPU, 130 MB RAM each
- **1 load balancer** — 0.26 CPU, 78 MB RAM
- Total budget: **1 CPU core + 350 MB RAM**

The API receives JSON transaction data, converts it to a 14-dimensional feature vector, and runs a k-nearest-neighbor search against ~1 million labeled reference vectors to produce a fraud score.

## Approach

### Offline preprocessing

Before the container starts, a preprocessing step builds a compact binary index from the reference dataset:

- Quantizes float vectors to `Int16` (scale=10000) — 8× smaller, cache-friendly
- Trains an **IVF index** with K=2048 clusters via k-means++ — partitions the search space so only ~1% of vectors need to be scanned per query
- Builds an **IVF-PQ index** for approximate re-ranking

Everything is stored page-aligned for `mmap` — zero-copy loading at startup, no heap overhead.

### Runtime search stack

Each request goes through a tiered pipeline:

```
Request → Vectorize → IVF centroid scan (K=2048, once per query)
                    ↓
         Adaptive 3-tier nprobe: 2 → 8 → 16
              (expand only if result is ambiguous)
                    ↓
         Vote-based early cluster exit
         Distance-based early cluster exit
                    ↓
         Top-5 neighbors → majority vote → fraud score
```

**Key optimizations:**

- **AVX2 SIMD distance kernel** (C) — hand-vectorized L2² via `_mm256_madd_epi16`, processes 16 `Int16` lanes in a single instruction
- **Single centroid scan** — one O(K) scan for max nprobe, reused across adaptive tiers via prefix slicing
- **Vote-based early exit** — stops scanning clusters when top-k is already unanimous and close enough (dist² ≤ 2,500,000)
- **Pre-serialized HTTP responses** — all 6 possible outcomes (0/5 to 5/5 fraud votes) are pre-built as raw bytes at startup, bypassing the NIO HTTP encoder on the hot path
- **PGO** (Profile-Guided Optimization) — the Docker image is built with an instrumentation pass over real workloads, then recompiled with the profile data

### Infrastructure

- **[so-no-forevis](https://github.com/jrblatt/so-no-forevis)** — io_uring-based load balancer purpose-built for the competition, lower proxy overhead than HAProxy in this workload
- **Unix domain sockets** with socket handoff — the LB passes accepted connections directly to the API processes via `SCM_RIGHTS`, avoiding an extra copy through the kernel TCP stack
- **Single NIO event loop thread** — no context switching, no locks on the hot path

## Results

Local bench (synthetic dataset, Linux server):

| Bench | p99 | fp | fn | Score |
|-------|-----|----|----|-------|
| ×1    | 0.53ms | 0 | 0 | 6000 |
| ×2    | 0.51ms | 0 | 0 | 6000 |
| ×4    | 0.50ms | 0 | 0 | 6000 |

Official test (54,100 entries, 645 edge cases):

| Run | p99 | fp | fn | Score |
|-----|-----|----|----|-------|
| preview | 0.63ms | 0 | 0 | 6000 |

Perfect accuracy at 900–3600 req/s sustained load.

## Stack

- **Swift 6.1** — core API and search logic
- **C + AVX2** — distance computation kernel
- **Swift NIO** — async I/O (custom fork with socket handoff)
- **Docker** multi-stage build with PGO

## Structure

```
Sources/
  Domain/          — vectorizer, IVF search, scoring, fast JSON parser
    Parsing/       — hand-rolled byte-level JSON cursor
  api/             — NIO server, request handler, warmup
  preprocess/      — k-means++ IVF trainer, PQ trainer, binary writers
CSearch/           — AVX2 SIMD kernels (C)
resources/         — reference dataset + prebuilt indexes
```
