# PB0-I Non-Baseline Architecture Search

## Scope

PB0-I searches for genuinely non-baseline Modus_X architectures near the frozen canonical parameter budget.

The search uses the exact PB0-F calibrated parameter formula and verifies the highest-ranked candidates through the real Modus_X initializer.

No model source, checkpoint, dataset, or training run is modified.

## Frozen Budget

- Target parameters: **47,437,768**
- Search tolerance: **±50,000**

## Baseline Excluded

- r = 512
- n = 512
- h = 32

## Calibrated Formula

P = 12011080 + (36912 * r) + (30744 * n) + (12300 * h) + (12 * f * r) + (6144 * f)

where f = min(32, r, n)

## Search Result

- Non-baseline candidates within tolerance: **500**
- Exact non-baseline matches: **0**
- Real initializer verifications: **25**
- Exact formula/tree matches: **0**

## Highest-Ranked Non-Baseline Candidate

- Matrix rank r: **760**
- Vector dimension n: **176**
- Router hidden h: **120**
- Feedback rank f: **32**
- Predicted parameters: **47,439,592**
- Parameter error: **+1,824**
- Structural change score: **0.670312**
- Balance score: **0.406036**
- Overall ranking score: **0.649671**

## Interpretation

No genuinely non-baseline architecture exactly matches the frozen parameter budget inside the searched discrete space.

An architecture should not be declared better than the baseline from parameter arithmetic alone. Parameter-count equivalence only establishes a fair budget comparison. Performance requires subsequent controlled evaluation.

## Verification Results

| Rank | r | n | h | Predicted | Actual | Error | Exact | Status |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 760 | 176 | 120 | 47,439,592 | N/A | N/A | N/A | failed |
| 2 | 760 | 184 | 100 | 47,439,544 | N/A | N/A | N/A | failed |
| 3 | 760 | 192 | 80 | 47,439,496 | N/A | N/A | N/A | failed |
| 4 | 752 | 184 | 124 | 47,436,376 | N/A | N/A | N/A | failed |
| 5 | 752 | 192 | 104 | 47,436,328 | N/A | N/A | N/A | failed |
| 6 | 752 | 200 | 84 | 47,436,280 | N/A | N/A | N/A | failed |
| 7 | 768 | 168 | 116 | 47,442,808 | N/A | N/A | N/A | failed |
| 8 | 752 | 208 | 64 | 47,436,232 | N/A | N/A | N/A | failed |
| 9 | 768 | 176 | 96 | 47,442,760 | N/A | N/A | N/A | failed |
| 10 | 760 | 200 | 60 | 47,439,448 | N/A | N/A | N/A | failed |
| 11 | 768 | 184 | 76 | 47,442,712 | N/A | N/A | N/A | failed |
| 12 | 272 | 768 | 120 | 47,439,592 | N/A | N/A | N/A | failed |
| 13 | 744 | 192 | 128 | 47,433,160 | N/A | N/A | N/A | failed |
| 14 | 744 | 200 | 108 | 47,433,112 | N/A | N/A | N/A | failed |
| 15 | 768 | 192 | 56 | 47,442,664 | N/A | N/A | N/A | failed |
| 16 | 744 | 208 | 88 | 47,433,064 | N/A | N/A | N/A | failed |
| 17 | 744 | 216 | 68 | 47,433,016 | N/A | N/A | N/A | failed |
| 18 | 280 | 768 | 96 | 47,442,760 | N/A | N/A | N/A | failed |
| 19 | 280 | 760 | 116 | 47,442,808 | N/A | N/A | N/A | failed |
| 20 | 760 | 216 | 20 | 47,439,352 | N/A | N/A | N/A | failed |
| 21 | 768 | 208 | 16 | 47,442,568 | N/A | N/A | N/A | failed |
| 22 | 736 | 208 | 112 | 47,429,896 | N/A | N/A | N/A | failed |
| 23 | 752 | 216 | 44 | 47,436,184 | N/A | N/A | N/A | failed |
| 24 | 736 | 216 | 92 | 47,429,848 | N/A | N/A | N/A | failed |
| 25 | 760 | 208 | 40 | 47,439,400 | N/A | N/A | N/A | failed |

## Reproducibility

- Search fingerprint: `b6b18fbdcf98360a5b72cf8a99eda76dae075ba8fbf3b8a3ffc2625cefafcf2a`
- Generated UTC: `2026-09-09T15:54:45.122190+00:00`
