# Rescue Target Collapse Analysis

## Date: 2026-02-20

## Summary

The image neighborhood rescue mechanism (BCL step 9b) produces converging targets because the `affinity` matrix has an inherent column similarity of **0.30 at step 0 with random weights**, which amplifies to **0.95 by step 25** via a feedback loop. The root cause is **not** weight convergence (weights are only 0.06 similar at step 10 when affinity is already 0.81). The root cause is that `clamp(min=0)` on the affinity matrix destroys the negative-cosine signal that would differentiate features, combined with MNIST's natural image correlation (mean pairwise cosine = 0.40).

## The 5-Component Diagnostic

Each step of the rescue target computation was measured at every training step:

| Component | Shape | What it measures |
|-----------|-------|------------------|
| **affinity** | [B,D] | `cos(image, feature_weights)` clamped to [0,1] |
| **image_coverage** | [B] | How many features claim each image (sum of rank_score) |
| **image_need** | [B] | `1/(coverage+1)` — inverse underservedness |
| **weighted_affinity** | [B,D] | `affinity * image_need` — the combined signal |
| **rescue_pull** | [B,D] | `weighted_affinity` after top-k=8 sparsification + normalization |

Key metric: **column cosine similarity** — mean pairwise cosine between columns of each [B,D] matrix. When all columns look the same (sim → 1.0), all features get pulled toward the same target.

## Evidence: The Collapse Timeline (bcl-slow, som_lr=0.001)

### Step-by-step trace (first 30 steps)

```
step | aff_sim | wa_sim  | rp_sim  | wt_sim  | alive | delta_aff
-----|---------|---------|---------|---------|-------|-----------
   0 |  0.322  |  0.320  |  0.065  | -0.000  |   64  |     —
   1 |  0.381  |  0.379  |  0.067  |  0.001  |   64  |  +0.059
   2 |  0.448  |  0.448  |  0.078  |  0.002  |   64  |  +0.067
   3 |  0.518  |  0.515  |  0.103  |  0.005  |   64  |  +0.069
   5 |  0.636  |  0.638  |  0.120  |  0.014  |   64  |  +0.063
  10 |  0.813  |  0.815  |  0.167  |  0.060  |   48  |  +0.024/step
  15 |  0.901  |  0.903  |  0.285  |  0.128  |   40  |  +0.009/step
  25 |  0.954  |  0.955  |  0.329  |  0.258  |   32  |  +0.002/step
  29 |  0.964  |  0.965  |  0.317  |  0.299  |   31  |  +0.001/step
```

### Extended trajectory

```
step | aff_sim | wa_sim  | rp_sim  | wt_sim  | alive
-----|---------|---------|---------|---------|------
   0 |  0.322  |  0.320  |  0.065  | -0.000  |   64
   5 |  0.636  |  0.638  |  0.120  |  0.014  |   64
  10 |  0.813  |  0.815  |  0.167  |  0.060  |   48
  25 |  0.954  |  0.955  |  0.329  |  0.258  |   32
  50 |  0.979  |  0.981  |  0.426  |  0.470  |   18
 100 |  0.988  |  0.988  |  0.573  |  0.706  |    5
 200 |  0.990  |  0.990  |  0.484  |  0.846  |   25
 300 |  0.991  |  0.991  |  0.510  |  0.885  |   21
```

### Cross-condition comparison

```
           |  Step 0  |  Step 5  | Step 10  | Step 25  | Step 50  | Step 100 | Step 300
-----------|----------|----------|----------|----------|----------|----------|----------
bcl-micro  |    0.338 |    0.357 |    0.411 |    0.451 |    0.510 |    0.710 |    0.903
bcl-tiny   |    0.322 |    0.394 |    0.483 |    0.637 |    0.806 |    0.952 |    0.976
bcl-slow   |    0.307 |    0.578 |    0.763 |    0.945 |    0.980 |    0.987 |    0.991
```

Lower som_lr slows the collapse but does not prevent it. All conditions converge toward aff_sim > 0.97.

## Root Cause Analysis

### Finding 1: Affinity leads weight convergence, not the reverse

Affinity columns reach 0.81 similarity at step 10. Weights are only 0.06 similar at the same point. Affinity is the **cause**, not the effect.

```
Threshold | aff_sim crosses at | wt_sim crosses at
0.50      |       step 3       |      > step 29
0.70      |       step 7       |      > step 29
0.90      |       step 15      |      > step 29
```

### Finding 2: clamp(min=0) creates the initial 0.30 similarity

Without `clamp(min=0)`, affinity column similarity with random weights is **-0.003** (near zero, as expected for random vectors in 784D). With the clamp, it jumps to **0.298**.

```
Without clamp: affinity col cos_sim = -0.003
With clamp(min=0): affinity col cos_sim =  0.298
Fraction of entries clamped to 0: 50.8%
```

Why: With random weights in 784D, roughly half of `cos(image, weight)` values are negative. Clamping these to 0 creates a shared "zero floor" across all columns. Two columns that share the same zero-pattern are automatically correlated — they agree on "which images have zero affinity" even if they disagree on the magnitudes.

### Finding 3: MNIST image correlation amplifies the problem

MNIST images have mean pairwise cosine similarity of **0.40**. This means:
- If image_i and image_j are similar, then `affinity[i, f]` and `affinity[j, f]` are correlated for **every** feature f
- The shared structure among rows propagates to shared structure among columns
- With random uniform images, baseline affinity column similarity is 0.265 (still high due to clamp)

### Finding 4: image_need is flat and does not help

At step 0: `image_need` ranges from 0.042 to 0.066 — a max/min ratio of only **1.57**. Multiplying affinity by this flat vector does not change column structure:

```
wa_sim ≈ aff_sim at every step (within 0.002)
```

The `image_need` signal is too weak to create diversity because `image_coverage` (= sum of rank_scores) is nearly uniform — all images are "about equally claimed."

### Finding 5: Top-k sparsification helps but not enough

Rescue_pull (top-k=8 per column, normalized) has much lower column similarity than weighted_affinity:

```
Step  0: wa_sim = 0.320, rp_sim = 0.065   (sparsification helps 5x)
Step 25: wa_sim = 0.955, rp_sim = 0.329   (helps 3x, but still high)
Step 50: wa_sim = 0.981, rp_sim = 0.426   (collapsing too)
```

Top-k selects the 8 most-affine images per feature. When affinity columns are similar, the top-8 images overlap heavily across features. Sparsification only delays, doesn't prevent.

## The Feedback Loop (Causal Chain)

```
clamp(min=0) creates baseline 0.30 col_sim
          |
          v
SOM update: weight[f] += lr * (target[f] - weight[f])
  where target[f] = rescue_pull[:, f].T @ X   (weighted avg of batch images)
          |
          v
Similar rescue_pull columns → similar targets → weights move toward same point
          |
          v
More similar weights → more similar affinity columns
          |
          v
More similar rescue_pull → more similar targets → ...
          |
          v
Column similarity → 1.0 (convergence locked in by step 25)
```

The loop is self-reinforcing. Each SOM step makes the next step's targets more similar. The initial 0.30 from clamp(min=0) provides the seed.

## What image_need SHOULD do (but doesn't)

The idea: if image 72 has low coverage (nobody wins it), `image_need[72]` is high, so feature 26 gets pulled harder toward image 72. Different dead features should get pulled toward **different** underserved images because their weights point different directions (different affinity columns).

The reality: `image_need` varies by only 1.57x (max/min). All images are "about equally covered" because rank_score distributes fairly uniformly at step 0. By the time coverage becomes non-uniform (step 50: CV=0.41), affinity columns are already 0.98 similar — the diversity signal arrives too late.

## Implications for Algorithm Design

The current rescue mechanism (`affinity * image_need` → top-k → normalize → matmul with images) cannot produce diverse targets because:

1. **affinity is fundamentally a low-diversity signal** — `clamp(min=0)` on cosine similarity in 784D creates 50% shared zeros
2. **image_need is flat** — coverage is uniform at init, non-uniform only after features die (too late)
3. **SOM amplifies whatever similarity exists** — any initial correlation in targets becomes self-reinforcing

### Potential fixes (not yet tested)

1. **Remove the clamp**: Use raw cosine similarity [-1, 1]. This drops baseline col_sim from 0.30 to -0.003. But negative affinity means "this image is OPPOSITE to my weights" — pulling toward it would move weights in the wrong direction. Need a different interpretation.

2. **Rank-based affinity**: Instead of raw cosine, rank images per feature (which images am I MOST aligned with?). Ranking is inherently feature-specific — two features with different weights produce different rankings even if the raw cosines are correlated.

3. **Contrastive rescue**: Instead of `pull.T @ X` (weighted average), use the top-1 image per feature: `rescue_target[f] = X[argmax(affinity[:, f])]`. Hard assignment instead of soft average. Two features get different targets if their top-1 images differ — which they do at step 0 with random weights.

4. **Orthogonal SOM targets**: Project out the mean target from each feature's target. This forces diversity but may not have a meaningful geometric interpretation.

5. **Delayed rescue onset**: Don't apply rescue until features are actually dead (win_rate < threshold). At step 0 all features are alive — the rescue force is unnecessary and harmful.

---

## Fix Attempt 1: Hard Top-1 Rescue (Contrastive)

**Implemented**: Changed rescue from soft weighted average (top-k=8, normalized weights)
to hard top-k=1 (argmax — each feature's rescue target is the single best image).

### Results (bcl-slow, som_lr=0.001, with clamp(min=0) affinity)

```
step | rp_sim_old | rp_sim_new | tgt_sim_old | tgt_sim_new
-----|------------|------------|-------------|------------
   0 |     0.065  |     0.011  |    ~0.99    |     0.38
  50 |     0.426  |     0.435  |    ~0.99    |     0.78
 300 |     0.510  |     0.342  |    ~0.99    |     0.74
```

rescue_pull diversity improved 6x at init, target similarity improved from ~0.99 to
0.38-0.74. But affinity collapse unchanged (aff_sim still hits 0.98 by step 50).

---

## Fix Attempt 2: Rank-Based Affinity + Hard Top-1

**Implemented**: Replace `clamp(min=0)` cosine affinity with per-feature ranks:
```python
raw_affinity = X_norm @ W_norm.T                        # [B, D] cosine [-1, 1]
affinity = raw_affinity.argsort(dim=0).argsort(dim=0).float() / (B-1)  # [B, D] ranks [0, 1]
```

### Key insight: cosine similarity of rank columns is misleading

Random rank columns have cosine similarity 0.747 because all columns have identical
mean (0.5) and variance. The cosine metric is dominated by the shared mean, not by
actual correlation. **Pearson correlation** (= centered cosine) is the correct metric:

```
Random clamp(min=0) columns: cosine_sim = 0.298, pearson = 0.298
Random rank columns:         cosine_sim = 0.747, pearson = 0.002
```

### Rank affinity results (bcl-slow, Pearson correlation)

```
step | aff_corr | rp_corr | tgt_sim | wt_sim | unique_tgt | alive
-----|----------|---------|---------|--------|------------|------
   0 |  -0.001  |  0.026  |  0.470  | -0.001 |         24 |   64
   5 |   0.041  |  0.092  |  0.620  |  0.007 |         21 |   64
  10 |   0.161  |  0.213  |  0.699  |  0.034 |         20 |   64
  25 |   0.414  |  0.185  |  0.601  |  0.197 |         13 |   64
  50 |   0.569  |  0.216  |  0.678  |  0.425 |         12 |   64
 100 |   0.652  |  0.271  |  0.680  |  0.678 |          6 |   64
 200 |   0.688  |  0.458  |  0.808  |  0.827 |          9 |   64
 300 |   0.666  |  0.444  |  0.765  |  0.863 |          7 |   64
```

### Comparison: rank vs clamp (bcl-slow, step 300)

| Metric              | clamp(min=0) | rank-based |
|---------------------|-------------|-----------|
| aff column corr     |    ~0.99    |    0.67   |
| rescue_pull corr    |    ~0.51    |    0.44   |
| target similarity   |    ~0.99    |    0.77   |
| weight similarity   |     0.89    |    0.86   |
| features alive      |       21    |      64   |
| unique rescue tgts  |       ~1    |       7   |

### What improved
- **Affinity correlation near-zero at init** (vs 0.30 with clamp)
- **Slower collapse** — 0.67 at step 300 vs 0.99
- **All 64 features stay alive** (vs 18-31 dying with clamp)
- **Rescue targets more diverse** — 7-24 unique (vs ~1 with clamp)

### What did NOT improve
- **Weight similarity still reaches 0.86** at step 300
- **Unique rescue targets collapse** from 24 to 6-7 by step 100
- **Affinity correlation still grows** — feedback loop is slower but not broken

---

## Finding 6: Gradient Is Effectively Zero — SOM Dominates Everything

The most important finding from this session. The BCL grad_mask suppresses the
gradient to near-zero, making SOM the only force on weights.

```
                   |  grad_masked norm |  som_delta norm  | ratio (SOM/grad)
                   |  (per feature)    |  (per feature)   |
-------------------|-------------------|------------------|------------------
Step 0             |     0.000003      |     0.011        |   ~3,000x
Step 50            |     0.000000      |     0.010        |   ~10,000,000x
Step 100           |     0.000000      |     0.010        |   ~4,000,000x
```

Why gradient is zero: `grad_mask = rank_score * novelty * gradient_weight` where
`gradient_weight = effective_win = win_rate * feature_novelty`.
At step 0: `effective_win` mean = 0.016 (feature_novelty is ~0.05).
The triple-product mask (rank_score * novelty * 0.016) → ~0.0003.
Applied to already-small reconstruction gradients → effectively zero.

### The blending is also rescue-dominated

```
step | effective_win (mean) | rescue_weight (1-ew)
-----|---------------------|--------------------
   0 |        0.016        |       0.984
  10 |        0.016        |       0.984
  50 |        0.016        |       0.984
```

98.4% of SOM targets come from rescue. Winner pull barely contributes.

### Implication

**The algorithm is not "gradient with SOM regularization." It is "SOM with gradient
almost completely suppressed."** Weight convergence is caused by rescue-SOM pulling
all features toward the same few images (6-24 unique targets for 64 features).
Fixing the affinity matrix helps with initial diversity but cannot prevent convergence
because weights converge → rankings converge → argmax picks converge → targets converge.

The gradient suppression is a separate problem from rescue collapse. Even if rescue
targets were perfectly diverse, the algorithm wouldn't learn useful features because
the gradient signal (which carries task information) is being zeroed out.

## Raw Data

### Baseline affinity with random weights (5 seeds)

```
Seed  42: affinity col cos_sim = 0.298
Seed 123: affinity col cos_sim = 0.296
Seed 456: affinity col cos_sim = 0.308
Seed 789: affinity col cos_sim = 0.330
Seed 999: affinity col cos_sim = 0.320
```

### MNIST image statistics

```
Mean pairwise cosine similarity: 0.400
Min: 0.049, Max: 0.918
```

### clamp effect on column similarity

```
Without clamp(min=0): col_sim = -0.003 (random, as expected)
With clamp(min=0):    col_sim =  0.298 (50.8% of entries clamped to 0)
```

### Rank-based affinity column similarity

```
Random rank columns: cosine_sim = 0.747 (misleading — due to shared mean 0.5)
Random rank columns: pearson_corr = 0.002 (correct metric — near-zero as expected)
```
