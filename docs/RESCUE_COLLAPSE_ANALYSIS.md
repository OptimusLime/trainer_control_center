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
