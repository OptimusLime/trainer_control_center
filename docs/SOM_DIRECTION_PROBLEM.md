# The SOM Direction Problem

## Date: 2026-02-23

## The Observation

The SOM delta is almost entirely positive (red in the diverging colormap) while the
gradient has both positive and negative values (red and blue). This means the SOM is
pushing all 64 features in the **same direction** — from near-zero initialization toward
bright pixel values.

```
weights:     mean = 0.004, std = 0.02   (near-zero init, range [-0.04, 0.05])
som_targets: mean = 0.170               (raw MNIST pixel averages, range [0, 1])
som_delta:   71.6% positive, 28.4% negative  (dominated by positive direction)
```

The SOM update is `weight += som_lr * som_weight_d * (target - weight)`. When:
- weights ≈ 0 (near-zero init)
- targets ≈ 0.17 (MNIST image averages, non-negative)
- then `(target - weight) ≈ target` → all positive → all features move the same way

Even after training, weights stay small (mean 0.004 at step 10) while targets are
0.17 on average. The SOM force is a **DC bias** pulling everything toward the mean
image, not a diversifying force.

## Why This Breaks the Algorithm

The SOM's job is to spread features apart across image space. To do that, different
features need to be pushed in **different directions**. But:

1. All rescue targets are non-negative (they're pixel averages)
2. All rescue targets have similar means (~0.17, because MNIST images have similar
   overall brightness)
3. Even the "diverse" part (which specific image is selected) is swamped by the
   shared positive bias

The SOM doesn't push Feature A left and Feature B right. It pushes both toward
the same bright blob. Increasing som_lr amplifies this shared push, accelerating
convergence instead of preventing it.

## The Mathematical Framing

What we want: a force on each feature's weight vector that moves it toward
**novel territory** — regions of image space not already covered by other features.

What we have: `delta_f = lr * (avg_image_near_f - weight_f)` — pull toward
the nearest images. When all images share a common component (the mean image),
all deltas share that component.

### The Core Issue: Absolute vs Relative Targets

The current SOM target is an **absolute position** in image space (a raw pixel
average). The delta is the vector from the current weight to that position.

What we need is a **relative direction** — "move away from your neighbors" or
"move toward the gap between existing features." The target shouldn't be "go to
image X" but "go in the direction that increases your novelty."

## Potential Approaches

### Approach 1: Mean-Subtracted Targets

The simplest fix. Subtract the mean target across all features before computing
the delta:

```python
som_targets_centered = som_targets - som_targets.mean(dim=0, keepdim=True)  # [D, 784]
som_delta = som_lr * som_weight_d * som_targets_centered
# No longer (target - weight) — just the centered target direction
```

This removes the shared DC bias. Each feature's delta becomes "how your target
differs from the average target." If all targets are the same, the centered
delta is zero (no force). If a feature's target is uniquely bright in some region,
its delta points toward that region.

**Pros**: Trivial to implement. Removes the convergent bias.
**Cons**: Still operates on the target, not on the weight. The "move toward your
unique target component" might not correspond to "move toward novel territory."
The centered target could still point all features in similar directions if the
target variation is low-rank.

### Approach 2: Repulsive Forces Between Features

Instead of pulling toward a target, push features **away from each other**. Each
feature's SOM delta is the negative sum of attraction to its neighbors:

```python
W = encoder.weight  # [D, 784]
W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

# Pairwise cosine similarity
sim = W_norm @ W_norm.T  # [D, D]

# Each feature is repelled by similar features
# Repulsion vector for feature f = -sum_g(sim[f,g] * W_norm[g])
# (weighted sum of neighbor directions, negated = push away)
repulsion = -sim @ W_norm  # [D, 784]

# Only repel from nearby features (threshold or top-k neighbors)
# Scale by som_weight_d (redundant features get more repulsion)
som_delta = som_lr * som_weight_d.unsqueeze(1) * repulsion
```

**Pros**: Directly addresses the goal — push features apart. No dependence on
image targets at all. Mathematically clean: minimizes pairwise similarity.
**Cons**: No connection to the data. Features spread out, but toward what? They
might spread into useless regions of weight space that don't correspond to any
images. The SOM should spread features across the **data manifold**, not across
arbitrary directions.

### Approach 3: Contrastive Novelty Gradient

Treat novelty as a differentiable objective and compute its gradient with respect
to the weights. Feature novelty = "of the images I win, how exclusively do I win
them?" We want to maximize this.

Define a soft novelty score:

```
novelty_f = sum_b[ rank_score[b,f] / (sum_g rank_score[b,g]) ] / sum_b[rank_score[b,f]]
```

This is already `feature_novelty` in the algorithm. The question is: what direction
should weight_f move to **increase** this score?

```python
# Novelty increases when:
# 1. You win images that nobody else wins (increase rank_score for exclusive images)
# 2. You stop winning images that everyone else wins (decrease rank_score for crowded images)
#
# rank_score = sigmoid(temperature * (strength_f - max_neighbor_strength))
# strength = |act| = |X @ W.T|
#
# d(novelty)/d(W_f) is complex but the key insight is:
# To increase novelty, move toward images where you're the ONLY winner
# and away from images where many features win.

# Approximate: weight the batch by "exclusivity"
exclusivity = rank_score / (image_coverage.unsqueeze(1) + 1e-8)  # [B, D]
# Normalize per feature
excl_weights = exclusivity / (exclusivity.sum(dim=0, keepdim=True) + 1e-8)  # [B, D]

# Novelty target: weighted average of images by exclusivity
novelty_target = excl_weights.T @ X  # [D, 784]

# Delta: move toward your exclusive images, away from current position
novelty_delta = som_lr * som_weight_d.unsqueeze(1) * (novelty_target - weight)
```

**Pros**: Data-aware — targets are images the feature exclusively wins. Directly
optimizes the novelty metric. Different features get different targets because
they have different exclusivity profiles.
**Cons**: At init, all features have similar exclusivity (all win similarly) →
targets may still converge. The exclusivity signal might be too weak early
(same chicken-and-egg problem as current feature_novelty).

### Approach 4: Novelty Gradient Descent (Analytic)

Compute the actual gradient of a diversity loss with respect to encoder weights.
Define:

```
L_diversity = mean pairwise cosine similarity of encoder weights
            = (1/D^2) * sum_{f,g} (W_f · W_g) / (||W_f|| * ||W_g||)
```

This is differentiable. The gradient `dL_diversity/dW_f` tells each feature exactly
which direction to move to reduce its similarity with other features:

```python
W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # [D, 784]
sim = W_norm @ W_norm.T  # [D, D]

# Gradient of cosine similarity w.r.t. W_f:
# d/dW_f [cos(W_f, W_g)] = (W_g_norm - cos(f,g) * W_f_norm) / ||W_f||
# Summed over all g ≠ f:
for f in range(D):
    grad_f = sum over g≠f: (W_norm[g] - sim[f,g] * W_norm[f]) / W_norms[f]

# Vectorized:
# diversity_grad[f] = (sum_g W_norm[g] - sim[f,:].sum() * W_norm[f]) / ||W_f||
# = (mean_W_norm * D - sim_row_sum * W_norm[f]) / ||W_f||

# SOM delta = -som_lr * diversity_grad  (negative because we want to DECREASE similarity)
```

**Pros**: Mathematically exact. Provably moves features apart. Each feature gets
a unique gradient because its similarity profile is unique. No dependence on image
targets — purely geometric.
**Cons**: Same as Approach 2 — no connection to the data manifold. Features
might become orthogonal but useless. Also, the gradient of cosine similarity
can be unstable when features are very similar (the "which direction to push"
becomes ambiguous when vectors are nearly identical).

### Approach 5: Data-Constrained Repulsion (Hybrid)

Combine Approach 2 (repulsion) with Approach 3 (data-awareness). The SOM delta
has two components:

1. **Attraction**: pull toward your exclusive images (data-aware target)
2. **Repulsion**: push away from neighboring features (geometric diversity)

```python
# Attraction: move toward images you exclusively win
exclusivity = rank_score / (image_coverage.unsqueeze(1) + 1e-8)
excl_weights = exclusivity / (exclusivity.sum(dim=0, keepdim=True) + 1e-8)
attraction_target = excl_weights.T @ X  # [D, 784]

# Repulsion: push away from similar features
W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
sim = W_norm @ W_norm.T
sim.fill_diagonal_(0)  # don't repel from yourself
repulsion = -(sim @ W_norm)  # [D, 784]

# Combine: attraction pulls toward data, repulsion spreads apart
som_delta = som_lr * som_weight_d.unsqueeze(1) * (
    alpha * (attraction_target - weight) + (1 - alpha) * repulsion
)
```

**Pros**: Best of both worlds — data-aware AND geometrically diversifying.
Features spread out on the data manifold rather than into empty space.
**Cons**: Extra hyperparameter (alpha). More complex. May need tuning.

## Evaluation Criteria

Any fix should satisfy:
1. **At step 0**: SOM deltas should point in diverse directions (not all positive)
2. **At step 100**: weight similarity should be LOWER than without SOM
3. **At step 3000**: weight similarity should plateau well below 0.95
4. **Loss should not degrade**: SOM diversity force should not fight gradient so
   hard that reconstruction suffers (loss should still reach ~0.09)

## Recommendation

Start with **Approach 1 (mean-subtracted targets)** as a sanity check — does removing
the DC bias help at all? It's a one-line change.

Then try **Approach 4 (analytic diversity gradient)** — it's the most principled and
has no hyperparameters beyond som_lr. If features spread into useless space, combine
with Approach 3 to constrain them to the data manifold (→ Approach 5).

Approach 2 (pure repulsion) is worth trying because it's simple, but likely has the
"useless orthogonal features" problem without data-awareness.
