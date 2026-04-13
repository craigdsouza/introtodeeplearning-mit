# Debugging Neural Networks — A Diagnostic Framework

## Where to Focus Your Intuition

Understanding intermediate features matters more than evaluating results. A loss number tells you *that* something is wrong; intermediate features tell you *where* and *why*.

Think of it like debugging a geospatial pipeline: if your final map output is wrong, you don't just stare at the map — you check each stage: raw data ingestion, coordinate transforms, reprojection, rendering. Neural networks are the same: input → learned features → predictions. The features in the middle are where problems live.

---

## Diagnostic Framework: When Your Network Isn't Learning

Work through these in order. Each level rules out a class of problems before you go deeper.

### Level 1 — Data
- Are inputs normalized? (centered, similar scale across features)
- Are labels correct? (mislabeled data is more common than you'd expect)
- Is the dataset balanced? (90% one class = the network just predicts the majority class)
- Can you overfit a tiny batch? (train on 4-8 samples — if loss doesn't hit near-zero, the bug is in your code, not your data)

### Level 2 — Loss and Gradients
- Is the loss decreasing at all? If flat from the start: learning rate may be too low, or gradients are dead
- Is the loss oscillating wildly? Learning rate too high
- Is the loss decreasing then plateauing well above zero? Model may lack capacity (too few neurons/layers), or the problem isn't learnable by the architecture (like XOR with a single perceptron)
- Check gradient magnitudes: all zeros = dead neurons or vanishing gradients; exploding values = need gradient clipping or lower learning rate

### Level 3 — Architecture
- Is the model expressive enough? (a linear model can't learn non-linear boundaries — your XOR lesson)
- Is it too expressive? (overfitting: perfect training loss, terrible validation loss — the model memorized instead of learning)
- Are activations saturating? (sigmoid outputs all near 0 or 1 = gradients vanish; ReLU outputs all zero = dead neurons)

### Level 4 — Optimization
- Try a different learning rate (sweep 0.0001 to 10 on a log scale)
- Try a different optimizer (Adam is more forgiving than SGD for most problems)
- Train longer — some problems just need more epochs
- Try a different weight initialization — bad init can trap the network in a poor local minimum

### Level 5 — Intermediate Features
- Visualize hidden layer activations — are they diverse or all the same? (if all neurons output the same thing, the network has no representational power)
- Check what the first layer learned — in CV, first-layer weights often look like edge detectors. If they look like noise after training, something is wrong
- Compare activations across classes — the hidden features should look different for different classes

---

## Quick Checklist (copy this when debugging)

```
[ ] Can I overfit a tiny batch? (if no, bug in code)
[ ] Is the loss decreasing? (if no, check learning rate)
[ ] Is the loss plateauing too high? (if yes, model too small or wrong architecture)
[ ] Are gradients flowing? (not zero, not exploding)
[ ] Are activations diverse? (not all saturated or dead)
[ ] Is training data correctly labeled and normalized?
```

---

## The Core Intuition

A neural network is a chain of transformations. Each layer transforms its input into something more useful for the final task. When the network fails, one or more links in that chain are broken. Your job is to find which link — and intermediate features are how you look inside the chain.
