# Concepts: The Perceptron and Neural Network Foundations

> Pre-reading for Week 1 exercises. Read this before opening the exercises file.

---

## What This Is About

A neural network is a function approximator — given an input, it produces an output, and we can tune it to produce *better* outputs by adjusting its internal parameters. The perceptron is the smallest unit of that idea: one neuron, a handful of weights, and a rule for deciding when to "fire." Understanding it completely — the math, the geometry, the failure modes — is what lets you reason about why deep networks work at all, and what goes wrong when they don't.

---

## The Math (Plain English First)

### Forward Pass

A perceptron takes a vector of inputs `x = [x₁, x₂, ..., xₙ]`, multiplies each by a learned weight `w = [w₁, w₂, ..., wₙ]`, adds a bias `w₀`, and passes the result through an activation function `g`:

```
ŷ = g(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)
  = g(w₀ + wᵀx)
```

- `**wᵀx**` — the dot product. It's a weighted sum: how much does each input feature contribute?
- `**w₀` (bias)** — shifts the decision boundary. Without it, the boundary is always forced through the origin.
- `**g` (activation)** — squashes the raw sum into a useful range, and introduces non-linearity.

Think of it like this in NumPy (which you already know):

```python
import numpy as np

def perceptron(x, w, w0, activation):
    return activation(w0 + np.dot(w, x))
```

### Loss Function

To train, we need to measure how wrong our output is. For binary classification:

```
J(W) = (1/n) * Σ L(yᵢ, ŷᵢ)
```

- `L` is a per-sample loss (e.g. binary cross-entropy: `-y log(ŷ) - (1-y) log(1-ŷ)`)
  - only one term at a time is used , because of the terms y and (1-y), in a binary classification problem, both can't be non zero simultaneously.
  - thus the BCE is the "negative log-likelihood of a Bernoulli distribution"
  - the Binary Cross Entropy formula penalizes confidently wrong predictions exponentially more than somewhat uncertain predictions. If truth is 1 and prediction is 0.4, that's bad. However if prediction is 0.01 that's much worse.
- `J(W)` is the average loss over all training examples — the number we want to minimise

### Gradient Descent

We update weights in the direction that reduces `J`:

```
W ← W - η * ∇J(W)
```

- `**η` (learning rate)** — how big a step to take. Too large → overshoot and diverge. Too small → takes forever.
- `**∇J(W)` (gradient)** — the direction of steepest increase in loss. We go the *opposite* way.

## Problem - Computing the gradient was expensive

Before backprop (popularized by Rumelhart, Hinton & Williams in 1986), the main approach was:  
  **Finite differences (numerical gradients)** - Perturb each weight slightly and measure how the loss changes:

```
∂J/∂wᵢ ≈ [J(wᵢ + ε) - J(wᵢ)] / ε
```

This works, but it's catastrophically slow — you need one full forward pass per weight. A modern network with millions of weights would require millions of forward passes just to compute one gradient update.

### Backpropagation

Backprop is just the chain rule from calculus, applied systematically. The chain rule says: if you have a pipeline of functions, the derivative of the whole thing is the product of each step's derivative.

#### The sigmoid derivative — a clean result

The sigmoid `σ(z) = 1 / (1 + e⁻ᶻ)` has a derivative that can be expressed entirely in terms of its own output:

```
dσ/dz = σ(z) · (1 - σ(z))
```

**Derivation step by step:**

1. Rewrite as `(1 + e⁻ᶻ)⁻¹`
2. Power rule: `-1 · (1 + e⁻ᶻ)⁻²`
3. Chain rule — multiply by derivative of inner function `(1 + e⁻ᶻ)`:
  - derivative of `1` is `0`
  - derivative of `e⁻ᶻ` is `-e⁻ᶻ` (chain rule again: derivative of `e^u` is `e^u`, times derivative of `-z` which is `-1`)
4. Full result: `-1 · (1 + e⁻ᶻ)⁻² · (-e⁻ᶻ)` = `e⁻ᶻ / (1 + e⁻ᶻ)²`
5. Split the fraction: `1/(1 + e⁻ᶻ) · e⁻ᶻ/(1 + e⁻ᶻ)`
6. First part is `σ(z)`. For the second part, rewrite `e⁻ᶻ` as `(1 + e⁻ᶻ) - 1`:
  `(1 + e⁻ᶻ - 1) / (1 + e⁻ᶻ)` = `1 - 1/(1 + e⁻ᶻ)` = `1 - σ(z)`
7. Final: `**σ(z) · (1 - σ(z))`**

This is convenient because during backprop you already have `σ(z)` (the activation output) — no need to store `z` separately.

```python
def sigmoid_deriv(a):       # a = σ(z), already computed in forward pass
    return a * (1 - a)
```

#### Backprop for a single-layer network (0 hidden layers)

The forward pass pipeline and the chain rule applied to each step:

```
Forward:  X ──→ z ──→ ŷ ──→ L
          │     │     │
          z=Xw+b  ŷ=σ(z)  L=BCE(y,ŷ)

Chain:    ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w
```

Each partial derivative in the chain:


| Term            | What it means                                 | Formula              | With sigmoid + BCE |
| --------------- | --------------------------------------------- | -------------------- | ------------------ |
| `∂L/∂ŷ`         | How sensitive is loss to the prediction?      | `-y/ŷ + (1-y)/(1-ŷ)` |                    |
| `∂ŷ/∂z`         | How sensitive is sigmoid output to its input? | `ŷ(1-ŷ)`             |                    |
| `∂L/∂ŷ · ∂ŷ/∂z` | **Combined: these simplify**                  |                      | `**ŷ - y`**        |
| `∂z/∂w`         | How sensitive is linear output to weights?    | `X`                  |                    |
| `∂z/∂b`         | How sensitive is linear output to bias?       | `1`                  |                    |


The sigmoid + BCE combination simplifies beautifully: `∂L/∂z = ŷ - y`. This is why the gradient code is so clean:

```python
dw = X.T @ (y_hat - y) / n
db = mean(y_hat - y)
```

#### Backprop for a two-layer network (1 hidden layer)

The forward pipeline is longer — two layers, each with a linear step and an activation:

```
Forward:  X → z1 → a1 → z2 → a2 → L

Where:
  z1 = X  @ W1 + b1     a1 = σ(z1)      ← layer 1
  z2 = a1 @ W2 + b2     a2 = σ(z2)      ← layer 2
```

**Gradients for layer 2 (output layer)** — same pattern as the single-layer case, just with `a1` playing the role of `X`:

```
∂L/∂W2 = ∂L/∂a2 · ∂a2/∂z2 · ∂z2/∂W2
                └────┬────┘
              (a2 - y)         ·    a1

dW2 = a1.T @ (a2 - y) / n
db2 = mean(a2 - y)
```

**Gradients for layer 1 (hidden layer)** — the chain is longer. To reach W1, you pass through every step back from the loss:

```
∂L/∂W1 = ∂L/∂a2 · ∂a2/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂W1
          └────┬────┘         │          │          │
           (a2 - y)          W2     σ'(z1)         X
```

Why `W2` and not `dW2/da1`? Because `z2 = a1 @ W2 + b2` — here `a1` and `W2` are independent inputs to the same operation. The partial derivative of `z2` with respect to `a1` is simply `W2`. (Just like `∂z/∂x = w` when `z = x·w + b`.)

```python
dz1 = (a2 - y) @ W2.T * sigmoid_deriv(a1)    # error at hidden layer
dW1 = X.T @ dz1 / n
db1 = mean(dz1, axis=0)
```

#### How to determine shapes and transposes

**The rule:** A gradient must have the **same shape** as the parameter it updates. Use this to figure out every transpose.

Start with what you know:


| Variable | Shape                      | Why                         |
| -------- | -------------------------- | --------------------------- |
| `X`      | (samples, inputs) = (4, 2) | 4 samples, 2 features       |
| `W1`     | (inputs, hidden) = (2, 4)  | 2 inputs → 4 hidden neurons |
| `W2`     | (hidden, outputs) = (4, 1) | 4 hidden → 1 output         |
| `a2 - y` | (4, 1)                     | error per sample            |


**Step-by-step shape derivation for dW1:**

```
Goal: dW1 must be shape (2, 4) — same as W1

Start from the output error and work backwards:

1. (a2 - y)                         → (4, 1)   error at output
2. (a2 - y) @ W2.T                  → (4, 1) @ (1, 4) = (4, 4)   error spread to hidden layer
3. ... * sigmoid_deriv(a1)          → (4, 4) * (4, 4) = (4, 4)   element-wise, shapes match ✓
4. X.T @ ...                        → (2, 4) @ (4, 4) = (2, 4)   same shape as W1 ✓
```

**Step-by-step shape derivation for dW2:**

```
Goal: dW2 must be shape (4, 1) — same as W2

1. (a2 - y)                         → (4, 1)   error at output
2. a1.T @ (a2 - y)                  → (4, 4).T @ (4, 1) = (4, 4) @ (4, 1) = (4, 1) ✓
```

**The pattern:** To compute `dW` for any layer:

- Left side: transpose the **input** to that layer (puts samples dimension in the right place)
- Right side: the **error at that layer's output** (accumulated from all layers above)

**Why the transpose?** In the forward pass, `z = input @ W`, so data flows *through* W left-to-right. In the backward pass, the error needs to flow right-to-left, so we transpose. It's the same matrix, reversed direction.

#### Generalizing to N hidden layers

Every layer follows the exact same two-step pattern. Here is a network with layers numbered 1 through N:

```
Forward:  X → z1 → a1 → z2 → a2 → ... → zN → aN → L
```

**Step 1 — Compute the error at each layer, starting from the output:**

```
dzN     = (aN - y)                                   ← output layer (sigmoid + BCE simplification)
dz(N-1) = dzN @ WN.T * sigmoid_deriv(a(N-1))        ← one layer back
dz(N-2) = dz(N-1) @ W(N-1).T * sigmoid_deriv(a(N-2))
...
dz1     = dz2 @ W2.T * sigmoid_deriv(a1)            ← first hidden layer
```

The pattern: **error at layer i = (error at layer i+1) @ W(i+1).T * activation_derivative(a_i)**

Each layer takes the error from the layer ahead, sends it back through that layer's weights (transposed), and scales by how sensitive the local activation was.

**Step 2 — Compute weight gradients using the error:**

```
dWN = a(N-1).T @ dzN / n        ← input to layer N was a(N-1)
...
dW2 = a1.T @ dz2 / n           ← input to layer 2 was a1
dW1 = X.T @ dz1 / n            ← input to layer 1 was X
```

The pattern: **dW_i = (input to layer i).T @ (error at layer i) / n**

**As a loop** (pseudocode for a network with N layers):

```python
# Forward pass — store all activations
a[0] = X
for i in range(1, N+1):
    z[i] = a[i-1] @ W[i] + b[i]
    a[i] = sigmoid(z[i])

# Backward pass — compute errors, then gradients
dz[N] = a[N] - y                                  # output error
dW[N] = a[N-1].T @ dz[N] / n
db[N] = mean(dz[N], axis=0)

for i in range(N-1, 0, -1):                       # N-1, N-2, ..., 1
    dz[i] = dz[i+1] @ W[i+1].T * sigmoid_deriv(a[i])
    dW[i] = a[i-1].T @ dz[i] / n
    db[i] = mean(dz[i], axis=0)

# Update all weights
for i in range(1, N+1):
    W[i] -= lr * dW[i]
    b[i] -= lr * db[i]
```

This is essentially what `loss.backward()` does in PyTorch — it walks the computation graph in reverse, applying the chain rule at every node. The reason you'd never write a 100-layer network by hand is the bookkeeping, not the math.

---

## Intuition Builder

Think of a perceptron as drawing a straight line (in 2D) or a hyperplane (in higher dimensions) through your data. Points on one side are class 0, points on the other are class 1. Training moves that line around until it separates the classes as cleanly as possible.

### The XOR problem — why one line isn't enough

XOR is the logic gate that outputs 1 only when inputs differ:


| x₁  | x₂  | XOR output |
| --- | --- | ---------- |
| 0   | 0   | 0          |
| 0   | 1   | 1          |
| 1   | 0   | 1          |
| 1   | 1   | 0          |


With continuous inputs, the XOR pattern looks like this — O's cluster in the top-left and bottom-right, X's in the top-right and bottom-left:

```
x₂
1 |  o o o X X X X X
  |  o o o o X X X X
  |  o o o o o X X X
  |  X X o o o o o o
  |  X X X o o o o o
  |  X X X X o o o o
0 |  X X X X X o o o
  +-------------------
     0               1   x₁

O = output 1  (inputs differ)
X = output 0  (inputs similar)
```

Try to draw **one straight line** that puts all O's on one side and all X's on the other. You can't — the O's occupy two separate diagonal bands. Any line capturing the top-left O's cuts through the bottom-right O's. This is called **linearly non-separable**.

**A single perceptron can only draw one straight line. It will always fail on XOR.**

### How a hidden layer with 2 neurons solves it

Each hidden neuron draws its own line, creating a new coordinate for the output neuron to use.

- Hidden neuron 1 might learn: "are x₁ and x₂ both high?"
- Hidden neuron 2 might learn: "are x₁ and x₂ both low?"

Together these two signals let the output neuron say: "if neither is true (Bottom left quadrant), output 1."

After the hidden layer, the network has **transformed the input into new coordinates** (h₁, h₂) where the problem is now linearly separable. The exact positions depend on the learned weights, but the key outcome looks like this:

```
h₂ (hidden neuron 2 activation)
 1 |  O          O
   |        
   |       
 0 |  X          O
   +-------------------
      0           1    h₁ (hidden neuron 1 activation)

The two O's (originally diagonal) are now on the same side.
One straight line separates them cleanly.
```

Each layer re-represents the data until the final decision boundary is easy to draw. This is the entire motivation for deep networks.

**Geospatial analogy:** Classifying map tiles as "urban" vs "rural" by road density and building count. A perceptron draws one threshold line — works if the classes are cleanly separated. Fails for industrial zones (high roads, low buildings) that sit in the "wrong" region. A second layer can carve out that awkward corner with a second line.

---

## Activation Functions


| Function | Formula                   | Output range      | When to use                                                            |
| -------- | ------------------------- | ----------------- | ---------------------------------------------------------------------- |
| Sigmoid  | `1 / (1 + e⁻ᶻ)`           | (0, 1)            | Binary classification output layer                                     |
| Tanh     | `(eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)` | (-1, 1)           | Hidden layers (zero-centred, but still saturates)                      |
| ReLU     | `max(0, z)`               | [0, ∞)            | Default for hidden layers — fast, doesn't saturate for positive values |
| Softmax  | `eᶻᵢ / Σeᶻⱼ`              | (0, 1), sums to 1 | Multi-class classification output layer                                |


**Why non-linearity matters:** Without an activation function, stacking layers is mathematically equivalent to a single layer — the composition of linear transforms is still linear. Activations break this, allowing the network to approximate non-linear functions.

### Sigmoid

```
output
 1.0 |               . . . . . . .
     |             .
 0.5 |           *             <- output = 0.5 when z = 0
     |         .
 0.0 | . . . .
     +-------------------------
       -4   -2   0   2   4    z
```

**Shape:** S-curve that squashes any input into (0, 1).
**Intuition:** Naturally interpretable as a probability — "how confident am I this is class 1?" That's why it belongs on the **output layer** of a binary classifier.
**Problem in hidden layers:** The curve flattens out (saturates) at both ends. When `z` is very large or very small, the gradient is near zero — the weight gets almost no update signal. In a deep network this kills learning in early layers. This is the **vanishing gradient** problem.

### Tanh

```
output
 1.0 |               . . . . . . .
     |             .
 0.0 |           *   <- zero-centred when z=0
     |         .
-1.0 | . . . . 
     +-------------------------
       -4   -2   0   2   4    z
```

**Shape:** Same S-curve as sigmoid but centred at 0, output range (-1, 1).
**Intuition:** Zero-centred means the average activation across a batch tends to be near zero, which keeps gradients more balanced during training. Better than sigmoid in hidden layers for this reason.
**Same problem:** Still saturates at the extremes — vanishing gradients persist in very deep networks.

### ReLU

```
output
  4 |                   .
  3 |                 .
  2 |               .
  1 |             .
  0 |. . . . .  *    <- zero for all negative z
    +-------------------------
      -4   -2   0   2   4    z
```

**Shape:** Zero for negative inputs, linear for positive inputs.
**Intuition:** Dead simple to compute (`max(0, z)`), and crucially — for positive inputs the gradient is always 1, never shrinking. This is what allows deep networks (10, 50, 100+ layers) to train at all.
**Problem — dying ReLU:** If a neuron's input is always negative (e.g. due to a large negative bias), it outputs 0 forever and its gradient is always 0 — it stops learning entirely. This is the "dead neuron" problem.

**Fix — Leaky ReLU:** Allow a small slope for negative inputs:

### Softmax

Not shown as a simple curve because it operates on a **vector**, not a scalar. Given `n` raw scores (one per class), it converts them into probabilities that sum to 1:

```
Raw scores:  [2.0,  1.0,  0.5]
Softmax out: [0.63, 0.23, 0.14]  ← sum = 1.0
```

**Intuition:** "Of all the classes, how do I distribute my confidence?" The highest score gets amplified, the lowest gets suppressed. Used exclusively on the **output layer** for multi-class classification (e.g. "is this lane marking / vehicle / pedestrian?").

### Choosing which to use


| Where                  | Use     | Why                                                 |
| ---------------------- | ------- | --------------------------------------------------- |
| Hidden layers          | ReLU    | No vanishing gradient, fast, works in deep networks |
| Binary output          | Sigmoid | Output is a probability between 0 and 1             |
| Multi-class output     | Softmax | Outputs sum to 1 across all classes                 |
| Hidden layers (legacy) | Tanh    | Better than sigmoid but ReLU is usually preferred   |


---

## How It Works in Practice

### NumPy from scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, y_hat):
    eps = 1e-8  # avoid log(0)
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# Forward pass
def forward(X, w, b):
    return sigmoid(X @ w + b)  # X shape: (n_samples, n_features)

# Gradient of loss w.r.t. w and b
def gradients(X, y, y_hat):
    n = len(y)
    dw = (X.T @ (y_hat - y)) / n
    db = np.mean(y_hat - y)
    return dw, db

# Training loop
def train(X, y, epochs=1000, lr=0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        y_hat = forward(X, w, b)
        dw, db = gradients(X, y, y_hat)
        w -= lr * dw
        b -= lr * db
    return w, b
```

### PyTorch equivalent

```python
import torch
import torch.nn as nn

model = nn.Linear(in_features=2, out_features=1)
criterion = nn.BCEWithLogitsLoss()  # sigmoid + BCE, numerically stable
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    loss.backward()        # computes ∂loss/∂w via backprop
    optimizer.step()       # W ← W - η * ∇J(W)
```

Note: `nn.Linear` implements `z = xWᵀ + b` — the bias is included by default (`bias=True`).

---

## Common Pitfalls

- **Forgetting to zero gradients in PyTorch** (`optimizer.zero_grad()`). PyTorch accumulates gradients by default; skipping this causes gradients from previous batches to bleed into the current update.
- **Learning rate too high** — loss oscillates or diverges instead of decreasing smoothly. Start at `0.01` or `0.1` and plot the loss curve before tuning further.
- **Not normalising inputs** — if features have very different scales (e.g. pixel values 0–255 mixed with binary flags), gradients will be lopsided. Standardise inputs: `(x - mean) / std`.
- **Using sigmoid in hidden layers** — sigmoid saturates near 0 and 1, causing vanishing gradients in deep networks. Use ReLU in hidden layers; reserve sigmoid for binary output layers.

---

## Connection to NVIDIA DRIVE / Mapping

Every perception model on DRIVE — object detection, lane segmentation, depth estimation — is built from stacked, non-linear transformations that start with this exact abstraction. The CUDA kernels that run these models at 30+ FPS are executing matrix multiplies (`wᵀx`) and element-wise activations (ReLU) in parallel across thousands of GPU cores.

Understanding the forward pass and gradient update also matters for fine-tuning: when NVIDIA updates a perception model with new sensor data, they're running the same gradient descent loop, just at scale. Knowing *why* a learning rate schedule matters or *why* a loss saturates is the difference between a researcher who can debug training and one who can only run pre-written scripts.

---

## Quick-Check Questions

Before opening the exercises, answer these mentally:

1. A perceptron with no activation function (identity `g(z) = z`) is trained on a binary classification task. What is the fundamental limitation of this model, and why?
2. You increase the learning rate from 0.01 to 10. Describe what you expect to see in the training loss curve and explain why.
3. Why can't a single perceptron learn the XOR function, but a two-layer network can?
4. Why logarithm in Binary Cross Entropy (BCE)? Why not just use mean squared error?

```
 For classification with a sigmoid output, MSE creates a non-convex loss surface with many flat regions where gradients vanish. BCE combined with sigmoid produces a clean gradient: ŷ - y (as shown in your gradients() function at line 117). Simple, well-behaved.

The eps = 1e-8 in your code (line 107) prevents log(0) which would be -infinity — a defensive trick for numerical stability. PyTorch's BCEWithLogitsLoss handles this automatically by fusing the sigmoid and BCE into a single numerically stable operation.
```

1. What is the negative log-likelihood of a Bernoulli distribution?

