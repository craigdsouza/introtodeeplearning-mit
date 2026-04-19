# PyTorch Primer

> Read this before Exercise 3. Focus on understanding what PyTorch is doing under the hood — not just the API.

---

## Installation

PyTorch requires a specific install command for CUDA support — `pip install torch` alone gives you the CPU-only build.

**Step 1 — Check your CUDA version:**
```
nvidia-smi
```
Look for "CUDA Version" in the top-right. This is the maximum version your driver supports.

**Step 2 — Install with CUDA support:**

| Driver CUDA Version | Install command |
|---|---|
| 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| 13.x | Use cu124 — PyTorch is backward-compatible, no cu13x build exists yet |

For this repo (RTX 4050, driver CUDA 13.1):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Step 3 — Verify:**
```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4050 ...
```

**Activating the venv (PowerShell):**
```powershell
# From the repo root
.venv\Scripts\activate

# To deactivate
deactivate
```

**Notes:**
- Download is ~2GB, takes a few minutes
- The package name is `torch`, not `pytorch` — `pip install pytorch` will fail

---

## What PyTorch Is

PyTorch is a numerical computation library built around **tensors** (n-dimensional arrays, like NumPy arrays) that can run on GPU and — crucially — track gradients automatically.

The key insight: every operation you perform on a PyTorch tensor can be recorded in a **computation graph**. When you call `loss.backward()`, PyTorch walks that graph in reverse and computes `dL/dw` for every parameter. This is automatic differentiation — it replaces your manual gradient code from Exercises 1 and 2.

---

## Tensors — The NumPy Analogy

If you know NumPy, you already know tensors. The shapes, broadcasting rules, and indexing are almost identical.

```python
import torch
import numpy as np

# NumPy
a = np.array([[1.0, 2.0], [3.0, 4.0]])

# PyTorch equivalent
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Common creation patterns
torch.zeros(3, 4)          # zeros, shape (3,4)
torch.ones(3, 4)
torch.randn(3, 4)          # normal distribution
torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]

# Shape inspection (same concept as np.shape)
a.shape        # torch.Size([2, 2])
a.dtype        # torch.float32

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = a.to(device)
```

Key difference from NumPy: tensors can carry **gradient tracking**.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2          # y = x²
y.backward()        # compute dy/dx
x.grad              # tensor([4.]) — correct: dy/dx = 2x = 4
```

`requires_grad=True` turns on gradient tracking. Model parameters always have this set.

---

## The Training Loop — Three Steps

Every PyTorch training loop is the same three steps, repeated:

```python
# Step 1: Forward pass — compute prediction and loss
y_hat = model(X)
loss = loss_fn(y_hat, y)

# Step 2: Backward pass — compute gradients
optimizer.zero_grad()   # clear gradients from previous step (important!)
loss.backward()         # populate .grad on every parameter

# Step 3: Update — move parameters in the direction that reduces loss
optimizer.step()
```

This replaces the manual gradient math from your NumPy exercises:

| NumPy (manual) | PyTorch (automatic) |
|---|---|
| `dw = X.T @ (y_hat - y) / n` | `loss.backward()` |
| `w -= lr * dw` | `optimizer.step()` |

**Why `zero_grad()`?** PyTorch *accumulates* gradients by default — each `.backward()` call adds to `.grad` rather than replacing it. If you forget to zero, you're training on the sum of all past gradients.

---

## `nn.Module` — Defining a Model

All PyTorch models inherit from `nn.Module`. The two things you must implement:

```python
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs, 1 output; creates w and b automatically

    def forward(self, x):
        return self.linear(x)          # raw logit (no activation here)
```

`nn.Linear(in, out)` is exactly `z = X @ W.T + b` — the same dot product you wrote in NumPy, but with weights initialized and gradient-tracked automatically.

`forward()` defines the computation. You never call `forward()` directly — call the model as a function: `model(X)`, which PyTorch routes through `forward()` plus hooks.

---

## Loss Functions

```python
# Binary classification (single output)
loss_fn = nn.BCEWithLogitsLoss()
# Equivalent to sigmoid(y_hat) then BCE loss, but numerically stable.
# Input: raw logits (do NOT apply sigmoid before passing in)

# Multi-class classification
loss_fn = nn.CrossEntropyLoss()
# Input: raw logits of shape (batch, num_classes), targets are class indices

# Regression
loss_fn = nn.MSELoss()
```

**Why `BCEWithLogitsLoss` instead of `BCELoss`?**
`BCELoss` requires you to apply sigmoid first, which can produce numerical issues (log(0)). `BCEWithLogitsLoss` takes the raw logit and applies the log-sum-exp trick internally — more stable. This is why `TwoLayerNet.forward()` in Exercise 3 returns the raw output without sigmoid.

---

## Optimizers

```python
import torch.optim as optim

# Stochastic Gradient Descent — what you implemented manually
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Adam — adaptive learning rate, more forgiving than SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)

# `model.parameters()` returns an iterator over all w and b tensors in the model
```

Adam is almost always a better default than SGD for new experiments. SGD requires careful learning rate tuning; Adam adapts per parameter. Exercise 5 will make this concrete with loss curves.

---

## Activations

```python
import torch.nn.functional as F

# Applied in forward()
x = F.relu(x)       # max(0, x)
x = F.sigmoid(x)    # 1 / (1 + e^-x) — avoid in hidden layers
x = F.tanh(x)

# Or as nn.Module layers (same math, different style)
self.relu = nn.ReLU()
x = self.relu(x)
```

In practice: use ReLU (or variants) in hidden layers, nothing (raw logit) at the output for classification, and let the loss function handle the final activation.

---

## Data Types Gotcha

PyTorch is strict about dtypes. The most common error when moving from NumPy:

```python
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]])          # dtype: int64 — will break
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)  # correct

y = torch.tensor([0,0,0,1], dtype=torch.float32)     # must match model output dtype
```

---

## Minimal Working Example

The full pattern for Exercise 3:

```python
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32)  # AND gate

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = Perceptron()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Inference
with torch.no_grad():                      # disable gradient tracking for inference
    probs = torch.sigmoid(model(X))        # apply sigmoid to get probabilities
    print(probs)
```

---

## What PyTorch Is NOT Doing For You

- Choosing a good architecture
- Choosing a learning rate
- Detecting if your model is overfitting
- Verifying your data is correctly labeled

`loss.backward()` will faithfully compute gradients for a broken model on garbage data. The diagnostics in `debugging_nn_guide.md` are still entirely your job.
