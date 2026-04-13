# Week 1 — Foundations: The Perceptron to Neural Networks

**Goal:** Implement a perceptron from scratch in NumPy, then rebuild it in PyTorch and TensorFlow, so you understand what the frameworks are doing under the hood — a prerequisite for debugging training failures in real DRIVE perception pipelines.

> Pre-reading: [concepts/W01_perceptron.md](../../concepts/W01_perceptron.md) — read this before starting.

---

## Background

A neural network is just function approximation: given inputs, produce outputs, and iteratively adjust parameters to reduce error. The perceptron is the atomic unit. This week you build it three ways: by hand (NumPy), then in PyTorch, then in TensorFlow. Doing all three forces you to see what the frameworks abstract away — and what they don't.

The three logic gates you'll train on (AND, OR, XOR) are a classic benchmark because they expose a fundamental limitation of single-layer networks. XOR is not linearly separable: no straight line can divide the input space into the correct two classes. Successfully training AND and OR but failing on XOR with a single perceptron — and then succeeding on XOR with two layers — is the most visceral way to understand *why* depth matters.

---

## Exercise 1 — NumPy Perceptron from Scratch

**File:** Create `weeks/W01_perceptron/perceptron_numpy.py`

Implement a perceptron that trains on AND, OR, and XOR logic gates using only NumPy (no PyTorch, no TensorFlow).

Requirements:

- Inputs are all four 2-bit combinations: `[0,0], [0,1], [1,0], [1,1]`
- Use sigmoid activation and binary cross-entropy loss
- Train with gradient descent (manual update loop, not a library optimizer)
- Print the loss every 100 epochs
- After training, print predicted probabilities for all four inputs and whether the perceptron "passes" (predicted class matches label for all inputs)

Architecture:

```
  INPUTS          WEIGHTS         OUTPUT
  ──────          ───────         ──────

  x1 ─────w1─────┐
                  ├──→ Σ + b ──→ σ(z) ──→ y_hat
  x2 ─────w2─────┘

  (2,)    (2,)        scalar      scalar
```

2 inputs, 2 weights, 1 bias, 1 neuron. One straight decision boundary — enough for AND and OR, but not XOR.

Data flow with shapes:

```
X (4,2) → w (2,) → z (4,) → σ → y_hat (4,)
           layer 1
```

Scaffold:

```python
import numpy as np

def sigmoid(z): ...
def bce_loss(y, y_hat): ...
def forward(X, w, b): ...
def gradients(X, y, y_hat): ...
def train(X, y, epochs=5000, lr=0.5): ...

gates = {
    'AND': np.array([0, 0, 0, 1], dtype=float),
    'OR':  np.array([0, 1, 1, 1], dtype=float),
    'XOR': np.array([0, 1, 1, 0], dtype=float),
}
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)

for name, y in gates.items():
    w, b = train(X, y)
    # print results
```

**What to observe:** AND and OR should converge cleanly. XOR will not — the loss plateaus above chance. This is not a bug. It's the linear separability limit. Note the final loss values for each gate.

---

## Exercise 2 — NumPy Two-Layer Network (Solve XOR)

**File:** Create `weeks/W01_perceptron/twolayer_numpy.py`

You proved in Exercise 1 that a single perceptron cannot learn XOR. Now solve it by implementing a two-layer network from scratch in NumPy — no frameworks.

Architecture:

- Input layer: 2 neurons (X1, X2)
- Hidden layer: 4 neurons, sigmoid activation
- Output layer: 1 neuron, sigmoid activation

```
  INPUTS        HIDDEN LAYER (4 neurons)           OUTPUT LAYER
  ──────        ────────────────────────           ────────────

               ┌──→ Σ+b1 → σ → a1[0] ──┐
  x1 ──┬─W1───┼──→ Σ+b1 → σ → a1[1] ──┼─W2──→ Σ+b2 → σ → y_hat
       │       ├──→ Σ+b1 → σ → a1[2] ──┤
  x2 ──┴─W1───┼──→ Σ+b1 → σ → a1[3] ──┘
               │                        │
               │    4 neurons           │   1 neuron
               │                        │
  (4,2)    W1:(2,4)  b1:(1,4)      W2:(4,1)  b2:(1,1)
            8 weights  4 biases      4 weights  1 bias
```

Every input connects to every hidden neuron (2×4 = 8 connections, hence shape `(2,4)`). Then every hidden neuron connects to the single output (4×1 = 4 connections, hence `(4,1)`).

XOR input space — why one line isn't enough:

```
  x2
  1 │  1       0
    │
  0 │  0       1
    └──────────── x1
       0       1
```

Each hidden neuron learns a different linear boundary. The output neuron combines them to carve out non-linear regions — which is what XOR requires.

Data flow with shapes (compare to Exercise 1's single-layer pipeline):

```
X (4,2) → W1 (2,4) → z1 (4,4) → σ → a1 (4,4) → W2 (4,1) → z2 (4,1) → σ → a2 (4,1)
           layer 1                                  layer 2
```

Key differences from Exercise 1: two sets of weights (W1, W2), sigmoid applied twice, and the hidden layer widens the data from 2 to 4 features before narrowing back to 1 output.

Requirements:

- Implement forward pass through both layers
- Implement backpropagation through both layers manually (chain rule through the hidden layer)
- Use sigmoid activation and binary cross-entropy loss
- Train with gradient descent, learning rate 0.5, 10000 epochs
- Print the loss every 500 epochs
- After training, print predicted probabilities and whether XOR passes

Scaffold:

```python
import numpy as np

def sigmoid(z): ...
def sigmoid_deriv(z): ...   # σ(z) * (1 - σ(z))
def bce_loss(y, y_hat): ...

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)  # XOR — note shape (4,1)

# Initialize weights and biases
W1 = np.random.randn(2, 4) * 0.5   # input → hidden (2 inputs, 4 hidden neurons)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   # hidden → output (4 hidden, 1 output)
b2 = np.zeros((1, 1))

for epoch in range(10000):
    # Forward pass
    z1 = ...          # X @ W1 + b1
    a1 = ...          # sigmoid(z1) — hidden activations
    z2 = ...          # a1 @ W2 + b2
    a2 = ...          # sigmoid(z2) — output prediction

    # Backward pass (this is the hard part)
    # Start from the output and work backwards:
    # dL/da2 → dL/dz2 → dL/dW2, dL/db2
    # Then propagate error back to hidden layer:
    # dL/da1 → dL/dz1 → dL/dW1, dL/db1

    # Update all weights and biases
    ...
```

**The key challenge:** In Exercise 1, you only had one layer of gradients. Now you must propagate the error signal *back through* the hidden layer. The gradient for W1 depends on the gradient at the output, multiplied back through W2 and the hidden layer's sigmoid derivative. This is backpropagation — the algorithm that makes deep learning possible.

**Hint:** The error at the hidden layer is: `dz1 = (dz2 @ W2.T) * sigmoid_deriv(z1)`. Think about why W2 appears transposed here — it's the same "reversal" idea you worked out for X.T in Exercise 1, but now applied between layers.

**What to observe:** XOR should now converge — the loss drops to near zero and predictions match `[0, 1, 1, 0]`. Compare this to your Exercise 1 XOR result. The difference is depth: two layers can represent the non-linear boundary that one layer cannot.

---

## Exercise 3 — PyTorch Perceptron (was Exercise 2)

**File:** Create `weeks/W01_perceptron/perceptron_pytorch.py`

Rebuild the same experiment using `torch.nn.Linear` and `torch.optim.SGD`. Train on AND, OR, and XOR using the same hyperparameters as Exercise 1.

Requirements:

- Use `nn.BCEWithLogitsLoss` (sigmoid + BCE combined, numerically stable)
- Use `torch.optim.SGD` with the same learning rate
- Print loss every 100 epochs
- After training, print predicted probabilities and pass/fail for all four inputs

Then add a two-layer network for XOR:

```python
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)  # hidden layer, 4 neurons
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)  # raw logit, BCEWithLogitsLoss handles sigmoid
```

Train `TwoLayerNet` on XOR and verify it converges.

**What to observe:** Compare the PyTorch single-layer result on XOR to your NumPy result — they should fail in the same way. Then observe how `TwoLayerNet` succeeds. Note what `loss.backward()` and `optimizer.step()` are doing — they replace your manual gradient and update code from Exercise 1.

---

## Exercise 4 — TensorFlow / Keras Perceptron (was Exercise 3)

**File:** Create `weeks/W01_perceptron/perceptron_tf.py`

Implement the same single-layer and two-layer experiments in TensorFlow/Keras.

Requirements:

- Single-layer: `tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])`
- Compile with `optimizer='sgd'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
- Train on AND, OR, XOR; print final accuracy for each
- Two-layer for XOR: add a hidden `Dense(4, activation='relu')` before the output layer
- Print a comparison table: gate | NumPy result | PyTorch result | TF result

**What to observe:** All three frameworks should produce equivalent results. Notice how Keras hides the training loop entirely (`model.fit()`). This is convenient but means you must understand what's happening inside to debug it — which is why Exercise 1 (manual NumPy) comes first.

---

## Exercise 5 — Learning Rate Sensitivity (was Exercise 4)

**File:** Update `perceptron_pytorch.py` (add a section at the bottom)

Using your PyTorch single-layer model trained on AND, run three training runs with learning rates `[0.001, 0.1, 10.0]`. Plot the loss curve for each run using `matplotlib`.

```python
import matplotlib.pyplot as plt

lrs = [0.001, 0.1, 10.0]
for lr in lrs:
    losses = train_and_record_losses(X, y_and, lr=lr, epochs=500)
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Rate Sensitivity — AND gate')
plt.savefig('weeks/W01_perceptron/lr_sensitivity.png')
plt.show()
```

**What to observe:** `lr=0.001` converges slowly. `lr=0.1` converges cleanly. `lr=10.0` likely oscillates or diverges. Save the plot — you'll reference it in the quiz.

---

## Exercise 6 — Run the MIT Lab 1 Notebook (was Exercise 5)

**File:** Work in the course notebook (do not copy it into this folder — run it in its original location)

Clone the course repo if you haven't already:

```
git clone https://github.com/MITDeepLearning/introtodeeplearning
```

Open `lab1/Part1_TensorFlow.ipynb` and run all cells. Verify your GPU is detected.

Document what you observe in a file `weeks/W01_perceptron/lab1_notes.md`:

- Did `tf.config.list_physical_devices('GPU')` return your GPU?
- What does the course notebook do differently from your implementations?
- One thing you found surprising or that you didn't fully understand yet

**What to observe:** This verifies your environment is working end-to-end. The notebook uses TensorFlow — having already implemented the same ideas in NumPy and PyTorch means you'll read the course code with understanding rather than just running it.

---

## Checkpoint

You've passed Week 1 when you can:

- Implement forward pass, loss, gradients, and weight updates manually in NumPy without referencing notes
- Explain why XOR fails for a single perceptron and succeeds with two layers
- Describe what `loss.backward()` and `optimizer.step()` do in terms of the math
- Explain the effect of learning rate on training convergence, with reference to your plotted curves
- Run the MIT Lab 1 notebook end-to-end with GPU detected

