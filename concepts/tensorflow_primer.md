# TensorFlow / Keras Primer

> Read this before Exercise 4. If you've read the PyTorch primer, you'll notice the concepts are identical — the APIs just look different.

---

## Installation

TensorFlow's GPU support on native Windows was dropped after TF 2.10. Since you have WSL2, use that — it's the recommended path for GPU-accelerated TF on Windows.

### Option A — WSL2 (recommended, GPU support)

Open your WSL2 terminal. The venv lives inside the project folder but is created from Linux, so it has a `bin/` layout (not `Scripts/`) — it cannot be shared with the Windows venv.

```bash
# Step 1 — install venv support (Ubuntu ships Python without it)
sudo apt install python3.12-venv -y

# Step 2 — create a Linux venv alongside the Windows one
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

# Step 3 — install TF (large download ~2GB, use higher timeout to avoid mid-download failures)
pip install tensorflow[and-cuda] --timeout 300
```

`tensorflow[and-cuda]` installs TF plus CUDA/cuDNN as pip packages under `.venv-wsl/lib/python3.12/site-packages/nvidia/` — no separate CUDA install needed inside WSL.

**Step 4 — fix LD_LIBRARY_PATH (required — TF can't find its own CUDA libs without this)**

The pip-installed CUDA libraries aren't on the system path by default. Add them:

```bash
export LD_LIBRARY_PATH=$(find .venv-wsl/lib/python3.12/site-packages/nvidia -name "lib" -type d | tr '\n' ':'):/usr/lib/wsl/lib/:$LD_LIBRARY_PATH
```

Make it permanent. First load your repo path from `.env`, then append to `.bashrc`:

```bash
# Load REPO_ROOT from .env
export $(grep -v '^#' .env | xargs)

echo "export LD_LIBRARY_PATH=\$(find $REPO_ROOT/.venv-wsl/lib/python3.12/site-packages/nvidia -name \"lib\" -type d | tr '\n' ':'):/usr/lib/wsl/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

`REPO_ROOT` is defined in `.env` (gitignored). Copy `.env.example` to `.env` and set it to your local repo path if setting up on a new machine.

Verify the line was written correctly (`-type d` must be present):
```bash
tail -3 ~/.bashrc
```

If `-type d` is missing, fix with: `sed -i '$ s/-type /-type d /' ~/.bashrc`

**Step 5 — verify GPU is detected:**
```bash
source ~/.bashrc
source .venv-wsl/bin/activate   # re-activate — sourcing .bashrc deactivates the venv
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should return: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Notes for WSL2 GPU passthrough:**
- Requires Windows 11 or Windows 10 21H2+ (you're on Windows 11 — you're fine)
- NVIDIA driver must be installed on the **Windows** side only — do not install a Linux driver inside WSL
- WSL2 exposes the GPU to Linux automatically via the Windows driver
- `libcuda.so.1` comes from `/usr/lib/wsl/lib/` (Windows driver); the rest (cudart, cublas, cudnn, cufft) come from the pip-installed nvidia-* packages
- `source ~/.bashrc` will deactivate your venv — always re-run `source .venv-wsl/bin/activate` after it

**Diagnostic — if GPU still not detected:**
```python
import ctypes
libs = ['libcuda.so.1', 'libcudart.so.12', 'libcublas.so.12', 'libcudnn.so.9', 'libcufft.so.11']
for lib in libs:
    try:
        ctypes.CDLL(lib)
        print(f'OK: {lib}')
    except OSError as e:
        print(f'MISSING: {lib} — {e}')
```
Any `MISSING` line means that library's directory isn't on `LD_LIBRARY_PATH`.

### Option B — Native Windows (CPU only, simpler)

If you don't want to use WSL for Week 1 exercises (AND/OR/XOR datasets are tiny — CPU is fast enough):

```
pip install tensorflow
```

This gives you the CPU-only build. Good enough for Weeks 1–2. Switch to WSL before Week 3 (CNNs) where GPU speed starts to matter.

**Verify (CPU build):**
```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))  # returns [] on CPU build — expected
```

### Which to use now?

| | WSL2 (Option A) | Native Windows (Option B) |
|---|---|---|
| GPU support | Yes | No |
| Setup effort | Moderate | Minimal |
| Week 1–2 exercises | Fine | Fine |
| Week 3+ (CNNs) | Required | Too slow |

For Week 1, either works. Set up WSL properly before Week 3.

---

## What TensorFlow Is

TensorFlow is Google's numerical computation library. For most practical work you interact with it through **Keras**, the high-level API bundled with TensorFlow since TF 2.0. Keras hides the training loop entirely — you define the model, compile it with a loss and optimizer, and call `model.fit()`.

The tradeoff vs. PyTorch: Keras is faster to write but harder to debug, because the training loop is inside the framework. Understanding your NumPy and PyTorch implementations is what lets you reason about what `model.fit()` is doing internally.

---

## The Core Difference From PyTorch

| | PyTorch | TensorFlow/Keras |
|---|---|---|
| Training loop | You write it | `model.fit()` handles it |
| Gradient computation | `loss.backward()` | Automatic inside `fit()` |
| Model definition | `nn.Module` subclass | `Sequential` or `Model` subclass |
| Style | Imperative (Pythonic) | Declarative (describe, then run) |

Both frameworks use automatic differentiation under the hood. The math is identical — only the interface differs.

---

## Tensors

TensorFlow tensors work like PyTorch tensors and NumPy arrays. For most operations you can use NumPy arrays directly and Keras will convert them.

```python
import tensorflow as tf
import numpy as np

# Create tensors
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
a = tf.zeros((3, 4))
a = tf.ones((3, 4))
a = tf.random.normal((3, 4))

# Shape and dtype
a.shape      # TensorShape([3, 4])
a.dtype      # tf.float32

# Convert to NumPy
a.numpy()    # returns np.ndarray

# TF is happy to accept NumPy arrays directly in most places
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
model.predict(X)   # works fine
```

---

## Defining a Model — Sequential API

The simplest way: stack layers in order.

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # hidden layer
    keras.layers.Dense(1, activation='sigmoid')                  # output layer
])
```

`Dense(units, activation)` is the Keras equivalent of `nn.Linear` + activation. It creates `W` and `b` automatically.

`input_shape` tells Keras the shape of one sample (not the batch). You only need it on the first layer.

For a single-layer perceptron (no hidden layer):

```python
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])
```

---

## Compiling — Loss, Optimizer, Metrics

Before training you must `compile()` the model:

```python
model.compile(
    optimizer='sgd',                    # or optimizer=keras.optimizers.SGD(lr=0.5)
    loss='binary_crossentropy',         # string shorthand, or keras.losses.BinaryCrossentropy()
    metrics=['accuracy']                # optional: what to track during training
)
```

Common loss strings:

| String | Use case |
|---|---|
| `'binary_crossentropy'` | Binary classification (sigmoid output) |
| `'categorical_crossentropy'` | Multi-class (softmax output, one-hot labels) |
| `'sparse_categorical_crossentropy'` | Multi-class (integer labels) |
| `'mse'` | Regression |

Common optimizer strings: `'sgd'`, `'adam'`, `'rmsprop'`.

To set learning rate explicitly:

```python
optimizer=keras.optimizers.SGD(learning_rate=0.5)
```

---

## Training — `model.fit()`

```python
history = model.fit(
    X, y,
    epochs=5000,
    batch_size=4,      # for small datasets, set to full dataset size
    verbose=0          # 0=silent, 1=progress bar, 2=one line per epoch
)
```

`history.history` is a dict of metrics per epoch:

```python
history.history['loss']      # list of loss values, one per epoch
history.history['accuracy']  # if you added accuracy to metrics
```

This is what you'd plot for a loss curve — equivalent to the losses list you'd collect manually in PyTorch.

---

## Evaluation and Prediction

```python
# Evaluate on data — returns [loss, ...metrics]
loss, accuracy = model.evaluate(X, y, verbose=0)

# Predict — returns probabilities (already has sigmoid applied if activation='sigmoid')
probs = model.predict(X)    # shape: (4, 1)
classes = (probs > 0.5).astype(int)
```

Note: unlike PyTorch where you manually apply `torch.sigmoid()` at inference, Keras `predict()` runs the full forward pass including the output activation you specified at model definition.

---

## Inspecting the Model

```python
model.summary()
# Prints layer names, output shapes, parameter counts

model.get_weights()
# Returns list of NumPy arrays: [W1, b1, W2, b2, ...]

# After training, inspect what the model learned:
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(layer.name, weights.shape, biases.shape)
```

---

## Minimal Working Example

The full pattern for Exercise 4:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([0,0,0,1], dtype=np.float32)  # AND gate

model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, y, epochs=5000, batch_size=4, verbose=0)

# Print final loss
print(f"Final loss: {history.history['loss'][-1]:.4f}")

# Predictions
probs = model.predict(X, verbose=0)
print(probs)
```

---

## Functional API — For Non-Sequential Architectures

When layers branch or merge (skip connections, multi-input models), use the functional API instead of Sequential:

```python
inputs = keras.Input(shape=(2,))
x = keras.layers.Dense(4, activation='relu')(inputs)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

You won't need this in Week 1, but it's what the course notebooks use for more complex architectures.

---

## Subclassing `Model` — The PyTorch-Style Equivalent

For full control (equivalent to `nn.Module`):

```python
class TwoLayerNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Dense(4, activation='relu')
        self.layer2 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):         # `call()` in Keras = `forward()` in PyTorch
        x = self.layer1(x)
        return self.layer2(x)
```

`call()` is Keras's equivalent of PyTorch's `forward()`. You still compile and fit the same way.

---

## GPU Verification

```python
print(tf.config.list_physical_devices('GPU'))
# Should return something like: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# Empty list means TF is not seeing your GPU
```

If it returns empty but you have an NVIDIA GPU: check that `tensorflow` (not `tensorflow-cpu`) is installed and that your CUDA/cuDNN versions match what the installed TF version expects.

---

## Key Differences to Keep in Mind vs. PyTorch

| Topic | PyTorch | TensorFlow/Keras |
|---|---|---|
| `BCEWithLogitsLoss` equivalent | `nn.BCEWithLogitsLoss()` | `loss='binary_crossentropy'` with `activation='sigmoid'` on output |
| Inference mode | `with torch.no_grad():` | `model.predict()` (no_grad is implicit) |
| Access learned weights | `model.parameters()` or `layer.weight` | `model.get_weights()` or `layer.get_weights()` |
| Custom training step | Manual loop with `loss.backward()` | `tf.GradientTape()` (rarely needed with Keras) |

---

## What Keras Is NOT Doing For You

Same caveat as PyTorch: `model.fit()` will faithfully train a bad model on bad data and report decreasing loss. The diagnostics in `debugging_nn_guide.md` apply equally here — Keras just makes it easier to overlook them because the training loop is hidden.
