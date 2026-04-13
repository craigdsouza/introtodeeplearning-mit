# Answers — Week 1: Perceptron and Neural Network Foundations

> Read this only after writing your own answers in the quiz file.

---

## Question 1

**A loss of ~0.693 means the model is predicting 0.5 probability for every input — it has learned nothing.**

Binary cross-entropy for a prediction of 0.5 is `-log(0.5) ≈ 0.693`. The model is outputting maximum uncertainty, which is what you'd get from random initialisation. The reason a single perceptron can't learn XOR is geometric: a perceptron's decision boundary is always a hyperplane (a straight line in 2D). XOR requires two separate regions — `(0,1)` and `(1,0)` must be class 1, but they sit on diagonally opposite corners of the unit square, with `(0,0)` and `(1,1)` (class 0) in between. No single straight line can separate these correctly. This is the definition of *linear inseparability*, and no amount of training or learning-rate tuning can overcome a structural constraint in the model's hypothesis class.

---

## Question 2

**Curve B has a learning rate that is too large.**

When the learning rate `η` is too large, gradient descent takes steps that overshoot the minimum. Imagine a parabolic loss surface: instead of rolling smoothly down to the bottom, the update lands on the far side of the bowl — where the gradient points back the other way — and the loss bounces back up. With a very large `η`, this oscillation can grow over time (divergence). The fix is to reduce `η`. A practical approach: start with `η = 0.01`, plot the loss, and increase until you see instability, then back off. Learning rate schedules (e.g. cosine annealing, exponential decay) reduce `η` over training to allow large initial steps and fine-grained convergence later.

---

## Question 3

**Without normalisation, features with larger numerical ranges dominate the gradient, causing slow or biased convergence.**

The gradient of the loss with respect to each weight `wᵢ` is proportional to the magnitude of the corresponding feature `xᵢ`. Road width (range 3–40) and intersection count (range 0–200) are on very different scales. The weights for large-magnitude features will have larger gradients, so gradient descent will primarily optimise those directions and move very little in the direction of small-magnitude features (road pixel density, range 0–1). The loss surface becomes a long narrow valley — easy to overshoot in one direction, slow to converge in the other. Fix: standardise each feature to zero mean and unit variance: `x_norm = (x - mean) / std`. After normalisation, all features contribute comparably to the gradient.

---

## Question 4

**The bias shifts the decision boundary away from the origin, allowing the perceptron to fit datasets that are not centred at zero.**

Without a bias, the decision boundary (the hyperplane where `wᵀx = 0`) must pass through the origin. Consider classifying whether a road segment is a highway based on a single feature: average speed (km/h). Highways might have speed > 80 km/h. The correct boundary is at 80, but without a bias, the boundary is forced through 0, and no weight value can move it to 80. With bias `w₀`, the decision is `w·speed + w₀ = 0`, so the boundary is at `speed = -w₀/w`, which can be placed anywhere. In general, removing the bias reduces the model's expressiveness and makes it fail on any dataset whose correct boundary doesn't pass through the origin — which is most real datasets.

---

## Question 5

**`loss.backward()` corresponds to computing `dw, db = gradients(X, y, y_hat)`. `optimizer.step()` corresponds to `w -= lr * dw; b -= lr * db`.**

In the NumPy implementation, `gradients()` manually computes `∂J/∂w = (Xᵀ(ŷ - y)) / n` and `∂J/∂b = mean(ŷ - y)` — this is the chain rule applied by hand. PyTorch's autograd engine does this automatically for any computation graph built with torch tensors. The update lines `w -= lr * dw` and `b -= lr * db` are what `optimizer.step()` executes, using the `.grad` values that `backward()` populated.

`optimizer.zero_grad()` must be called before each backward pass because PyTorch **accumulates** gradients by default — it adds the new gradient to whatever `.grad` already contains. If you omit `zero_grad()`, gradients from epoch 1, epoch 2, etc. all pile up, and the effective gradient by epoch 10 is ten times too large. The result is erratic training that looks exactly like Curve B from Question 2: the learning rate appears to grow every epoch even though `η` is fixed.

```python
# Correct PyTorch training loop
for epoch in range(epochs):
    optimizer.zero_grad()   # ← reset accumulated gradients to 0
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    loss.backward()         # ← compute ∂loss/∂(every parameter)
    optimizer.step()        # ← w = w - lr * w.grad for each parameter
```
