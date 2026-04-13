# Quiz — Week 1: Perceptron and Neural Network Foundations

Write your answers in `W01_perceptron_answers.md` before checking the answer key. Aim for 2–5 sentences per answer — you're being graded on reasoning, not just the correct label.

---

## Question 1

You train a single perceptron with sigmoid activation on the AND gate and it converges successfully. You then train the same architecture on XOR and the loss plateaus around 0.693 (which is `-log(0.5)`). 

What does a loss of `~0.693` tell you about the model's predictions, and why does a single perceptron structurally fail on XOR regardless of how long you train it or how you tune the learning rate?

## Answer 1

---

## Question 2

A colleague shows you two training loss curves for the same model and dataset. Curve A decreases smoothly to near-zero over 1000 epochs. Curve B oscillates wildly — spiking up and down — and never settles below 0.4.

What is the most likely difference between the two runs, and what would you change to fix Curve B? Explain the mechanism, not just the remedy.

## Answer 2

---

## Question 3

You're building a binary classifier to identify whether a map tile contains a highway (1) or not (0). Your input features are: road pixel density (0.0–1.0), average road width in meters (3–40), and number of intersections (0–200).

Without normalising these features, what problem might occur during training, and why? How would you fix it?

## Answer 3

---

## Question 4

Explain the role of the bias term `w₀` in a perceptron. What happens geometrically if you remove it? Give a concrete example where a bias-free perceptron would fail to learn a correct decision boundary even for a linearly separable dataset.

## Answer 4

---

## Question 5

PyTorch's `loss.backward()` populates `.grad` attributes on each parameter tensor. `optimizer.step()` then uses those gradients to update the parameters.

In your NumPy implementation (Exercise 1), which lines correspond to `loss.backward()` and which lines correspond to `optimizer.step()`? Why does PyTorch require `optimizer.zero_grad()` before each backward pass, and what bug does omitting it cause?

## Answer 5

---

## Grade Log
