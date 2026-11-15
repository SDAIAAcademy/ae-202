# %% [markdown]
# # Pre-work · Arrays, shapes and a derivative
#
# About two hours, before Day 1. The course starts from here: every lab of Day 1 assumes
# that you read the shape of an array without stopping, that broadcasting does not surprise
# you, and that you can take a derivative with the chain rule. The entry quiz on Day 1 asks
# the same kinds of question, on paper, without an assistant.
#
# **Parts**
# - [ ] Shapes and indexing · 25 min
# - [ ] Broadcasting · 25 min
# - [ ] Matrix products · 20 min
# - [ ] Softmax and cross-entropy by hand · 30 min
# - [ ] A derivative with the chain rule · 20 min
#
# AI policy: use an assistant to explain, then close it and do the exercise again alone.
# The entry quiz is without one.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-202.git"
    if not Path("/content/ae-202").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ae-202"], check=True)
    os.chdir("/content/ae-202/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1 · Shapes and indexing
#
# A batch of 6 grey images of 4 by 5 pixels is an array of shape `(6, 4, 5)`: images, rows,
# columns. The first exercises are about saying what a shape will be before you run the line.

# %%
batch = rng.random((6, 4, 5))
print("the batch:", batch.shape)
print("one image:", batch[0].shape, "| one row of every image:", batch[:, 0].shape, "| the last column of every image:", batch[:, :, -1].shape)

# %% [markdown]
# ✏️ **Exercise 1** · Return the batch with every image flattened to one row of 20 numbers:
# shape `(6, 20)`. Do not write the number 6 in your code.

# %%
def flatten(batch):
    # BEGIN SOLUTION
    return batch.reshape(len(batch), -1)
    # END SOLUTION

# %%
try:
    out = flatten(batch)
    assert out.shape == (6, 20) and np.allclose(out[2, 5], batch[2, 1, 0]), "row by row: pixel (1, 0) is number 5"
    print("✓ flatten:", out.shape)
except NotImplementedError:
    print("exercise 1 is not done yet")

# %% [markdown]
# ✏️ **Exercise 2** · Return the mean brightness of each image: one number per image, shape `(6,)`.

# %%
def brightness(batch):
    # BEGIN SOLUTION
    return batch.mean(axis=(1, 2))
    # END SOLUTION

# %%
try:
    out = brightness(batch)
    assert out.shape == (6,) and np.isclose(out[3], batch[3].mean())
    print("✓ brightness:", np.round(out, 3))
except NotImplementedError:
    print("exercise 2 is not done yet")

# %% [markdown]
# ## 2 · Broadcasting
#
# When two arrays of different shapes meet in an operation, the shapes are compared from the
# right. Two sizes agree when they are equal or when one of them is 1; a missing dimension
# counts as 1. An array of shape `(6, 20)` minus an array of shape `(20,)` works: the 20
# numbers are subtracted from every row.

# %%
flat = batch.reshape(6, 20)
print((flat - flat.mean(axis=0)).shape, "each column minus its mean")
try:
    flat - flat.mean(axis=1)
except ValueError as e:
    print("each row minus its mean, written this way, fails:", e)

# %% [markdown]
# ✏️ **Exercise 3** · Subtract from each row its own mean. The result has shape `(6, 20)` and
# every row of it has mean 0. One argument of `mean` makes the shapes agree.

# %%
def centre_rows(flat):
    # BEGIN SOLUTION
    return flat - flat.mean(axis=1, keepdims=True)
    # END SOLUTION

# %%
try:
    out = centre_rows(flat)
    assert out.shape == (6, 20) and np.allclose(out.mean(axis=1), 0)
    print("✓ centre_rows")
except NotImplementedError:
    print("exercise 3 is not done yet")

# %% [markdown]
# ✏️ **Exercise 4** · `labels` holds one class number per image, from 0 to 7. Return the one-hot
# array of shape `(6, 8)`: a 1 in the column of the label, 0 elsewhere. Compare the labels,
# as a column, with `np.arange(8)`.

# %%
labels = np.array([3, 0, 7, 3, 1, 5])


def one_hot(labels, n=8):
    # BEGIN SOLUTION
    return (labels[:, None] == np.arange(n)).astype(float)
    # END SOLUTION

# %%
try:
    out = one_hot(labels)
    assert out.shape == (6, 8) and out.sum() == 6 and out[2, 7] == 1
    print("✓ one_hot")
except NotImplementedError:
    print("exercise 4 is not done yet")

# %% [markdown]
# ## 3 · Matrix products
#
# A linear layer is a matrix product and a sum: `scores = x @ W + b`. With `x` of shape
# `(6, 20)` and 8 classes, `W` must have shape `(20, 8)` and `b` shape `(8,)`. The inner
# sizes of a product must agree; the outer sizes are the shape of the result.

# %%
W = rng.normal(0, 0.1, (20, 8))
b = np.zeros(8)

# %% [markdown]
# ✏️ **Exercise 5** · Return the scores of the batch, shape `(6, 8)`, and the predicted class of
# each image, shape `(6,)`: the column with the highest score.

# %%
def scores_and_classes(flat, W, b):
    # BEGIN SOLUTION
    scores = flat @ W + b
    return scores, scores.argmax(axis=1)
    # END SOLUTION

# %%
try:
    scores, classes = scores_and_classes(flat, W, b)
    assert scores.shape == (6, 8) and classes.shape == (6,) and np.isclose(scores[1, 2], flat[1] @ W[:, 2])
    print("✓ scores:", scores.shape, "| predicted classes:", classes, "| accuracy against the labels:", (classes == labels).mean())
except NotImplementedError:
    scores = flat @ W + b
    print("exercise 5 is not done yet")

# %% [markdown]
# ## 4 · Softmax and cross-entropy by hand
#
# The softmax turns a row of scores into probabilities: the exponential of each score,
# divided by the sum of the exponentials of the row. The cross-entropy loss of one image is
# minus the logarithm of the probability given to its true class. Day 1 uses both in every lab.

# %% [markdown]
# ✏️ **Exercise 6** · Return the softmax of each row of `scores`. Subtract the largest score of
# the row first: the result is the same, and the exponential cannot overflow.

# %%
def softmax(scores):
    # BEGIN SOLUTION
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
    # END SOLUTION

# %%
try:
    p = softmax(scores)
    assert p.shape == scores.shape and np.allclose(p.sum(axis=1), 1) and np.isfinite(softmax(scores + 1000)).all()
    print("✓ softmax: each row sums to 1; the first row:", np.round(p[0], 3))
except NotImplementedError:
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    print("exercise 6 is not done yet")

# %% [markdown]
# ✏️ **Exercise 7** · Return the mean cross-entropy loss of the batch: for each image take the
# probability of its true class, then minus the logarithm, then the mean.

# %%
def cross_entropy(p, labels):
    # BEGIN SOLUTION
    return float(-np.log(p[np.arange(len(labels)), labels]).mean())
    # END SOLUTION

# %%
try:
    loss = cross_entropy(p, labels)
    assert np.isclose(loss, -np.mean([np.log(p[i, c]) for i, c in enumerate(labels)]))
    print(f"✓ cross-entropy: {loss:.3f}. A model that gives each of 8 classes the same probability has ln 8 = {np.log(8):.3f}")
except NotImplementedError:
    print("exercise 7 is not done yet")

# %% [markdown]
# ## 5 · A derivative with the chain rule
#
# One weight, one example: the prediction is `w * x` and the loss is `L = (w * x - y) ** 2`.
# By the chain rule, `dL/dw = 2 * (w * x - y) * x`. A derivative can always be checked with
# numbers: move `w` by a small step `h` in both directions and divide the change of the loss
# by `2 * h`.

# %% [markdown]
# ✏️ **Exercise 8** · Return the derivative of the loss with respect to `w`, from the formula.

# %%
def loss(w, x, y):
    return (w * x - y) ** 2


def dloss_dw(w, x, y):
    # BEGIN SOLUTION
    return 2 * (w * x - y) * x
    # END SOLUTION

# %%
try:
    w, x, y, h = 1.5, 2.0, 7.0, 1e-5
    by_formula = dloss_dw(w, x, y)
    by_numbers = (loss(w + h, x, y) - loss(w - h, x, y)) / (2 * h)
    assert np.isclose(by_formula, by_numbers), "compare with the check by numbers"
    print(f"✓ the derivative: {by_formula:.4f} by the formula, {by_numbers:.4f} by numbers")
    print("  it is negative: a larger w lowers the loss. One step of gradient descent: w - 0.01 * derivative =", round(w - 0.01 * by_formula, 3))
except NotImplementedError:
    print("exercise 8 is not done yet")

# %% [markdown]
# ## Check yourself
#
# Answer on paper, then open the answers.
#
# 1. `a` has shape `(32, 576)` and `W` has shape `(576, 64)`. What is the shape of `a @ W`?
# 2. `x` has shape `(32, 8)` and `m` has shape `(32,)`. Why does `x - m` fail, and what fixes it?
# 3. The scores of one image over three classes are 0, 0 and 0. What is the cross-entropy loss
#    if the true class is the first?
# 4. `L = (3 * w + 1) ** 2`. What is `dL/dw` at `w = 1`?
#
# <details><summary>Answers</summary>
#
# 1. `(32, 64)`: the inner sizes (576) agree and disappear.
# 2. The shapes are compared from the right: 8 against 32. `m[:, None]`, of shape `(32, 1)`,
#    subtracts one number from each row.
# 3. Each class gets the probability 1/3, and the loss is ln 3, about 1.10.
# 4. `2 * (3 * w + 1) * 3 = 24`.
# </details>
#
# If one of the four took you more than a few minutes, go through its part again before Day 1.
