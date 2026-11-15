# %% [markdown]
# # Lab 1.1 · A gradient by hand and by autograd
#
# Day 1 · Session 1 (Tensors, shapes and gradients) · 25 min.
#
# Naqsh, the cooperative's pattern classifier, starts as a two-layer network on grey swatches.
# Before any library trains it, you compute its gradients yourself with the chain rule in
# NumPy, then let automatic differentiation do the same and compare the numbers.
#
# **Tasks**
# - [ ] Shapes and broadcasting on a batch of swatches · 5 min
# - [ ] The forward pass by hand · 5 min
# - [ ] The backward pass with the chain rule · 8 min
# - [ ] Automatic differentiation and the comparison · 4 min
# - [ ] Number formats: 32 and 64 bits · 3 min
# - [ ] ✱ Stretch · Check one weight with a finite difference
#
# AI policy for this lab: an assistant may explain an error or a shape; you write the
# forward and backward passes yourself.

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
import torch
import nawl as U

Xtr, ytr, *_ = U.swatches()
X = Xtr[:4].numpy().astype(np.float64)      # four swatches, 576 grey values each
y = ytr[:4].numpy()
rng = np.random.default_rng(11)
W1, b1 = rng.normal(0, 0.1, (576, 5)), np.zeros(5)   # layer 1: 576 inputs, 5 hidden units
W2, b2 = rng.normal(0, 0.5, (5, 8)), np.zeros(8)     # layer 2: 5 hidden units, 8 classes
print("X", X.shape, "y", y, [U.CLASSES[i] for i in y])

# %% [markdown]
# ## 1 · Shapes and broadcasting on a batch of swatches
#
# Two arrays combine when their shapes match from the right, or one of the two sizes is 1.
# `X` is `(4, 576)`. A mean over the batch has shape `(576,)`: it lines up with the last
# axis. A mean over the pixels of each swatch has shape `(4,)`: it does not, unless you keep
# the axis.
#
# ✏️ **Exercise** · Return `X` with the mean of each swatch subtracted from that swatch
# (every row then has mean 0).

# %%
def centre_each_swatch(X):
    # BEGIN SOLUTION
    return X - X.mean(axis=1, keepdims=True)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    Xc = centre_each_swatch(X)
except NotImplementedError:
    Xc = X - X.mean(axis=1, keepdims=True)
assert Xc.shape == (4, 576), Xc.shape
assert np.allclose(Xc.mean(axis=1), 0), "each row must have mean 0: take the mean over axis 1 and keep the axis"
print("row means after centring:", np.abs(np.round(Xc.mean(axis=1), 12)))
print("(X - X.mean(axis=0)).shape =", (X - X.mean(axis=0)).shape, "  the per-pixel mean, shape (576,), broadcasts over the rows")
try:
    X - X.mean(axis=1)
except ValueError as e:
    print("X - X.mean(axis=1) fails:", e)

# %% [markdown]
# ## 2 · The forward pass by hand
#
# The network: `z1 = X W1 + b1`, `h = relu(z1)`, `z2 = h W2 + b2`, `p = softmax(z2)`, and the
# loss is the mean over the batch of `-log p[true class]` (cross-entropy).
#
# ✏️ **Exercise** · Return `z1, h, z2, p, loss`.

# %%
def forward(X, y, W1, b1, W2, b2):
    # BEGIN SOLUTION
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y]).mean()
    return z1, h, z2, p, loss
    # END SOLUTION

# %%
def _forward(X, y, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    e = np.exp(z2 - z2.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return z1, h, z2, p, -np.log(p[np.arange(len(y)), y]).mean()

try:
    z1, h, z2, p, loss = forward(X, y, W1, b1, W2, b2)
except NotImplementedError:
    forward = _forward
    z1, h, z2, p, loss = forward(X, y, W1, b1, W2, b2)
assert z1.shape == (4, 5) and z2.shape == (4, 8) and np.allclose(p.sum(axis=1), 1)
assert abs(loss - _forward(X, y, W1, b1, W2, b2)[4]) < 1e-12, "the loss differs from the reference"
print(f"loss {loss:.6f}   (guessing among 8 classes gives ln 8 = {np.log(8):.3f})")

# %% [markdown]
# ## 3 · The backward pass with the chain rule
#
# Work from the loss back to the inputs, one local derivative at a time:
#
# - softmax with cross-entropy: `dz2 = (p - onehot(y)) / N`
# - a linear layer `z = a W + b`: `dW = a.T @ dz`, `db = dz.sum(axis=0)`, and `da = dz @ W.T`
# - relu: the gradient passes where `z1 > 0` and is zero elsewhere
#
# ✏️ **Exercise** · Return `dW1, db1, dW2, db2`. Every gradient has the shape of the thing it
# is the gradient of: check that first when a number is wrong.

# %%
def backward(X, y, z1, h, p, W2):
    # BEGIN SOLUTION
    n = len(y)
    dz2 = p.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)
    dh = dz2 @ W2.T
    dz1 = dh * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2
    # END SOLUTION

# %%
def _backward(X, y, z1, h, p, W2):
    n = len(y)
    dz2 = p.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    return X.T @ dz1, dz1.sum(axis=0), h.T @ dz2, dz2.sum(axis=0)

try:
    grads = backward(X, y, z1, h, p, W2)
except NotImplementedError:
    backward = _backward
    grads = backward(X, y, z1, h, p, W2)
for g, w, name in zip(grads, (W1, b1, W2, b2), ("dW1", "db1", "dW2", "db2")):
    assert g.shape == w.shape, f"{name} has shape {g.shape}, its weight has {w.shape}"
    print(f"{name:4s} shape {str(g.shape):9s} largest absolute value {np.abs(g).max():.6f}")

# %% [markdown]
# ## 4 · Automatic differentiation and the comparison
#
# PyTorch records every operation on a tensor that has `requires_grad=True` in a graph and
# walks that graph backwards when you call `.backward()`.
#
# ✏️ **Exercise** · Compute the same loss with torch operations on the four tensors below and
# call `.backward()` on it. Return the loss tensor.

# %%
tW1, tb1, tW2, tb2 = (torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2))
tX, ty = torch.tensor(X), torch.tensor(y)


def loss_with_autograd():
    # BEGIN SOLUTION
    z2 = torch.relu(tX @ tW1 + tb1) @ tW2 + tb2
    loss = torch.nn.functional.cross_entropy(z2, ty)
    loss.backward()
    return loss
    # END SOLUTION

# %%
try:
    tloss = loss_with_autograd()
except NotImplementedError:
    tloss = torch.nn.functional.cross_entropy(torch.relu(tX @ tW1 + tb1) @ tW2 + tb2, ty)
    tloss.backward()
print(f"loss by hand {loss:.12f}   by autograd {tloss.item():.12f}")
diffs = {}
for g, t, name in zip(grads, (tW1, tb1, tW2, tb2), ("dW1", "db1", "dW2", "db2")):
    diffs[name] = float(np.abs(g - t.grad.numpy()).max())
    print(f"{name:4s} largest difference between hand and autograd: {diffs[name]:.2e}")
assert max(diffs.values()) < 1e-10, "hand and autograd disagree: check the shapes and the division by N"
U.save_reference("d1_1_diffs", {"loss": float(loss), "diffs_float64": diffs})

# %% [markdown]
# **Look at** the size of the differences. The two computations do the same arithmetic in a
# different order, so they agree to the last digits a 64-bit number holds.

# %% [markdown]
# ## 5 · Number formats: 32 and 64 bits
#
# Deep learning runs in 32-bit floats (and often 16-bit on a graphics processor): half the
# memory, about 7 significant digits in place of 16.

# %%
f32 = lambda a: a.astype(np.float32)
z1f, hf, z2f, pf, lossf = forward(f32(X), y, f32(W1), f32(b1), f32(W2), f32(b2))
gf = backward(f32(X), y, z1f, hf, pf, f32(W2))
d32 = max(float(np.abs(a - b).max()) for a, b in zip(gf, grads))
print(f"loss in 32 bits {lossf:.12f}, in 64 bits {loss:.12f}")
print(f"largest gradient difference between 32 and 64 bits: {d32:.2e}")
print("bytes per number:", np.float32().nbytes, "against", np.float64().nbytes, "| device of the tensors:", tX.device)
assert d32 < 1e-5

# %% [markdown]
# **Look at** the gap: many orders of magnitude larger than the gap between hand and
# autograd, and still far below anything training would notice. This is why "do the numbers
# match?" always needs a tolerance, and why the tolerance depends on the format.

# %% [markdown]
# ## ✱ Stretch · A finite difference
#
# Nudge one weight, `W2[0, 0]`, by `eps = 1e-6`, recompute the loss, and compare
# `(loss_plus - loss_minus) / (2 * eps)` with `dW2[0, 0]`. How many digits agree? Try
# `eps = 1e-12` and explain why it gets worse.

# %% [markdown]
# ## Check yourself
#
# 1. `X` is `(4, 576)` and `v` is `(4,)`. Why does `X - v` fail while `X - v[:, None]` works?
# 2. Why is `dz2` divided by `N`?
# 3. Hand and autograd agree almost to the last digit in 64 bits. Would you expect the same
#    agreement between a 32-bit and a 64-bit run?
#
# <details><summary>Answers</summary>
#
# 1. Shapes align from the right: 576 against 4 does not match. `v[:, None]` is `(4, 1)`, and
#    a size of 1 stretches over the 576 columns.
# 2. The loss is the mean over the batch, so each example carries a factor 1/N.
# 3. No. Each 32-bit operation rounds to about 7 digits, so the gap is around 1e-8, not 1e-17.
# </details>
#
# ## Carry forward
#
# From Lab 1.2 on, `loss.backward()` computes every gradient. You now know what it computes,
# and that a gradient always has the shape of its weight.
