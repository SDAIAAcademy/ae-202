# %% [markdown]
# # Lab 1.2 · The training loop
#
# Day 1 · Session 2 (Writing the training loop) · 25 min.
#
# You write the loop that trains Naqsh on 3,000 grey swatches: batches, a loss, an optimiser
# and a validation pass. The starter has no training code. Your curves are then compared with
# the reference curves within a stated tolerance: that comparison is the evidence for this
# outcome.
#
# **Tasks**
# - [ ] Mini-batches from a shuffled order · 4 min
# - [ ] Model, loss and optimiser · 4 min
# - [ ] The training step · 7 min
# - [ ] The validation pass · 5 min
# - [ ] Curves against the reference · 5 min
# - [ ] ✱ Stretch · Replace your batching with a library data loader
#
# AI policy for this lab: **AI-off for your first attempt.** Write the loop yourself; an
# assistant may explain an error message afterwards.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import nawl as U

Xtr, ytr, Xva, yva, Xte, yte = U.swatches()
EPOCHS = U.pick(12, 6)
print(f"mode {U.MODE}: {len(Xtr)} training swatches, {len(Xva)} for validation, {Xtr.shape[1]} numbers each, {EPOCHS} epochs")

# %% [markdown]
# ## 1 · Mini-batches from a shuffled order
#
# One epoch visits every training example once, in a new random order, in groups of
# `batch_size`.
#
# ✏️ **Exercise** · Return a list of index tensors that together hold every index from 0 to
# `n - 1` exactly once, in the order `torch.randperm(n, generator=g)` gives. The last batch may
# be smaller.

# %%
def batches(n, batch_size, g):
    # BEGIN SOLUTION
    order = torch.randperm(n, generator=g)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _batches(n, batch_size, g):
    order = torch.randperm(n, generator=g)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]

try:
    bs = batches(10, 4, torch.Generator().manual_seed(0))
except NotImplementedError:
    batches = _batches
    bs = batches(10, 4, torch.Generator().manual_seed(0))
assert [len(b) for b in bs] == [4, 4, 2], [len(b) for b in bs]
assert sorted(torch.cat(bs).tolist()) == list(range(10)), "every index once"
print("three batches of ten indices:", [b.tolist() for b in bs])

# %% [markdown]
# ## 2 · Model, loss and optimiser
#
# The model is the network of Lab 1.1 with 64 hidden units. `nn.Linear` holds a weight matrix
# and a bias; the optimiser holds the rule that changes them.
#
# ✏️ **Exercise** · Return a model `576 -> 64 -> relu -> 8` built with `nn.Sequential`, and an
# Adam optimiser over its parameters with learning rate `lr`.

# %%
def make_model_and_optimiser(lr=1e-3):
    torch.manual_seed(0)   # the same starting weights for everyone
    # BEGIN SOLUTION
    model = nn.Sequential(nn.Linear(576, 64), nn.ReLU(), nn.Linear(64, 8))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimiser
    # END SOLUTION

# %%
try:
    model, optimiser = make_model_and_optimiser()
except NotImplementedError:
    torch.manual_seed(0)
    model = U.mlp(64)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
assert U.n_params(model) == 576 * 64 + 64 + 64 * 8 + 8, U.n_params(model)
print(f"{U.n_params(model):,} weights; output for two swatches has shape {tuple(model(Xtr[:2]).shape)}")

# %% [markdown]
# ## 3 · The training step
#
# Five lines, always in this order: forward, loss, clear the old gradients, backward, update.
# `F.cross_entropy` takes the raw outputs of the network (logits) and the class numbers.
#
# ✏️ **Exercise** · Complete the step inside the loop. It must leave the loss of the batch in
# a tensor named `loss`.

# %%
def train_one_epoch(model, optimiser, X, y, batch_size, g):
    model.train()
    total = 0.0
    for b in batches(len(X), batch_size, g):
        # BEGIN SOLUTION
        logits = model(X[b])
        loss = F.cross_entropy(logits, y[b])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # END SOLUTION
        total += loss.item() * len(b)
    return total / len(X)

# %% [markdown]
# ## 4 · The validation pass
#
# Validation measures; it must not learn. Put the model in evaluation mode and switch
# gradient recording off.
#
# ✏️ **Exercise** · Return `(mean loss, accuracy)` on `X, y`.

# %%
def validate(model, X, y):
    # BEGIN SOLUTION
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy
    # END SOLUTION

# %%
# The loop, with a safety net for the two exercises
g = torch.Generator().manual_seed(0)
history = {"train_loss": [], "val_loss": [], "val_acc": []}
try:
    for epoch in range(EPOCHS):
        tl = train_one_epoch(model, optimiser, Xtr, ytr, 32, g)
        vl, va = validate(model, Xva, yva)
        for k, v in zip(history, (tl, vl, va)):
            history[k].append(v)
        print(f"epoch {epoch + 1:2d}  train loss {tl:.3f}  val loss {vl:.3f}  val acc {va:.3f}")
except NotImplementedError:
    print("(exercise not done yet: showing the reference run)")
    torch.manual_seed(0)
    model = U.mlp(64)
    history = U.fit(model, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=1e-3, bs=32)
assert history["val_loss"][-1] < history["val_loss"][0], "the validation loss must fall"

# %% [markdown]
# ## 5 · Curves against the reference
#
# The reference run used the same model, data, learning rate and batch size. Your numbers
# need not be identical (the order of the batches depends on how you drew them), so the
# comparison has a tolerance: **the last validation loss within 0.15 and the last validation
# accuracy within 0.08 of the reference at the same epoch.**

# %%
ref = U.reference("d1_2_curves") if (U.REF / "d1_2_curves.json").exists() else history
k = EPOCHS - 1
d_loss = abs(history["val_loss"][k] - ref["val_loss"][k])
d_acc = abs(history["val_acc"][k] - ref["val_acc"][k])
print(f"epoch {EPOCHS}: your val loss {history['val_loss'][k]:.3f} (reference {ref['val_loss'][k]:.3f}), "
      f"your val acc {history['val_acc'][k]:.3f} (reference {ref['val_acc'][k]:.3f})")
print("within tolerance:", d_loss <= 0.15 and d_acc <= 0.08)
assert d_loss <= 0.15 and d_acc <= 0.08, "outside the tolerance: check zero_grad, the loss input and the batch order"
if not U.QUICK:
    U.save_reference("d1_2_curves", history)
U.save_run("d1_2_curves", history)
fig = U.plot_curves({"yours": history, "reference": {key: v[:EPOCHS] for key, v in ref.items()}}, "Swatches, 576-64-8")

# %% [markdown]
# **Look at** the two losses. Training loss (dashed) and validation loss (solid) fall
# together: the network is still learning things that hold for swatches it has not seen.
# Chance on eight classes is a loss of 2.08 and an accuracy of 0.125.

# %% [markdown]
# ## ✱ Stretch · A data loader
#
# Replace `batches` with `torch.utils.data.DataLoader(TensorDataset(Xtr, ytr), batch_size=32,
# shuffle=True)`. What does the loader add that your function lacks (think of data that does
# not fit in memory)?

# %% [markdown]
# ## Check yourself
#
# 1. What happens to the curve if you delete `optimiser.zero_grad()`?
# 2. Why `model.eval()` and `torch.no_grad()` in the validation pass, when this model has no
#    dropout?
# 3. Your run and a neighbour's differ in the third decimal. Is something wrong?
#
# <details><summary>Answers</summary>
#
# 1. Gradients add up across steps, so every update uses the sum of all gradients so far: the
#    loss jumps around or grows.
# 2. `no_grad` saves the memory and time of recording a graph nobody uses. `eval` changes
#    nothing for this model, and is the habit that saves you the day a layer behaves
#    differently in training (dropout, batch normalisation).
# 3. No. A different batch order gives a different path to a similar place. That is why the
#    outcome is checked within a tolerance.
# </details>
#
# ## Carry forward
#
# Your loop is in `nawl` as `U.fit` and `U.evaluate` from here on. Your curves are in
# `runs/d1_2_curves.json`.
