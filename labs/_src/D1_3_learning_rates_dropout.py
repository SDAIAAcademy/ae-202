# %% [markdown]
# # Lab 1.3 · Three learning rates and dropout
#
# Day 1 · Session 3 (Learning rates, batches and regularisation) · 25 min.
#
# The same network, the same swatches, the same loop: you change one setting at a time and
# read what the curves say. First the learning rate, then dropout and weight decay on a
# training set of only 400 swatches, then the epoch at which to stop.
#
# **Tasks**
# - [ ] Three learning rates, one model · 8 min
# - [ ] Read the three curves · 4 min
# - [ ] Dropout and weight decay on 400 swatches · 8 min
# - [ ] The early-stopping epoch · 5 min
# - [ ] ✱ Stretch · Repeat one run with batches of 8 and of 256
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.

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

import math
import torch
import nawl as U

Xtr, ytr, Xva, yva, Xte, yte = U.swatches()
print(f"mode {U.MODE}")

# %% [markdown]
# ## 1 · Three learning rates, one model
#
# `U.fit` is the loop you wrote in Lab 1.2. Here it runs with plain gradient descent
# (`optimiser="sgd"`), where the learning rate is the step size and nothing else.
#
# ✏️ **Exercise** · Train a fresh `U.mlp(64)` for `EPOCHS` epochs at each learning rate in
# `RATES` and return `{rate: history}`. Call `torch.manual_seed(0)` before creating each model
# so that the three start from the same weights.

# %%
RATES = [0.001, 0.1, 3.0]
EPOCHS = U.pick(8, 5)


def three_runs():
    # BEGIN SOLUTION
    out = {}
    for lr in RATES:
        torch.manual_seed(0)
        out[lr] = U.fit(U.mlp(64), Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=lr, optimiser="sgd", verbose=False)
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    runs = three_runs()
except NotImplementedError:
    runs = {}
    for lr in RATES:
        torch.manual_seed(0)
        runs[lr] = U.fit(U.mlp(64), Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=lr, optimiser="sgd", verbose=False)
for lr, h in runs.items():
    print(f"lr {lr:<6} val loss per epoch " + " ".join(f"{v:7.2f}" for v in h["val_loss"]) + f"   last acc {h['val_acc'][-1]:.3f}")
assert set(runs) == set(RATES)
fig = U.plot_curves({f"lr {lr}": {k: [min(v, 4.0) for v in vals] for k, vals in h.items()} for lr, h in runs.items()},
                    "Three learning rates (losses above 4 are drawn at 4)")

# %% [markdown]
# ## 2 · Read the three curves
#
# Chance is a loss of `ln 8 = 2.08`.
#
# ✏️ **Exercise** · Write the rule you just read off the plot as code. Return `"too high"`
# when the validation loss ever rose above 2.5 (well above chance) or is not a number,
# `"too low"` when it moved by less than 0.05 from its first epoch, and `"usable"` otherwise.

# %%
def diagnose(history):
    # BEGIN SOLUTION
    v = history["val_loss"]
    if any(math.isnan(x) or x > 2.5 for x in v):
        return "too high"
    if abs(v[-1] - v[0]) < 0.05:
        return "too low"
    return "usable"
    # END SOLUTION

# %%
def _diagnose(history):
    v = history["val_loss"]
    if any(math.isnan(x) or x > 2.5 for x in v):
        return "too high"
    return "too low" if abs(v[-1] - v[0]) < 0.05 else "usable"

try:
    verdicts = {lr: diagnose(h) for lr, h in runs.items()}
except NotImplementedError:
    verdicts = {lr: _diagnose(h) for lr, h in runs.items()}
print(verdicts)
assert verdicts == {0.001: "too low", 0.1: "usable", 3.0: "too high"}, verdicts
U.save_reference("d1_3_rates", {"mode": U.MODE, "runs": {str(lr): {"val_loss": h["val_loss"], "val_acc": h["val_acc"], "verdict": verdicts[lr]} for lr, h in runs.items()}})

# %% [markdown]
# **Look at** the middle curve again: usable is not the same as best. The search for a
# learning rate is a sweep over powers of ten, read from curves like these, and it comes
# before every other tuning.

# %% [markdown]
# ## 3 · Dropout and weight decay on 400 swatches
#
# With 400 training swatches and 256 hidden units the network can memorise. Three runs with
# Adam: plain, with dropout 0.5 after the hidden layer, and with weight decay 0.001. Watch
# the **gap** between validation and training loss, and watch the validation loss itself.
#
# ✏️ **Exercise** · Return the gap at the last epoch: validation loss minus training loss.

# %%
def gap(history):
    # BEGIN SOLUTION
    return history["val_loss"][-1] - history["train_loss"][-1]
    # END SOLUTION

# %%
SMALL, E2 = 400, U.pick(60, 25)
settings = {"plain": {}, "dropout 0.5": {"dropout": 0.5}, "weight decay 0.001": {"weight_decay": 1e-3}}
small_runs = {}
for name, kw in settings.items():
    torch.manual_seed(0)
    model = U.mlp(256, dropout=kw.get("dropout", 0.0))
    small_runs[name] = U.fit(model, Xtr[:SMALL], ytr[:SMALL], Xva, yva, epochs=E2, lr=1e-3,
                             weight_decay=kw.get("weight_decay", 0.0), verbose=False)
try:
    gap(small_runs["plain"])
except NotImplementedError:
    gap = lambda h: h["val_loss"][-1] - h["train_loss"][-1]
table = {}
print(f"{'run':20s} {'train loss':>10s} {'val loss':>9s} {'gap':>6s} {'val acc':>8s}")
for name, h in small_runs.items():
    table[name] = {"train_loss": h["train_loss"][-1], "val_loss": h["val_loss"][-1], "gap": gap(h), "val_acc": h["val_acc"][-1]}
    print(f"{name:20s} {h['train_loss'][-1]:10.3f} {h['val_loss'][-1]:9.3f} {gap(h):6.3f} {h['val_acc'][-1]:8.3f}")
assert table["plain"]["gap"] > 0, "on 400 swatches the plain run must show a gap"
assert table["dropout 0.5"]["gap"] < table["plain"]["gap"], "dropout must narrow the gap"
U.save_reference("d1_3_regularisation_" + U.MODE, {"epochs": E2, "table": table})
fig = U.plot_curves(small_runs, f"400 swatches, {E2} epochs")

# %% [markdown]
# **Look at** two columns. The gap shrinks with dropout and with weight decay: both make
# memorising harder. The validation loss barely moves. Regularisation keeps a network from
# fooling you; it does not replace the 2,600 swatches this run is missing. Compare with
# Lab 1.2, which reached a lower validation loss in a fifth of the epochs with all 3,000.

# %% [markdown]
# ## 4 · The early-stopping epoch
#
# Training longer keeps lowering the training loss. The validation loss stops improving at
# some epoch: that is where you keep the weights.
#
# ✏️ **Exercise** · Return the epoch (counting from 1) whose weights early stopping with
# `patience` would keep: the epoch of the lowest validation loss seen before the loss failed
# to improve for `patience` epochs in a row (or the best epoch overall if that never happens).

# %%
def early_stop_epoch(val_loss, patience=3):
    # BEGIN SOLUTION
    best, best_epoch, waited = float("inf"), 0, 0
    for epoch, v in enumerate(val_loss, start=1):
        if v < best:
            best, best_epoch, waited = v, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_epoch
    # END SOLUTION

# %%
def _early(val_loss, patience=3):
    best, best_epoch, waited = float("inf"), 0, 0
    for epoch, v in enumerate(val_loss, start=1):
        if v < best:
            best, best_epoch, waited = v, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_epoch

try:
    early_stop_epoch([3, 2, 1])
except NotImplementedError:
    early_stop_epoch = _early
assert early_stop_epoch([2.0, 1.5, 1.2, 1.3, 1.25, 1.21, 1.0], patience=3) == 3
assert early_stop_epoch([2.0, 1.5, 1.6, 1.4], patience=3) == 4
for name, h in small_runs.items():
    e3, e10 = early_stop_epoch(h["val_loss"], 3), early_stop_epoch(h["val_loss"], 10)
    print(f"{name:20s} patience 3 keeps epoch {e3:2d} (val loss {h['val_loss'][e3 - 1]:.3f});"
          f" patience 10 keeps epoch {e10:2d} (val loss {h['val_loss'][e10 - 1]:.3f})")

# %% [markdown]
# **Look at** the two patience values. A validation curve on 500 swatches is noisy: a
# patience of 3 can stop on a bump. Patience is a setting you choose from the noise of your
# own curve.

# %% [markdown]
# ## ✱ Stretch · Batch size
#
# Run the usable learning rate again with `bs=8` and with `bs=256`. Which takes longer per
# epoch? Which curve is noisier? Why does the larger batch need a larger learning rate to
# make the same progress per epoch?

# %% [markdown]
# ## Check yourself
#
# 1. A validation loss sits at 2.08 for eight epochs on an eight-class problem. Name two
#    different causes.
# 2. Dropout narrowed the gap and left the validation loss almost unchanged. Was it useless?
# 3. Why is the early-stopping epoch chosen on the validation set and not on the test set?
#
# <details><summary>Answers</summary>
#
# 1. A learning rate far too low (the weights barely move), or a network that receives no
#    usable signal at all (Lab 1.4 shows three ways). 2.08 is ln 8: the loss of guessing.
# 2. No: it told you that the training loss of the plain run was flattering. It did not add
#    information the 400 swatches do not hold.
# 3. Choosing anything on the test set makes the test result optimistic. The test set is
#    used once, at the end.
# </details>
#
# ## Carry forward
#
# The sweep over learning rates and the gap column come back on Day 2, where a fine-tuned
# network overfits 800 photographs within a few epochs.
