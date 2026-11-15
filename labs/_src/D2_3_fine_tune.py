# %% [markdown]
# # Lab 2.3 · Fine-tune the last block
#
# Day 2 · Session 3 (Fine-tuning with little data) · 25 min.
#
# Frozen features left the backbone as it was. Now you let its last block change: you freeze
# everything before it, add augmentation so that 800 photographs go further, train the block
# and the new head at two learning rates, and put three approaches in one table, on one test
# set: from scratch, frozen features, partial fine-tuning.
#
# **Tasks**
# - [ ] Freeze all but the last block and the head · 5 min
# - [ ] Augmentation that keeps the label true · 5 min
# - [ ] Fine-tune with two learning rates · 9 min
# - [ ] The results table on one test set · 6 min
# - [ ] ✱ Stretch · Unfreeze one more block and measure the cost in time
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# This lab starts from supplied results: it does not need your Lab 2.2.

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

import json
import time
import numpy as np
import torch
import torch.nn as nn
import nawl as U

Xsm, ysm = U.images("small", U.pick(None, 240))
Xva, yva = U.images("val", U.pick(500, 200))
Xte, yte = U.images("test", U.pick(None, 300))
Psm, Pva, Pte = (U.to_backbone_input(X) for X in (Xsm, Xva, Xte))   # resized to 64, normalised
scratch = json.loads((U.DATA / "runs" / "day2_scratch.json").read_text(encoding="utf-8"))
frozen = json.loads((U.DATA / "runs" / "day2_frozen.json").read_text(encoding="utf-8"))
print(f"mode {U.MODE}: {len(Xsm)} labelled photographs; supplied test accuracy: from scratch"
      f" {scratch['test_accuracy']:.3f}, frozen features {frozen['test_accuracy']:.3f}")

# %% [markdown]
# ## 1 · Freeze all but the last block and the head
#
# A weight is frozen when `requires_grad` is `False`: no gradient is computed for it and the
# optimiser never moves it.
#
# ✏️ **Exercise** · In the ResNet-18 `m`: replace `m.fc` by a new `nn.Linear(512, 8)`, set
# `requires_grad = False` on every parameter, then `True` on the parameters of `m.layer4` and
# `m.fc`. Return `m`.

# %%
def freeze_all_but_last_block(m):
    # BEGIN SOLUTION
    m.fc = nn.Linear(512, 8)
    for p in m.parameters():
        p.requires_grad = False
    for p in list(m.layer4.parameters()) + list(m.fc.parameters()):
        p.requires_grad = True
    return m
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _freeze(m):
    m.fc = nn.Linear(512, 8)
    for p in m.parameters():
        p.requires_grad = False
    for p in list(m.layer4.parameters()) + list(m.fc.parameters()):
        p.requires_grad = True
    return m

torch.manual_seed(0)
try:
    m = freeze_all_but_last_block(U.backbone())
except NotImplementedError:
    torch.manual_seed(0)
    m = _freeze(U.backbone())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
total = U.n_params(m)
assert m.fc.out_features == 8 and not m.conv1.weight.requires_grad and m.layer4[0].conv1.weight.requires_grad
print(f"trainable: {trainable / 1e6:.2f} of {total / 1e6:.2f} million weights ({100 * trainable / total:.0f}%)")
model = U.partial_backbone(m)   # runs the frozen blocks without gradients and in evaluation mode

# %% [markdown]
# **Look at** the share: the last block alone holds three quarters of the network's weights.
# "Only the last block" is still 8 million numbers fitted to 800 photographs.

# %% [markdown]
# ## 2 · Augmentation that keeps the label true
#
# A flipped or slightly shifted chevron is still a chevron. Showing changed copies makes it
# harder to memorise 800 photographs. An augmentation must never change the label: for these
# patterns flips and small shifts are safe.
#
# ✏️ **Exercise** · Return a changed copy of the batch `X` (N x 3 x H x W): with probability
# 0.5 flip it left to right (`X.flip(3)`), with probability 0.5 flip it top to bottom
# (`X.flip(2)`), then shift it by up to 4 pixels in each direction with `torch.roll`. Draw
# the random numbers with `torch.rand(1, generator=g)` and `torch.randint(-4, 5, (2,), generator=g)`.

# %%
def augment(X, g=None):
    # BEGIN SOLUTION
    if torch.rand(1, generator=g).item() < 0.5:
        X = X.flip(3)
    if torch.rand(1, generator=g).item() < 0.5:
        X = X.flip(2)
    dx, dy = torch.randint(-4, 5, (2,), generator=g).tolist()
    return torch.roll(X, (dy, dx), (2, 3))
    # END SOLUTION

# %%
try:
    A = augment(Xsm[:8], torch.Generator().manual_seed(3))
except NotImplementedError:
    augment = U.augment
    A = augment(Xsm[:8], torch.Generator().manual_seed(3))
assert A.shape == Xsm[:8].shape
assert torch.allclose(A.sort(dim=-1).values.sum(), Xsm[:8].sort(dim=-1).values.sum(), rtol=1e-4), "flips and rolls keep every pixel value"
fig = U.show_images(torch.cat([Xsm[:8], A]), [U.CLASSES[i] for i in ysm[:8]] + ["augmented"] * 8, cols=8)

# %% [markdown]
# ## 3 · Fine-tune with two learning rates
#
# The pretrained block needs small steps: its weights are already good, and large steps
# destroy them. The new head starts from random numbers and can take larger ones.
#
# ✏️ **Exercise** · Return the two parameter groups for the optimiser: `m.layer4` at `3e-4`
# and `m.fc` at `3e-3`, as a list of dictionaries with the keys `"params"` and `"lr"`.

# %%
def parameter_groups(m):
    # BEGIN SOLUTION
    return [{"params": m.layer4.parameters(), "lr": 3e-4},
            {"params": m.fc.parameters(), "lr": 3e-3}]
    # END SOLUTION

# %%
try:
    groups = parameter_groups(m)
except NotImplementedError:
    groups = [{"params": m.layer4.parameters(), "lr": 3e-4}, {"params": m.fc.parameters(), "lr": 3e-3}]
assert [g["lr"] for g in groups] == [3e-4, 3e-3]
EPOCHS = U.pick(4, 1)
t = time.time()
history = U.fit(model, Psm, ysm, Pva, yva, epochs=EPOCHS, bs=32, params=groups, augment=augment, seed=0)
seconds = time.time() - t
print(f"{EPOCHS} epoch(s) in {seconds:.0f} s on this processor")

# %% [markdown]
# **Look at** the training loss against the validation loss, epoch by epoch. If the training
# loss runs far below the validation loss, the block is memorising: stop, do not add epochs.

# %% [markdown]
# ## 4 · The results table on one test set
#
# ✏️ **Exercise** · Return the probabilities of the fine-tuned model for the test photographs
# (`U.predict(model, Pte)`, shape N x 8) and its test accuracy as a Python number.

# %%
def test_result():
    # BEGIN SOLUTION
    probs = U.predict(model, Pte)
    accuracy = (probs.argmax(1) == yte).float().mean().item()
    return probs, accuracy
    # END SOLUTION

# %%
try:
    probs, acc = test_result()
except NotImplementedError:
    probs = U.predict(model, Pte)
    acc = (probs.argmax(1) == yte).float().mean().item()
assert probs.shape == (len(Xte), 8)
rows = [("small network from scratch (supplied)", scratch["test_accuracy"], U.n_params(U.small_cnn()), scratch["seconds"]),
        ("frozen features + linear layer (supplied)", frozen["test_accuracy"], 4104, frozen["seconds"]),
        ("last block fine-tuned, with augmentation", acc, trainable, seconds)]
print(f"{'approach, 800 labelled photographs':44s} {'test accuracy':>13s} {'trained weights':>16s} {'seconds':>8s}")
for name, a, w, s in rows:
    print(f"{name:44s} {a:13.3f} {w:16,d} {s:8.0f}")
table = {"mode": U.MODE, "n_test": len(Xte), "rows": [{"approach": n, "test_accuracy": a, "trained_weights": w, "seconds": s} for n, a, w, s in rows],
         "history": history}
U.save_run("d2_3_results", table)
U.save_reference("d2_3_results_" + U.MODE, table)
if not U.QUICK:
    assert acc > scratch["test_accuracy"] + 0.1, "fine-tuning should beat training from scratch on 800 photographs"
    if os.environ.get("AE202_WRITE_REF"):
        np.save(U.DATA / "runs" / "day2_finetuned_test_probs.npy", probs.numpy().astype(np.float16))

# %% [markdown]
# **Look at** all three columns, not only the first. This table, on the same test set, is
# the evidence this outcome asks for. Say what transfer learning bought over training from
# scratch, and what fine-tuning added over frozen features for its extra time. If the last
# two rows are close, that is a finding: write it down as it is.

# %% [markdown]
# ## ✱ Stretch · One more block
#
# Also set `requires_grad = True` on `m.layer3` and add it to the optimiser at `1e-4`. The
# wrapper `U.partial_backbone` runs `layer3` without gradients, so write the forward pass
# yourself. What happens to the time per epoch, and to the gap between training and
# validation loss?

# %% [markdown]
# ## Check yourself
#
# 1. Why does the frozen part stay in evaluation mode during training?
# 2. Would a 90-degree rotation be a safe augmentation here? A change of colours?
# 3. Why a smaller learning rate for the block than for the head?
#
# <details><summary>Answers</summary>
#
# 1. Its batch-normalisation layers hold statistics of a million photographs. In training
#    mode they would be overwritten by batches of 32 textile images.
# 2. Both keep the pattern class: a rotated stripe is a stripe, and the class does not depend
#    on colour. It would not be safe for a task where direction or colour is the label.
# 3. The block starts from useful weights and must be nudged; the head starts from random
#    numbers and must travel. One large rate for both wrecks the block in the first steps.
# </details>
#
# ## Carry forward
#
# The fine-tuned model's test probabilities are supplied to Lab 2.4 as
# `data/runs/day2_finetuned_test_probs.npy`. Your table is in `runs/d2_3_results.json`.
