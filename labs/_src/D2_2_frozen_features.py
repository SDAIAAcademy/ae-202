# %% [markdown]
# # Lab 2.2 · Frozen features
#
# Day 2 · Session 2 (Pretrained backbones) · 25 min.
#
# Shaima Al-Wadaani has labelled 800 photographs, 100 per pattern. That is all the labelled
# data the cooperative has. You take an image model trained on a million everyday
# photographs, remove its last layer, use the rest as a fixed feature extractor and train one
# linear layer on its features. Then you compare with training from scratch on the same 800.
#
# **Tasks**
# - [ ] Load the backbone and remove its head · 4 min
# - [ ] Resize and normalise as the backbone expects · 4 min
# - [ ] Extract features without gradients · 7 min
# - [ ] A linear classifier on the features · 6 min
# - [ ] Against training from scratch on the same 800 · 4 min
# - [ ] ✱ Stretch · Take the features one block earlier and compare
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# The backbone's weights (about 45 MB) are downloaded on first use.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import nawl as U

N = U.pick(None, 240)
Xsm, ysm = U.images("small", N)       # the 800 labelled photographs (quick mode: 240)
Xva, yva = U.images("val", U.pick(None, 300))
Xte, yte = U.images("test", U.pick(None, 300))
print(f"mode {U.MODE}: {len(Xsm)} labelled photographs, {len(Xva)} for validation, {len(Xte)} for the test")

# %% [markdown]
# ## 1 · Load the backbone and remove its head
#
# `U.backbone()` returns a ResNet-18 with the weights torchvision publishes, trained on
# ImageNet-1k. Its last layer, `fc`, maps 512 features to 1,000 everyday classes.
#
# ✏️ **Exercise** · Return the backbone with `fc` replaced by `nn.Identity()`, in evaluation
# mode. Its output is then the 512 features.

# %%
def feature_extractor():
    m = U.backbone()
    # BEGIN SOLUTION
    m.fc = nn.Identity()
    m.eval()
    return m
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    extractor = feature_extractor()
except NotImplementedError:
    extractor = U.backbone()
    extractor.fc = nn.Identity()
    extractor.eval()
assert not extractor.training, "put the backbone in evaluation mode"
print(f"{U.n_params(extractor) / 1e6:.1f} million weights, none of which you will train")

# %% [markdown]
# ## 2 · Resize and normalise as the backbone expects
#
# A pretrained network expects its inputs prepared the way they were during its training:
# here, each colour channel minus `U.IMAGENET_MEAN` and divided by `U.IMAGENET_STD`. Our
# photographs are 48 pixels wide; the backbone was trained on 224. We resize to 64: small
# enough for a laptop, large enough for its filters.
#
# ✏️ **Exercise** · Resize `X` (values in [0, 1]) to `size` x `size` with
# `F.interpolate(..., mode="bilinear", align_corners=False)` and normalise each channel.

# %%
def prepare(X, size=64):
    # BEGIN SOLUTION
    mean = torch.tensor(U.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(U.IMAGENET_STD).view(1, 3, 1, 1)
    X = F.interpolate(X, size=size, mode="bilinear", align_corners=False)
    return (X - mean) / std
    # END SOLUTION

# %%
try:
    P = prepare(Xsm[:4])
except NotImplementedError:
    prepare = U.to_backbone_input
    P = prepare(Xsm[:4])
assert P.shape == (4, 3, 64, 64), P.shape
assert torch.allclose(P, U.to_backbone_input(Xsm[:4]), atol=1e-5), "differs from the reference preparation"
print("prepared batch:", tuple(P.shape), "| channel means", [round(v, 2) for v in P.mean(dim=(0, 2, 3)).tolist()])

# %% [markdown]
# ## 3 · Extract features without gradients
#
# ✏️ **Exercise** · Run `prepare(X)` through `extractor` in batches of `bs` inside
# `torch.no_grad()` and return one tensor of shape `(N, 512)`.

# %%
def extract(X, bs=100):
    # BEGIN SOLUTION
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(extractor(prepare(X[i:i + bs])))
    return torch.cat(out)
    # END SOLUTION

# %%
def _extract(X, bs=100):
    with torch.no_grad():
        return torch.cat([extractor(U.to_backbone_input(X[i:i + bs])) for i in range(0, len(X), bs)])

t = time.time()
try:
    Fsm = extract(Xsm)
except NotImplementedError:
    extract = _extract
    Fsm = extract(Xsm)
Fva, Fte = extract(Xva), extract(Xte)
seconds = time.time() - t
assert Fsm.shape == (len(Xsm), 512) and not Fsm.requires_grad
print(f"features: {tuple(Fsm.shape)}, extracted for {len(Xsm) + len(Xva) + len(Xte)} photographs in {seconds:.0f} s"
      f" ({1000 * seconds / (len(Xsm) + len(Xva) + len(Xte)):.1f} ms per photograph)")

# %% [markdown]
# ## 4 · A linear classifier on the features
#
# The features are fixed numbers now. Standardise them with the mean and standard deviation
# of the **training** features, then train one linear layer: 512 inputs, 8 classes.
#
# ✏️ **Exercise** · Return `(model, history)`: an `nn.Linear(512, 8)` trained with
# `U.fit(..., epochs=30, lr=1e-2, bs=64, weight_decay=1e-3, verbose=False)` on the
# standardised training features, validated on the standardised validation features.

# %%
mu, sd = Fsm.mean(0), Fsm.std(0) + 1e-6
z = lambda Fx: (Fx - mu) / sd


def train_linear():
    torch.manual_seed(0)
    # BEGIN SOLUTION
    model = nn.Linear(512, 8)
    history = U.fit(model, z(Fsm), ysm, z(Fva), yva, epochs=30, lr=1e-2, bs=64, weight_decay=1e-3, verbose=False)
    return model, history
    # END SOLUTION

# %%
try:
    linear, h_lin = train_linear()
except NotImplementedError:
    torch.manual_seed(0)
    linear = nn.Linear(512, 8)
    h_lin = U.fit(linear, z(Fsm), ysm, z(Fva), yva, epochs=30, lr=1e-2, bs=64, weight_decay=1e-3, verbose=False)
test_loss, test_acc = U.evaluate(linear, z(Fte), yte)
print(f"linear classifier on frozen features: {U.n_params(linear):,} trained weights,"
      f" validation accuracy {h_lin['val_acc'][-1]:.3f}, test accuracy {test_acc:.3f}")
assert test_acc > 0.6, "the linear classifier should be far above chance"

# %% [markdown]
# ## 5 · Against training from scratch on the same 800
#
# `data/runs/day2_scratch.json` is the supplied result of the small convolutional network of
# Lab 2.1 trained from scratch on these 800 photographs only (20 epochs).

# %%
scratch = json.loads((U.DATA / "runs" / "day2_scratch.json").read_text(encoding="utf-8"))
frozen = {"what": "linear classifier on frozen ResNet-18 features, 64-pixel input, 800 labelled photographs",
          "test_accuracy": test_acc, "val_accuracy": h_lin["val_acc"][-1], "train_loss": h_lin["train_loss"][-1],
          "val_loss": h_lin["val_loss"][-1], "seconds": round(seconds, 1), "trained_weights": U.n_params(linear)}
print(f"{'approach':38s} {'trained weights':>16s} {'test accuracy':>14s}")
print(f"{'small network from scratch (supplied)':38s} {U.n_params(U.small_cnn()):16,d} {scratch['test_accuracy']:14.3f}")
print(f"{'frozen features + linear layer':38s} {U.n_params(linear):16,d} {test_acc:14.3f}")
if not U.QUICK:
    assert test_acc > scratch["test_accuracy"] + 0.1
    if os.environ.get("AE202_WRITE_REF"):
        (U.DATA / "runs" / "day2_frozen.json").write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
U.save_run("d2_2_frozen", frozen)

# %% [markdown]
# **Look at** the two rows. Both saw the same 800 labelled photographs. The second one
# started with filters that a million other photographs had already paid for, and trained
# 4,104 numbers on top.

# %% [markdown]
# ## ✱ Stretch · One block earlier
#
# Set `extractor.layer4 = nn.Identity()` (the features are then 256 numbers per photograph
# after the average pool) and repeat tasks 3 and 4 with `nn.Linear(256, 8)`. Earlier
# features are more general and less abstract: which helps more for woven patterns?

# %% [markdown]
# ## Check yourself
#
# 1. Why is the backbone in evaluation mode while you extract features?
# 2. You standardised with the mean of the training features. Why not the mean of each set?
# 3. ImageNet has no class "chevron textile". Why do its features help at all?
#
# <details><summary>Answers</summary>
#
# 1. Its batch-normalisation layers would otherwise use the statistics of each batch and
#    update their running averages: the features would depend on what else is in the batch.
# 2. The test set must be treated exactly as new data will be: with numbers fixed during
#    training. Statistics of the test set are information you will not have in use.
# 3. Its early and middle layers detect edges, textures and repeated shapes, which every
#    image task needs. Only the last layer was about its 1,000 classes, and you removed it.
# </details>
#
# ## Carry forward
#
# The frozen-feature result is supplied to Lab 2.3 as `data/runs/day2_frozen.json`. Yours is
# in `runs/d2_2_frozen.json`.
