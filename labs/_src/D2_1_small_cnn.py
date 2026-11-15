# %% [markdown]
# # Lab 2.1 · A small convolutional network
#
# Day 2 · Session 1 (Convolutions and what they detect) · 25 min.
#
# The catalogue photographs are not laid straight: every cloth is turned and shifted. You see
# what that does to yesterday's network, compute one convolution by hand, build a small
# convolutional network for Naqsh, continue its training from a supplied checkpoint and look
# at what its first layer learnt.
#
# **Tasks**
# - [ ] A convolution by hand · 5 min
# - [ ] Build the network and count its weights · 6 min
# - [ ] Train from the supplied checkpoint · 8 min
# - [ ] First-layer filters and feature maps · 6 min
# - [ ] ✱ Stretch · Shift every test image by four pixels and compare the two networks
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import nawl as U

Xtr, ytr = U.images("train", U.pick(None, 1500))
Xva, yva = U.images("val", U.pick(None, 500))
Xtr, Xva = Xtr - 0.5, Xva - 0.5          # pixel values in [-0.5, 0.5]
print(f"mode {U.MODE}: {len(Xtr)} training photographs of shape {tuple(Xtr.shape[1:])}, {len(Xva)} for validation")
fig = U.show_images(Xtr[:16] + 0.5, [U.CLASSES[i] for i in ytr[:16]], cols=8)

# %% [markdown]
# First, yesterday's network on today's data: a two-layer network on the flattened pixels.

# %%
torch.manual_seed(0)
flat = lambda X: X.reshape(len(X), -1)
mlp = U.mlp(128, n_in=3 * 48 * 48)
h_mlp = U.fit(mlp, flat(Xtr), ytr, flat(Xva), yva, epochs=U.pick(5, 2), lr=1e-3, bs=64, verbose=False)
print(f"two-layer network on photographs: {U.n_params(mlp):,} weights, validation accuracy {h_mlp['val_acc'][-1]:.3f} (chance 0.125)")

# %% [markdown]
# **Look at** that number next to yesterday's 0.79 on swatches. A fully connected layer has
# one weight per pixel position: a stripe moved by three pixels is, to it, a new input.

# %% [markdown]
# ## 1 · A convolution by hand
#
# A convolution slides one small grid of weights (a kernel) over the image and writes, at
# each position, the sum of the products of kernel and pixels under it.
#
# ✏️ **Exercise** · Return the "valid" convolution of a grey image `img` (H x W) with a kernel
# `k` (kh x kw): the output has shape `(H - kh + 1, W - kw + 1)`. Two loops are fine.

# %%
def convolve(img, k):
    # BEGIN SOLUTION
    kh, kw = k.shape
    out = torch.zeros(img.shape[0] - kh + 1, img.shape[1] - kw + 1)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i:i + kh, j:j + kw] * k).sum()
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
grey = Xtr[0].mean(0)
edge = torch.tensor([[-1.0, 0.0, 1.0]] * 3)        # responds where brightness rises from left to right
library = F.conv2d(grey[None, None], edge[None, None])[0, 0]
try:
    mine = convolve(grey, edge)
except NotImplementedError:
    mine = library
assert mine.shape == (46, 46), mine.shape
assert torch.allclose(mine, library, atol=1e-5), "differs from the library's convolution"
print("output shape", tuple(mine.shape), "| largest difference from F.conv2d:", f"{(mine - library).abs().max().item():.1e}")
fig = U.show_images([grey[1:-1, 1:-1].numpy(), mine.abs().numpy()], ["the photograph, grey", "response to a vertical edge"], cols=2, size=2.6)

# %% [markdown]
# ## 2 · Build the network and count its weights
#
# Three convolution blocks, then an average over all positions, then one linear layer.
#
# ✏️ **Exercise** · Build it with `nn.Sequential`:
# `Conv2d(3, 16, 5, padding=2)`, `BatchNorm2d(16)`, `ReLU`, `MaxPool2d(2)`;
# `Conv2d(16, 32, 3, padding=1)`, `BatchNorm2d(32)`, `ReLU`, `MaxPool2d(2)`;
# `Conv2d(32, 64, 3, padding=1)`, `BatchNorm2d(64)`, `ReLU`;
# `AdaptiveAvgPool2d(1)`, `Flatten`, `Linear(64, 8)`.

# %%
def build_cnn():
    # BEGIN SOLUTION
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 8))
    # END SOLUTION

# %%
try:
    cnn = build_cnn()
except NotImplementedError:
    cnn = U.small_cnn()
assert cnn(Xtr[:2]).shape == (2, 8)
assert U.n_params(cnn) == U.n_params(U.small_cnn()), "the weight count differs from the specification"
first = cnn[0]
print(f"convolutional network: {U.n_params(cnn):,} weights; the two-layer network: {U.n_params(mlp):,}")
print(f"first layer: {first.weight.numel() + first.bias.numel():,} weights for {tuple(first.weight.shape)}"
      f" against {3 * 48 * 48 * 128 + 128:,} in the first layer of the two-layer network")

# %% [markdown]
# **Look at** the two counts. The same 16 kernels are used at every position of the image:
# that sharing is why the network is small, and why a stripe is a stripe wherever it lies.

# %% [markdown]
# ## 3 · Train from the supplied checkpoint
#
# `data/checkpoints/cnn_warm.pt` holds this network after 8 epochs on the 6,000 training
# photographs (about two minutes on a laptop). You continue from it.
#
# ✏️ **Exercise** · Load the checkpoint into `cnn` (`torch.load`, then `load_state_dict`),
# measure it on the validation set with `U.evaluate`, train `EPOCHS` more epochs with
# `U.fit(..., lr=5e-4, bs=64)` and return `(accuracy before, history)`.

# %%
EPOCHS = U.pick(2, 1)


def continue_training(cnn):
    # BEGIN SOLUTION
    cnn.load_state_dict(torch.load(U.DATA / "checkpoints" / "cnn_warm.pt"))
    _, before = U.evaluate(cnn, Xva, yva)
    torch.manual_seed(0)
    history = U.fit(cnn, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=5e-4, bs=64)
    return before, history
    # END SOLUTION

# %%
try:
    before, h_cnn = continue_training(cnn)
except NotImplementedError:
    cnn = U.small_cnn()
    cnn.load_state_dict(torch.load(U.DATA / "checkpoints" / "cnn_warm.pt"))
    before = U.evaluate(cnn, Xva, yva)[1]
    torch.manual_seed(0)
    h_cnn = U.fit(cnn, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=5e-4, bs=64)
after = h_cnn["val_acc"][-1]
print(f"validation accuracy: {before:.3f} from the checkpoint, {after:.3f} after {EPOCHS} more epoch(s);"
      f" the two-layer network reached {h_mlp['val_acc'][-1]:.3f}")
assert before > 0.5, "the checkpoint was not loaded"
assert after > h_mlp["val_acc"][-1] + 0.2
U.save_reference("d2_1_" + U.MODE, {"mlp_val_acc": h_mlp["val_acc"][-1], "mlp_params": U.n_params(mlp), "cnn_params": U.n_params(cnn),
                                    "cnn_before": before, "cnn_after": after, "history": h_cnn})

# %% [markdown]
# ## 4 · First-layer filters and feature maps
#
# The first layer holds 16 kernels of 5 x 5 pixels in three colours: small enough to look at.
#
# ✏️ **Exercise** · Return the first layer's weights as 16 small images: shape
# `(16, 3, 5, 5)`, each one rescaled so that its smallest value is 0 and its largest is 1.

# %%
def filters_as_images(cnn):
    # BEGIN SOLUTION
    w = cnn[0].weight.detach().clone()
    lo = w.amin(dim=(1, 2, 3), keepdim=True)
    hi = w.amax(dim=(1, 2, 3), keepdim=True)
    return (w - lo) / (hi - lo)
    # END SOLUTION

# %%
try:
    filt = filters_as_images(cnn)
except NotImplementedError:
    w = cnn[0].weight.detach().clone()
    filt = (w - w.amin(dim=(1, 2, 3), keepdim=True)) / (w.amax(dim=(1, 2, 3), keepdim=True) - w.amin(dim=(1, 2, 3), keepdim=True))
assert filt.shape == (16, 3, 5, 5) and float(filt.min()) == 0.0 and float(filt.max()) == 1.0
fig = U.show_images(filt, [f"filter {i}" for i in range(16)], cols=8)

x = Xva[:1]
with torch.no_grad():
    maps = torch.relu(cnn[1](cnn[0](x)))[0]           # 16 feature maps of 48 x 48
strongest = maps.mean(dim=(1, 2)).argsort(descending=True)[:5]
fig = U.show_images([(x[0] + 0.5).mean(0).numpy()] + [maps[i].numpy() for i in strongest],
                    [U.CLASSES[yva[0]]] + [f"map of filter {int(i)}" for i in strongest], cols=6, size=2.0)

# %% [markdown]
# **Look at** the filters: several are small edge or bar detectors at different angles, some
# compare colours. Nobody designed them; they are what lowered the loss. The feature maps
# show where in the photograph each one fires.

# %% [markdown]
# ## ✱ Stretch · Shift the test images
#
# Roll every validation photograph by four pixels (`torch.roll(Xva, 4, dims=3)`) and measure
# both networks again. Which one loses more? Why is the convolutional network not perfectly
# unchanged either (think of the pooling grid and of the borders)?

# %% [markdown]
# ## Check yourself
#
# 1. The first convolution has 1,216 weights and the first fully connected layer 884,864.
#    Where does the difference come from?
# 2. What does `MaxPool2d(2)` do to a feature map of 48 x 48, and why is that useful?
# 3. The filters were not designed. What made them edge detectors?
#
# <details><summary>Answers</summary>
#
# 1. The convolution has 16 kernels of 5 x 5 x 3 weights (plus 16 biases) used at every
#    position; the fully connected layer has one weight for each of 6,912 inputs and each of
#    128 units.
# 2. It keeps the largest value of every 2 x 2 block: 24 x 24. The next layer then sees a
#    wider part of the image, and small shifts change the result less.
# 3. Gradient descent: edges and bars at different angles are what tells these eight
#    patterns apart, so weights that respond to them lowered the loss.
# </details>
#
# ## Carry forward
#
# The network is `U.small_cnn()` from here on. Its accuracy from scratch on only 800
# photographs is the baseline of the next two labs.
