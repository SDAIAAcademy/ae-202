# %% [markdown]
# # Lab 1.4 · Three faults in a training script
#
# Day 1 · Session 4 (When training goes wrong) · 25 min.
#
# Shaima Al-Wadaani's first script for Naqsh, `broken/train_swatches.py`, runs without an
# error and learns nothing. Three things are wrong in it: one in what the loss receives, one
# in the learning rate, one in the order of the data. You find them with a checklist that
# starts with the cheapest test there is, and you keep a log of each fault and its fix.
#
# **Tasks**
# - [ ] Run the broken script and read its log · 3 min
# - [ ] The overfit-one-batch test · 6 min
# - [ ] Fault in the loss input · 5 min
# - [ ] Fault in the learning rate · 4 min
# - [ ] Fault in the data order, and the corrected curve · 7 min
# - [ ] ✱ Stretch · Plant a fourth fault for your neighbour
#
# AI policy for this lab: an assistant may explain and hint, one step at a time. It does
# not read the broken script for you.

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

import inspect
import torch
import torch.nn.functional as F
import nawl as U

sys.path.insert(0, "broken")
import train_swatches as B

Xtr, ytr, Xva, yva, Xte, yte = U.swatches()
EPOCHS = U.pick(8, 6)
log = {"steps": []}   # your debugging log: every test, what it showed, what you changed

# %% [markdown]
# ## 1 · Run the broken script and read its log
#
# Read the three pieces first, then run them. Do not fix the file.

# %%
print(inspect.getsource(B.compute_loss))
print(inspect.getsource(B.make_batches))
print("LEARNING_RATE =", B.LEARNING_RATE, "\n")
torch.manual_seed(0)
broken = B.train(U.mlp(64), Xtr, ytr, Xva, yva, epochs=3)
log["steps"].append({"test": "run as found", "val_loss": broken["val_loss"], "val_acc": broken["val_acc"],
                     "finding": "training loss flat near ln 8, validation loss far above it, accuracy at chance"})

# %% [markdown]
# **Look at** the two losses. The training loss sits near 2.1 and the true validation loss
# is in the tens or hundreds. A flat curve has many causes; do not start guessing.

# %% [markdown]
# ## 2 · The overfit-one-batch test
#
# Take 16 swatches and train on them alone for 300 steps. A healthy model, loss and
# optimiser can memorise 16 examples: the loss goes towards zero. If it does not, the fault is
# in the model, the loss or the update, and no amount of data will help. If it does, look at
# the data pipeline.
#
# ✏️ **Exercise** · Train the model on `x, y` for `steps` steps of plain gradient descent
# (`torch.optim.SGD`) with the given `compute_loss` and `lr`.

# %%
def overfit_one_batch(compute_loss, lr, steps=300):
    torch.manual_seed(0)
    model = U.mlp(64)
    x, y = Xtr[:16], ytr[:16]
    # BEGIN SOLUTION
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = compute_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # END SOLUTION
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()   # the true loss on the 16 swatches

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _overfit(compute_loss, lr, steps=300):
    torch.manual_seed(0)
    model = U.mlp(64)
    x, y = Xtr[:16], ytr[:16]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = compute_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()

try:
    one = overfit_one_batch(B.compute_loss, B.LEARNING_RATE)
except NotImplementedError:
    overfit_one_batch = _overfit
    one = overfit_one_batch(B.compute_loss, B.LEARNING_RATE)
PASS = 0.1
print(f"one batch, script as found: loss {one:.3f}  ->  {'passes' if one < PASS else 'FAILS'} (pass: below {PASS})")
assert not one < PASS
log["steps"].append({"test": "overfit one batch, as found", "loss": one, "finding": "fails: the fault is not in the data order alone"})

# %% [markdown]
# ## 3 · Fault in the loss input
#
# Read `compute_loss` again, and the documentation of `F.cross_entropy`: it expects raw
# scores (logits) and applies the softmax itself.
#
# ✏️ **Exercise** · Write the corrected loss.

# %%
def fixed_loss(logits, y):
    # BEGIN SOLUTION
    return F.cross_entropy(logits, y)
    # END SOLUTION

# %%
try:
    fixed_loss(torch.zeros(2, 8), torch.tensor([0, 1]))
except NotImplementedError:
    fixed_loss = lambda logits, y: F.cross_entropy(logits, y)
confident = torch.full((1, 8), -20.0)
confident[0, 3] = 20.0
floor = B.compute_loss(confident, torch.tensor([3])).item()
print(f"a perfect, confident prediction: broken loss {floor:.3f}, corrected loss {fixed_loss(confident, torch.tensor([3])).item():.6f}")
one = overfit_one_batch(fixed_loss, B.LEARNING_RATE)
print(f"one batch, loss fixed, learning rate as found: loss {one:.3f}  ->  {'passes' if one < PASS else 'FAILS'}")
assert not one < PASS
log["steps"].append({"fault": "loss input", "was": "softmax applied before cross_entropy, which applies it again",
                     "fix": "pass the logits", "loss_of_a_perfect_prediction_before": floor, "one_batch_loss_after": one,
                     "finding": "still fails: another fault in the update"})

# %% [markdown]
# **Look at** the first line: with a softmax applied twice the loss of a perfect prediction
# is 1.27, not 0. The loss could never tell the network it was right.

# %% [markdown]
# ## 4 · Fault in the learning rate
#
# ✏️ **Exercise** · Run the one-batch test at 0.001, 0.01, 0.1 and 1.0 and return the rate
# with the lowest loss. Sweep powers of ten as in Lab 1.3; do not guess one value.

# %%
def choose_learning_rate():
    # BEGIN SOLUTION
    results = {lr: overfit_one_batch(fixed_loss, lr) for lr in (0.001, 0.01, 0.1, 1.0)}
    for lr, v in results.items():
        print(f"  lr {lr:<6} one-batch loss {v:.3f}")
    return min(results, key=results.get)
    # END SOLUTION

# %%
try:
    LR = choose_learning_rate()
except NotImplementedError:
    LR = 0.1
one = overfit_one_batch(fixed_loss, LR)
print(f"one batch, loss fixed, lr {LR}: loss {one:.3f}  ->  {'passes' if one < PASS else 'FAILS'}")
assert one < PASS, "choose a rate with which 16 swatches can be memorised"
log["steps"].append({"fault": "learning rate", "was": B.LEARNING_RATE, "fix": LR, "one_batch_loss_after": one,
                     "finding": "passes: model, loss and update are healthy"})

# %% [markdown]
# ## 5 · Fault in the data order, and the corrected curve
#
# The one-batch test passes, so the model, the loss and the update work. Run the full
# training with the two fixes.

# %%
torch.manual_seed(0)
two_fixes = B.train(U.mlp(64), Xtr, ytr, Xva, yva, epochs=3, compute_loss=fixed_loss, lr=LR)
assert two_fixes["val_acc"][-1] < 0.2
log["steps"].append({"test": "full training with two fixes", "val_loss": two_fixes["val_loss"],
                     "finding": "flat at ln 8: one batch can be learnt, the full data cannot, so look at the pipeline"})

# %% [markdown]
# Flat at 2.08, exactly chance: the network sees inputs that carry no information about
# their labels. Read `make_batches` once more: which rows of `X` go with which rows of `y`?
#
# ✏️ **Exercise** · Write the corrected generator: shuffled batches in which every swatch
# keeps its own label.

# %%
def fixed_batches(X, y, batch_size, generator):
    # BEGIN SOLUTION
    order = torch.randperm(len(X), generator=generator)
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size]
        yield X[idx], y[idx]
    # END SOLUTION

# %%
def _fixed_batches(X, y, batch_size, generator):
    order = torch.randperm(len(X), generator=generator)
    for i in range(0, len(X), batch_size):
        yield X[order[i:i + batch_size]], y[order[i:i + batch_size]]

try:
    next(iter(fixed_batches(Xtr[:8], ytr[:8], 4, torch.Generator().manual_seed(0))))
except NotImplementedError:
    fixed_batches = _fixed_batches
# a pipeline test worth keeping: tag every row with its own index and see that pairs survive
tag = torch.arange(40)
for xb, yb in fixed_batches(tag.float().unsqueeze(1), tag, 8, torch.Generator().manual_seed(1)):
    assert torch.equal(xb.squeeze(1).long(), yb), "a swatch lost its label"
pairs_kept = all(torch.equal(xb.squeeze(1).long(), yb)
                 for xb, yb in B.make_batches(tag.float().unsqueeze(1), tag, 8, torch.Generator().manual_seed(1)))
print("pairs kept by the script as found:", pairs_kept, "| by your generator: True")

torch.manual_seed(0)
fixed = B.train(U.mlp(64), Xtr, ytr, Xva, yva, epochs=EPOCHS, compute_loss=fixed_loss, lr=LR, make_batches=fixed_batches)
assert fixed["val_acc"][-1] > 0.4 and fixed["val_loss"][-1] < 1.8, "the corrected script must learn"
log["steps"].append({"fault": "data order", "was": "inputs shuffled, labels left in the stored order",
                     "fix": "index inputs and labels with the same shuffled indices",
                     "val_loss": fixed["val_loss"], "val_acc": fixed["val_acc"], "finding": "the loss falls"})
print("log written to", U.save_run("d1_4_debug_log", log).name)
U.save_reference("d1_4_log_" + U.MODE, log)
fig = U.plot_curves({"as found": {k: [min(v, 4.0) for v in vals] for k, vals in broken.items()}, "two fixes": two_fixes, "three fixes": fixed},
                    "The same script, before and after (losses above 4 drawn at 4)")

# %% [markdown]
# **Look at** your log: it is the evidence this outcome asks for. It shows the
# overfit-one-batch test, each fault, what was changed and the corrected curve, in the order
# you found them.

# %% [markdown]
# ## ✱ Stretch · A fourth fault
#
# Copy `broken/train_swatches.py`, plant one new fault (some ideas: gradients never cleared,
# validation on the training set, labels shifted by one class) and hand it to your
# neighbour. Which test of the checklist catches it?

# %% [markdown]
# ## Check yourself
#
# 1. The one-batch test passed and full training stayed at chance. What does that
#    combination tell you?
# 2. Why did fixing the loss alone not make the one-batch test pass?
# 3. Why start with one batch and not with a full run?
#
# <details><summary>Answers</summary>
#
# 1. Model, loss and optimiser work; the fault is in what reaches them: the data pipeline
#    (order, labels, preprocessing).
# 2. Two faults were stacked: with a learning rate of 30 every step overshoots. A test that
#    still fails after a correct fix means there is another fault, not that the fix was wrong.
# 3. It takes seconds, it has a known right answer (a loss near zero) and it cuts the search
#    in two.
# </details>
#
# ## Carry forward
#
# The checklist (run as found, overfit one batch, check the loss on a known case, sweep the
# rate, test the pipeline with tagged rows) is the one you use on Day 2 and in the capstone.
