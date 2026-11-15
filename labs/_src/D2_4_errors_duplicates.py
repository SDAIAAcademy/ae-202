# %% [markdown]
# # Lab 2.4 · Errors and near-duplicates
#
# Day 2 · Session 4 (Evaluating a deep model) · 25 min.
#
# Mariam Al-Hudaithi asks one question before Naqsh goes on the shop's upload page: "how
# good is it, really?" You take the fine-tuned model's predictions on the 1,000 test
# photographs and look past the single number: which classes it confuses, what its most
# confident errors are, and whether the test set is as independent as it should be.
#
# **Tasks**
# - [ ] The confusion matrix · 6 min
# - [ ] The ten most confident errors · 7 min
# - [ ] Near-duplicates between training and test · 8 min
# - [ ] Accuracy with and without them · 4 min
# - [ ] ✱ Stretch · Precision and recall per class
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# This lab starts from supplied predictions: it does not need your Lab 2.3.

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

Xte, yte = U.images("test")
Xsm, ysm = U.images("small")
probs = torch.from_numpy(np.load(U.DATA / "runs" / "day2_finetuned_test_probs.npy").astype(np.float32))
pred = probs.argmax(1)
accuracy = (pred == yte).float().mean().item()
print(f"fine-tuned model: {len(yte)} test photographs, accuracy {accuracy:.3f}")

# %% [markdown]
# ## 1 · The confusion matrix
#
# ✏️ **Exercise** · Return an 8 x 8 array of counts: row = the true class (the label),
# column = the predicted class.

# %%
def confusion_matrix(y_true, y_pred, n=8):
    # BEGIN SOLUTION
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        m[t, p] += 1
    return m
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    cm = confusion_matrix(yte, pred)
except NotImplementedError:
    cm = U.confusion(yte, pred)
assert cm.shape == (8, 8) and cm.sum() == len(yte) and np.trace(cm) == int((pred == yte).sum())
U.print_confusion(cm)
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = sorted(((int(off[i, j]), U.CLASSES[i], U.CLASSES[j]) for i in range(8) for j in range(8) if off[i, j]), reverse=True)[:5]
print("\nlargest confusions (count, labelled as, predicted as):", pairs)
recall = cm.diagonal() / cm.sum(axis=1)
print("accuracy within each labelled class:", {c: round(float(r), 2) for c, r in zip(U.CLASSES, recall)})

# %% [markdown]
# **Look at** which pairs carry the errors. A model that is 80% right is not 80% right on
# every class: one or two pairs usually hold most of what is wrong.

# %% [markdown]
# ## 2 · The ten most confident errors
#
# An error with a confidence of 0.55 is a hard photograph. An error with a confidence of
# 0.99 is more interesting: either the model has learnt something wrong, or the label is.
#
# ✏️ **Exercise** · Return the indices (into the test set) of the `k` wrong predictions with
# the highest predicted probability, most confident first.

# %%
def most_confident_errors(probs, y, k=10):
    # BEGIN SOLUTION
    conf, pred = probs.max(1)
    wrong = (pred != y).nonzero().squeeze(1)
    order = conf[wrong].argsort(descending=True)
    return wrong[order][:k].tolist()
    # END SOLUTION

# %%
def _mce(probs, y, k=10):
    conf, pred = probs.max(1)
    wrong = (pred != y).nonzero().squeeze(1)
    return wrong[conf[wrong].argsort(descending=True)][:k].tolist()

try:
    worst = most_confident_errors(probs, yte)
except NotImplementedError:
    worst = _mce(probs, yte)
assert worst == _mce(probs, yte), "not the ten most confident wrong predictions"
fig = U.show_images(Xte[worst], [f"label {U.CLASSES[yte[i]]}\nsays {U.CLASSES[pred[i]]} {probs[i].max():.2f}" for i in worst], cols=5, size=2.0)
for i in worst:
    print(f"test image {i:4d}: labelled {U.CLASSES[yte[i]]:9s} predicted {U.CLASSES[pred[i]]:9s} confidence {probs[i].max():.3f}")

# %% [markdown]
# Judge each of the ten by eye before you run the next cell: **is the model wrong, or is the
# label wrong?** Write your count down.

# %%
# The data generator's record of what it planted (in a real project nobody hands you this:
# you would send the ten photographs back to Shaima Al-Wadaani, who owns the labels)
import json
planted = json.loads((U.CACHE / "planted_key.json").read_text(encoding="utf-8"))
test_index = U.textiles()["test"]
mislabelled = {int(np.searchsorted(test_index, t)) for t, _, _ in planted["mislabelled"]}
label_faults = [i for i in worst if i in mislabelled]
print(f"of the ten most confident errors, {len(label_faults)} have a wrong label: test images {label_faults}")
print(f"the test set holds {len(mislabelled)} wrong labels in all; the model 'disagrees' with"
      f" {sum(int(pred[i] != yte[i]) for i in mislabelled)} of them")
U.save_reference("d2_4_errors", {"accuracy": accuracy, "confusions": pairs, "label_faults_in_top10": len(label_faults),
                                 "mislabelled_total": len(mislabelled),
                                 "model_disagrees_with": sum(int(pred[i] != yte[i]) for i in mislabelled)})

# %% [markdown]
# **Look at** what this means for the accuracy you report: some of the model's "errors" are
# the model being right against a wrong label. Looking at confident errors is the cheapest
# label audit there is.

# %% [markdown]
# ## 3 · Near-duplicates between training and test
#
# Makers sometimes upload the same photograph twice. If one copy is among the 800 training
# images and the other in the test set, the test asks the model about something it has
# seen. `U.thumbs` turns each photograph into a 24 x 24 colour fingerprint of length 1, so
# the dot product of two fingerprints is their cosine similarity.
#
# ✏️ **Exercise** · For every test photograph, return the highest similarity to any training
# photograph and the index of that training photograph: two tensors of length `len(test)`.

# %%
def nearest_training_image(test_thumbs, train_thumbs):
    # BEGIN SOLUTION
    sims = test_thumbs @ train_thumbs.T
    best, idx = sims.max(dim=1)
    return best, idx
    # END SOLUTION

# %%
Tte, Tsm = U.thumbs(Xte), U.thumbs(Xsm)
try:
    best, idx = nearest_training_image(Tte, Tsm)
except NotImplementedError:
    best, idx = (Tte @ Tsm.T).max(dim=1)
assert best.shape == (len(Xte),) and float(best.max()) <= 1.0001
order = best.argsort(descending=True)
print("similarity to the nearest training photograph, sorted (every 10th of the top 80):")
print("  " + " ".join(f"{best[i]:.3f}" for i in order[:80:10]))
THRESHOLD = 0.985
dups = (best > THRESHOLD).nonzero().squeeze(1).tolist()
print(f"{len(dups)} test photographs have a training photograph more similar than {THRESHOLD}")
show = order[:6].tolist()
fig = U.show_images(torch.cat([Xte[show], Xsm[idx[show]]]), [f"test {i}" for i in show] + [f"train, sim {best[i]:.2f}" for i in show], cols=6, size=1.8)

# %% [markdown]
# **Look at** the sorted similarities: they do not fade smoothly. There is a group above
# 0.99 and then a drop. A threshold belongs in that drop, and you check it by looking at pairs
# on both sides of it.

# %% [markdown]
# ## 4 · Accuracy with and without them
#
# ✏️ **Exercise** · Return three accuracies: on all test photographs, on the near-duplicates
# only, and on the test set without them.

# %%
def three_accuracies(pred, y, dups):
    # BEGIN SOLUTION
    correct = (pred == y).float()
    is_dup = torch.zeros(len(y), dtype=torch.bool)
    is_dup[dups] = True
    return correct.mean().item(), correct[is_dup].mean().item(), correct[~is_dup].mean().item()
    # END SOLUTION

# %%
try:
    a_all, a_dup, a_clean = three_accuracies(pred, yte, dups)
except NotImplementedError:
    correct = (pred == yte).float()
    mask = torch.zeros(len(yte), dtype=torch.bool)
    mask[dups] = True
    a_all, a_dup, a_clean = correct.mean().item(), correct[mask].mean().item(), correct[~mask].mean().item()
print(f"all {len(yte)} test photographs: {a_all:.3f}")
print(f"the {len(dups)} near-duplicates:   {a_dup:.3f}")
print(f"the other {len(yte) - len(dups)}:            {a_clean:.3f}   <- the number to report")
report = {"accuracy_all": a_all, "near_duplicates": len(dups), "accuracy_on_duplicates": a_dup, "accuracy_without": a_clean,
          "threshold": THRESHOLD}
U.save_run("d2_4_report", report)
U.save_reference("d2_4_report", report)

# %% [markdown]
# **Look at** the size of the correction. Forty photographs in a thousand move the headline
# number only a little here. In a set where a quarter of the test images have a twin in
# training, the same check changes the decision. You do not know which case you are in
# until you look.

# %% [markdown]
# ## ✱ Stretch · Precision and recall per class
#
# From the confusion matrix: recall of a class is its diagonal cell divided by its row sum,
# precision the diagonal cell divided by its column sum. Which class has high recall and low
# precision, and what does that mean for the shop's upload page?

# %% [markdown]
# ## Check yourself
#
# 1. Two classes are confused in one direction only (many A predicted as B, few B as A).
#    What might that tell you?
# 2. Why look at the most confident errors and not at ten random errors?
# 3. The near-duplicates were found in pixel space with small thumbnails. When would that
#    miss a duplicate?
#
# <details><summary>Answers</summary>
#
# 1. The model has a default between the two: features of B are a superset of what it finds
#    in A, or B has more or cleaner training examples.
# 2. Confident errors separate into few causes (wrong labels, a systematic confusion);
#    random errors are mostly hard cases near a boundary, which teach less per photograph.
# 3. When the same cloth is photographed again, cropped, rotated or lit differently: the
#    pixels differ while the cloth is the same. Then compare in the feature space of a model.
# </details>
#
# ## Carry forward
#
# The capstone rubric asks for the test result and for its limits. A confusion matrix, the
# confident errors you inspected and a duplicate check are what stands behind a number.
