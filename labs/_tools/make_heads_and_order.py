"""Numbers for three slide segments: batches stored by class, the head after the last convolution,
and one label against several labels per photograph.

    uv run python _tools/make_heads_and_order.py      (from labs/)

Writes data/reference/heads_and_order.json. Every number on the slides `batches-by-class`,
`batches-by-class-fix`, `head-average-or-flatten`, `several-labels` and `sigmoid-per-label`
comes from that file.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import torch
import torch.nn as nn
import torch.nn.functional as F

import nawl as U


# 1 · Batches stored by class --------------------------------------------------------------------

def run_order(mode: str, epochs: int = 3, bs: int = 32, lr: float = 1e-3):
    """The Day 1 network on the swatches, stored class by class. mode "stored": every batch holds
    one class, in the stored order; "batch_order": the same one-class batches, taken in a new
    random order each epoch; "examples": the swatches themselves in a new random order each epoch;
    "pairs_broken": for comparison, the swatches in their usual mixed order with the inputs shuffled
    each epoch and the labels left in place (inputs and labels no longer match)."""
    Xtr, ytr, Xva, yva, _, _ = U.swatches()
    if mode != "pairs_broken":
        order = torch.argsort(ytr, stable=True)      # stored class by class: 375 stripes, then 375 chevrons, ...
        Xtr, ytr = Xtr[order], ytr[order]
    torch.manual_seed(0)
    model = U.mlp(64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    log = {"batch_loss": [], "batch_class_mix": [], "epochs": []}
    for epoch in range(epochs):
        model.train()
        if mode == "examples":
            batches = list(torch.randperm(len(Xtr), generator=g).split(bs))
        elif mode == "pairs_broken":
            batches = list(torch.arange(len(Xtr)).split(bs))
            shuffled = torch.randperm(len(Xtr), generator=g)
        else:
            batches = list(torch.arange(len(Xtr)).split(bs))
            if mode == "batch_order":
                batches = [batches[j] for j in torch.randperm(len(batches), generator=g).tolist()]
        if epoch == 0:
            pure = [ytr[b].unique().tolist() for b in batches]
        for b in batches:
            xb = Xtr[shuffled[b]] if mode == "pairs_broken" else Xtr[b]
            loss = F.cross_entropy(model(xb), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch == 0:
                log["batch_loss"].append(round(loss.item(), 3))
                log["batch_class_mix"].append(len(set(ytr[b].tolist())))
        vl, va = U.evaluate(model, Xva, yva)
        pred = U.predict(model, Xva).argmax(1)
        counts = torch.bincount(pred, minlength=8).tolist()
        top = int(torch.tensor(counts).argmax())
        log["epochs"].append({"epoch": epoch + 1, "val_loss": round(vl, 3), "val_acc": round(va, 3),
                              "most_predicted": U.CLASSES[top], "share_most_predicted": round(counts[top] / len(pred), 3)})
    # for each class, its one-class batches of epoch 1 in the order they were taken: the loss of
    # the first and of the last of them (when batches hold one class)
    if mode in ("stored", "batch_order"):
        runs = []
        for c in range(8):
            mine = [j for j, u in enumerate(pure) if u == [c]]
            runs.append({"class": U.CLASSES[c], "first_batch": mine[0] + 1, "first_loss": log["batch_loss"][mine[0]],
                         "last_batch": mine[-1] + 1, "last_loss": log["batch_loss"][mine[-1]]})
        log["class_runs"] = runs
    later = log["batch_loss"][10:]
    log["batch_loss_range_after_10"] = [min(later), max(later)]
    log["batches_per_epoch"] = len(log["batch_loss"])
    log["one_class_batches"] = sum(1 for m in log["batch_class_mix"] if m == 1)
    log["mean_classes_per_batch"] = round(sum(log["batch_class_mix"]) / len(log["batch_class_mix"]), 2)
    del log["batch_class_mix"]
    return log


# 2 · The head after the last convolution ------------------------------------------------------------

def head_counts(channels: int, side: int, classes: int):
    """Weights of the linear layer that follows, with biases: after flattening the whole map, and
    after an average over all positions."""
    return {"map": f"{channels} x {side} x {side}", "flatten_inputs": channels * side * side,
            "flatten": channels * side * side * classes + classes, "average": channels * classes + classes}


def swatch_cnn(head: str):
    """A parallel network for the grey 24 x 24 swatches of Day 1: two pooled blocks, 32 channels."""
    body = [nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
    if head == "average":
        return nn.Sequential(*body, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 8))
    return nn.Sequential(*body, nn.Flatten(), nn.Linear(32 * 6 * 6, 8))


def heads():
    out = {}
    for head in ("average", "flatten"):
        m = swatch_cnn(head)
        last = m[-1]
        body = U.n_params(m) - U.n_params(last)
        try:
            m(torch.zeros(1, 1, 32, 32))
            other_size = "runs"
        except RuntimeError:
            other_size = "fails: the layer expects a fixed number of inputs"
        out[head] = {"head": U.n_params(last), "body": body, "total": U.n_params(m),
                     "head_share": round(U.n_params(last) / U.n_params(m), 3), "on_32x32": other_size}
    out["swatch_network_formula"] = head_counts(32, 6, 8)
    out["ratio_flatten_to_average"] = round(out["flatten"]["head"] / out["average"]["head"], 1)
    # a second parallel, for a practice question: 128 channels at 4 x 4, ten product categories
    out["question"] = head_counts(128, 4, 10)
    return out


# 3 · One label or several -----------------------------------------------------------------------

def labels():
    """Eight scores for one rug with stripes and stars, set by hand (not the output of a trained
    model), read by a softmax head and by eight independent sigmoids."""
    names = U.CLASSES
    z = torch.tensor([2.4, -1.5, -2.0, -0.8, -2.6, -1.9, -0.4, 2.0])
    target = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 1.0])     # stripe and star are both on the rug
    p_soft = torch.softmax(z, 0)
    p_sig = torch.sigmoid(z)
    rows = [{"class": n, "score": round(float(z[i]), 1), "softmax": round(float(p_soft[i]), 2),
             "sigmoid": round(float(p_sig[i]), 2)} for i, n in enumerate(names)]
    bce = F.binary_cross_entropy_with_logits(z, target).item()
    # the best a softmax head can do when two classes are equally present: 0.5 each, cross-entropy ln 2
    soft_floor = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    return {"rows": rows, "softmax_sum": round(float(p_soft.sum()), 2), "sigmoid_sum": round(float(p_sig.sum()), 2),
            "above_half_softmax": [n for n, p in zip(names, p_soft) if p > 0.5],
            "above_half_sigmoid": [n for n, p in zip(names, p_sig) if p > 0.5],
            "bce_with_logits": round(bce, 3), "softmax_best_two_labels": round(soft_floor, 3)}


if __name__ == "__main__":
    result = {"order": {"stored_by_class": run_order("stored"), "batches_shuffled": run_order("batch_order"),
                        "examples_shuffled": run_order("examples"), "pairs_broken": run_order("pairs_broken")},
              "heads": heads(), "labels": labels()}
    for k in result["order"]:
        result["order"][k].pop("batch_loss")
    path = U.REF / "heads_and_order.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # LF on every platform, as git stores it
        f.write(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=1, ensure_ascii=False))
