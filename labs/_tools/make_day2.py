"""Write the supplied files of Day 2 (the course author runs this once per edition, about
four minutes on a laptop processor):

    data/checkpoints/cnn_warm.pt   the small convolutional network after 8 epochs on the 6,000
                                   training photographs (Lab 2.1 continues from it)
    data/runs/day2_scratch.json    the same network trained from scratch on the 800 labelled
                                   photographs only (Labs 2.2 and 2.3 compare against it)

    uv run python _tools/make_day2.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nawl as U  # noqa: E402

Xtr, ytr = U.images("train")
Xva, yva = U.images("val")
Xte, yte = U.images("test")
Xsm, ysm = U.images("small")
(U.DATA / "checkpoints").mkdir(parents=True, exist_ok=True)
(U.DATA / "runs").mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
model = U.small_cnn()
t = time.time()
hist = U.fit(model, Xtr - 0.5, ytr, Xva - 0.5, yva, epochs=8, lr=2e-3, bs=64)
torch.save(model.state_dict(), U.DATA / "checkpoints" / "cnn_warm.pt")
(U.DATA / "checkpoints" / "cnn_warm.json").write_text(json.dumps(
    {"what": "small_cnn after 8 epochs on the 6,000 training photographs, Adam 0.002, batches of 64, inputs in [-0.5, 0.5]",
     "history": hist, "minutes": round((time.time() - t) / 60, 1)}, indent=1) + "\n", encoding="utf-8")

torch.manual_seed(0)
scratch = U.small_cnn()
t = time.time()
hist = U.fit(scratch, Xsm - 0.5, ysm, Xva - 0.5, yva, epochs=20, lr=2e-3, bs=32)
loss, acc = U.evaluate(scratch, Xte - 0.5, yte)
(U.DATA / "runs" / "day2_scratch.json").write_text(json.dumps(
    {"what": "small_cnn trained from scratch on the 800 labelled photographs, 20 epochs, Adam 0.002, batches of 32",
     "test_accuracy": acc, "val_accuracy": hist["val_acc"][-1], "train_loss": hist["train_loss"][-1],
     "val_loss": hist["val_loss"][-1], "seconds": round(time.time() - t, 1), "history": hist}, indent=1) + "\n", encoding="utf-8")
print("scratch on 800: test accuracy", round(acc, 3))
