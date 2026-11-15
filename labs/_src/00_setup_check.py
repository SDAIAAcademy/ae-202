# %% [markdown]
# # Setup check
#
# Run every cell once before Day 1. Seven check marks mean that the labs will run here. On a
# laptop the course needs no key and no graphics processor. One model file is downloaded,
# in Lab 2.2: the pretrained image model, about 45 MB. Everything else that a lab needs is
# in the repository or is drawn on this computer.

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

import importlib
import numpy as np
import nawl as U

ok = 0


def check(passed, text):
    global ok
    ok += bool(passed)
    print(("✓ " if passed else "✗ ") + text)


for name in ["numpy", "matplotlib", "torch", "torchvision", "transformers"]:
    try:
        m = importlib.import_module(name)
        print(f"  {name} {m.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: {e}")
import torch
check(torch.tensor([1.0, 2.0]).sum().item() == 3.0, "the tensor library works")

# %%
# The Day 1 swatches and the Day 2 photographs are drawn here on first use (under a minute)
Xtr, ytr, Xva, yva, _, _ = U.swatches()
check(tuple(Xtr.shape) == (3000, 576), f"the swatches of Day 1: {tuple(Xtr.shape)}")
X, y = U.images("test")
check(tuple(X.shape) == (1000, 3, 48, 48), f"the photographs of Day 2: {tuple(X.shape)} in the test set")

# %%
hist = U.fit(U.mlp(), Xtr[:640], ytr[:640], Xva, yva, epochs=2, verbose=False)
check(hist["train_loss"][1] < hist["train_loss"][0], f"a small network trains: loss {hist['train_loss'][0]:.2f}, then {hist['train_loss'][1]:.2f}")

# %%
docs, queries = U.descriptions(), U.queries()
v = U.embed([docs[0]["text"]], prefix="passage: ", live=False)
check(len(docs) == 4000 and len(queries) == 50 and v.shape == (1, 384),
      f"the catalogue of Day 3: {len(docs):,} descriptions, {len(queries)} queries, supplied vectors of {v.shape[1]} numbers")

# %%
rows = U.runs("d4_1") + U.runs("d4_2") + U.runs("d4_3")
check(len(rows) > 800, f"the logged runs of Day 4: {len(rows)} generations of two small open-weight models")

# %%
backend = U.llm_backend()
check(True, "a live language model: " + (backend if backend else "none here, and none is needed: Day 4 reads the logged runs"))
print(f"\n{ok} of 7 checks passed. Mode: {U.MODE}. Set AE202_QUICK=1 to run every lab on less data in a few seconds.")
assert ok == 7, "ask the trainer before Day 1: one of the checks failed"

# %% [markdown]
# ## What runs where
#
# | | On a laptop | In Colab |
# |---|---|---|
# | Days 1 and 2 | everything, on the processor | everything; a graphics processor is optional |
# | Day 3 | supplied vectors of the embedding model | the embedding model itself |
# | Day 4 | logged runs of the two language models | the instruction-tuned model itself, live |
#
# A free hosted endpoint is the third way to a live model on Day 4: set `LLM_API_KEY`,
# `LLM_BASE_URL` and `LLM_MODEL` before you start the notebook. Nothing in the course needs it.
