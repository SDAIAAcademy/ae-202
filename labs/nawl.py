"""Helpers shared by the labs of AE-202 (import nawl as U).

The case study is the Al-Nawl Crafts Cooperative: textile-pattern images, a bilingual
catalogue of product descriptions and a small open-weight language model. Everything here is
plain PyTorch and NumPy that you could write yourself after Day 1.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
CACHE = LABS / "cache"
RUNS = LABS / "runs"            # what your own runs write (not committed)
REF = DATA / "reference"

QUICK = os.environ.get("AE202_QUICK", "") not in ("", "0")
MODE = "quick" if QUICK else "full"
IN_COLAB = "google.colab" in sys.modules
CLASSES = ["stripe", "chevron", "diamond", "check", "triangle", "dot", "wave", "star"]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

EMBED_MODEL = os.environ.get("AE202_EMBED_MODEL", "intfloat/multilingual-e5-small")
LLM_INSTRUCT = os.environ.get("AE202_LLM_INSTRUCT", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_BASE = os.environ.get("AE202_LLM_BASE", "Qwen/Qwen2.5-0.5B")
N_CTX = 2048   # the context window the runtime is started with, as a small local runtime is
TOKENISERS = {"gpt2": "openai-community/gpt2", "e5": "intfloat/multilingual-e5-small", "qwen": "Qwen/Qwen2.5-0.5B-Instruct"}


def pick(full, quick):
    """The full value, or the small one in quick mode (AE202_QUICK=1)."""
    return quick if QUICK else full


def reference(name: str):
    """A supplied reference result: keeps a notebook running when an exercise is not done yet."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def save_reference(name: str, value) -> None:
    """Solutions write their results here when AE202_WRITE_REF=1 (the course author's switch)."""
    if os.environ.get("AE202_WRITE_REF"):
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def save_run(name: str, value) -> Path:
    """Write one of your own results to labs/runs/ (the evidence the course asks for)."""
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{name}.json"
    out.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def seed_everything(seed: int = 0) -> None:
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- images

def ensure_data() -> None:
    """Draw the image pack on first use (about 20 seconds). Nothing is downloaded."""
    if not (CACHE / "textiles.npz").exists():
        import prepare_data
        print("drawing the textile images (once) ...")
        prepare_data.write_images()


_pack = None


def textiles() -> dict:
    """The 8,000 catalogue photographs: X (uint8, N x 48 x 48 x 3), y, and the index arrays
    train (6,000), val (1,000), test (1,000) and small (the 800 labelled images of Day 2)."""
    global _pack
    if _pack is None:
        ensure_data()
        with np.load(CACHE / "textiles.npz") as d:
            _pack = {k: d[k] for k in d.files}
    return _pack


def images(split: str, limit: int | None = None):
    """(X, y) of one split as tensors: X is float32, N x 3 x 48 x 48, in [0, 1]."""
    import torch
    d = textiles()
    idx = d[split][:limit] if limit else d[split]
    X = torch.from_numpy(d["X"][idx]).permute(0, 3, 1, 2).float() / 255.0
    return X, torch.from_numpy(d["y"][idx])


def swatches():
    """The Day 1 data: 4,000 swatches laid straight on the sorting table, as grey 24 x 24 tensors
    flattened to 576 numbers in [-0.5, 0.5]. Returns Xtr, ytr (3,000), Xva, yva (500), Xte, yte (500)."""
    import torch
    f = CACHE / "swatches.npz"
    if not f.exists():
        import prepare_data
        CACHE.mkdir(exist_ok=True)
        y = np.arange(4000) % 8
        S = np.stack([prepare_data.draw(int(c), 100000 + i, 100000 + i, style=2) for i, c in enumerate(y)])
        g = S.astype(np.float32).mean(-1) / 255.0
        g = g.reshape(len(g), 24, 2, 24, 2).mean((2, 4)).reshape(len(g), -1) - 0.5
        np.savez_compressed(f, X=g.astype(np.float32), y=y.astype(np.int64))
    with np.load(f) as d:
        X, y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    return X[:3000], y[:3000], X[3000:3500], y[3000:3500], X[3500:], y[3500:]


def mlp(hidden: int = 64, dropout: float = 0.0, n_in: int = 576):
    import torch.nn as nn
    layers = [nn.Linear(n_in, hidden), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers, nn.Linear(hidden, len(CLASSES)))


def small_cnn():
    """The compact convolutional network of Day 2: three convolutions, 25 thousand weights."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(CLASSES)))


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate(model, X, y, bs: int = 250):
    """(mean loss, accuracy) without gradients, with the model in evaluation mode."""
    import torch
    import torch.nn.functional as F
    was = model.training
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])
    model.train(was)
    return F.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()


def predict(model, X, bs: int = 250):
    """Class probabilities, without gradients."""
    import torch
    was = model.training
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])
    model.train(was)
    return torch.softmax(out, 1)


def fit(model, Xtr, ytr, Xva, yva, epochs: int = 5, lr: float = 1e-3, bs: int = 32, seed: int = 0,
        weight_decay: float = 0.0, optimiser: str = "adam", params=None, augment=None, verbose: bool = True):
    """The training loop of Lab 1.2, kept here for the later labs. Returns the history:
    {"train_loss": [...], "val_loss": [...], "val_acc": [...]} with one value per epoch."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    params = params or model.parameters()
    opt = (torch.optim.Adam(params, lr=lr, weight_decay=weight_decay) if optimiser == "adam"
           else torch.optim.SGD(params, lr=lr, weight_decay=weight_decay))
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        total = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            xb = augment(Xtr[b], g) if augment else Xtr[b]
            loss = F.cross_entropy(model(xb), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        vl, va = evaluate(model, Xva, yva)
        hist["train_loss"].append(total / len(Xtr))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if verbose:
            print(f"  epoch {epoch + 1:2d}  train loss {total / len(Xtr):.3f}  val loss {vl:.3f}  val acc {va:.3f}")
    return hist


def plot_curves(histories: dict, title: str = "", key_train: str = "train_loss", key_val: str = "val_loss"):
    """Training (dashed) and validation (solid) loss per epoch for one or more named runs."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (name, h) in enumerate(histories.items()):
        e = range(1, len(h[key_val]) + 1)
        ax.plot(e, h[key_train], "--", color=f"C{k}", alpha=0.7)
        ax.plot(e, h[key_val], "-", color=f"C{k}", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (dashed: training, solid: validation)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def show_images(X, titles=None, cols: int = 10, size: float = 1.2):
    """A row of images. X is N x 3 x H x W in [0, 1], or N x H x W for one channel."""
    import matplotlib.pyplot as plt
    X = np.asarray(X)
    rows = int(np.ceil(len(X) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.15), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, x in enumerate(X):
        ax = axes[i // cols][i % cols]
        if x.ndim == 3:
            ax.imshow(np.clip(np.transpose(x, (1, 2, 0)), 0, 1))
        else:
            ax.imshow(x, cmap="gray")
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=7)
    fig.tight_layout()
    return fig


def backbone():
    """A ResNet-18 trained on ImageNet-1k (torchvision; about 45 MB on first use)."""
    import torchvision
    return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)


def to_backbone_input(X, size: int = 64):
    """Resize to size x size and normalise with the statistics the backbone was trained with."""
    import torch
    import torch.nn.functional as F
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (F.interpolate(X, size=size, mode="bilinear", align_corners=False) - mean) / std


def augment(X, g=None):
    """The augmentation of Lab 2.3: a random flip in each direction and a shift of up to 4 pixels."""
    import torch
    if torch.rand(1, generator=g).item() < 0.5:
        X = X.flip(3)
    if torch.rand(1, generator=g).item() < 0.5:
        X = X.flip(2)
    dx, dy = torch.randint(-4, 5, (2,), generator=g).tolist()
    return torch.roll(X, (dy, dx), (2, 3))


def confusion(y_true, y_pred, n: int = 8):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m


def print_confusion(m) -> None:
    print(" " * 10 + " ".join(f"{c[:5]:>5s}" for c in CLASSES) + "   (columns: predicted)")
    for i, row in enumerate(m):
        print(f"{CLASSES[i]:>9s} " + " ".join(f"{v:5d}" for v in row))


# --------------------------------------------------------------------------- catalogue text and embeddings

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def descriptions() -> list[dict]:
    """The 4,000 product descriptions: id, lang (ar or en), text and the attributes."""
    return read_jsonl(DATA / "descriptions.jsonl")


def queries() -> list[dict]:
    """The 50 labelled queries: id, lang (ar, en or mixed), text, relevant (description ids)."""
    return read_jsonl(DATA / "queries.jsonl")


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


_vectors = None
_embedder = None


def supplied_vectors() -> dict:
    """Vectors computed once with EMBED_MODEL for every text the labs embed, keyed by text."""
    global _vectors
    if _vectors is None:
        with np.load(DATA / "embeddings" / "vectors.npz") as d:
            V = d["vectors"].astype(np.float32)
        keys = json.loads((DATA / "embeddings" / "keys.json").read_text(encoding="utf-8"))
        _vectors = {k: V[i] for i, k in enumerate(keys)}
    return _vectors


def embedder_available() -> bool:
    """True when the embedding model may be loaded here: in Colab, or when AE202_MODELS=load."""
    return IN_COLAB or os.environ.get("AE202_MODELS", "") == "load"


def embed(texts: list[str], prefix: str = "query: ", live: bool | None = None) -> np.ndarray:
    """Unit-length sentence embeddings (N x 384). prefix is what the model expects before a
    text: "query: " for a search query, "passage: " for a description, "" for none.
    With live=None the model is used when it may be loaded, otherwise the supplied vectors."""
    global _embedder
    if live is None:
        live = embedder_available()
    if live:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        return _embedder.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False).astype(np.float32)
    table = supplied_vectors()
    missing = [t for t in texts if _key(prefix + t) not in table]
    if missing:
        raise KeyError(f"no supplied vector for {missing[0]!r}: open the lab in Colab, or set AE202_MODELS=load "
                       "to embed new text with the model (a download of about 470 MB)")
    return np.stack([table[_key(prefix + t)] for t in texts])


def tokeniser(name: str):
    """One of the three tokenisers of Lab 3.1 ("gpt2", "e5", "qwen"): only the vocabulary file
    is fetched (a few megabytes), never a model."""
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(TOKENISERS[name])


# --------------------------------------------------------------------------- the language model

def runs(lab: str) -> list[dict]:
    """The supplied logged runs of one lab (data/runs/<lab>.jsonl), then your own (runs/<lab>.jsonl)."""
    rows = read_jsonl(DATA / "runs" / f"{lab}.jsonl")
    mine = RUNS / f"{lab}.jsonl"
    return rows + (read_jsonl(mine) if mine.exists() else [])


def llm_backend() -> str | None:
    """Where a live generation runs: "runtime" (the model is loaded in this notebook: Colab, or
    AE202_LLM=runtime on a laptop), "hosted" (a free hosted endpoint, with LLM_API_KEY), or None."""
    choice = os.environ.get("AE202_LLM", "")
    if choice == "runtime" or (IN_COLAB and choice != "hosted" and not os.environ.get("LLM_API_KEY")):
        return "runtime"
    if os.environ.get("LLM_API_KEY"):
        return "hosted"
    return None


_llm = {}


def _load(model_id: str):
    if model_id not in _llm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model.eval()
        _llm[model_id] = (tok, model)
    return _llm[model_id]


def render(messages, kind: str = "instruct") -> str:
    """The text the model really receives. An instruction-tuned model gets the chat template
    around the messages; a base model gets the text of the last message and nothing else."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if kind == "base":
        return messages[-1]["content"]
    tok, _ = _load(LLM_INSTRUCT) if LLM_INSTRUCT in _llm else (None, None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(LLM_INSTRUCT)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(messages, kind: str = "instruct", temperature: float = 0.0, top_p: float = 1.0,
             max_new_tokens: int = 60, n: int = 1, seed: int = 0, n_ctx: int = N_CTX) -> list[dict]:
    """n live generations. Each result: text, prompt_tokens, new_tokens, truncated (tokens cut
    from the start because the prompt did not fit in n_ctx), seconds. Raises RuntimeError when
    no backend is set up: the labs then read the supplied logged runs only."""
    backend = llm_backend()
    if backend is None:
        raise RuntimeError("no live model here: the lab continues on the supplied logged runs")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if backend == "hosted":
        return _generate_hosted(messages, kind, temperature, top_p, max_new_tokens, n, seed)
    import torch
    tok, model = _load(LLM_BASE if kind == "base" else LLM_INSTRUCT)
    ids = tok(render(messages, kind), return_tensors="pt").input_ids
    room = n_ctx - max_new_tokens
    cut = max(0, ids.shape[1] - room)
    if cut:
        ids = ids[:, -room:]          # a local runtime keeps the end of a prompt that does not fit
    torch.manual_seed(seed)
    t = time.time()
    kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=tok.eos_token_id,
              repetition_penalty=1.0)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        kw.update(do_sample=False, num_return_sequences=1, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(ids, **kw)
    secs = time.time() - t
    res = []
    for row in out:
        new = row[ids.shape[1]:]
        new = new[new != tok.eos_token_id] if tok.eos_token_id is not None else new
        res.append({"text": tok.decode(new, skip_special_tokens=True).strip(), "prompt_tokens": int(ids.shape[1]),
                    "new_tokens": int(len(new)), "truncated": int(cut), "seconds": round(secs / len(out), 3)})
    return res * (n if temperature == 0 else 1)


def _generate_hosted(messages, kind, temperature, top_p, max_new_tokens, n, seed) -> list[dict]:
    import httpx
    base = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        raise RuntimeError("set LLM_MODEL to a small open-weight model your endpoint lists as free")
    res = []
    for k in range(n):
        t = time.time()
        r = httpx.post(f"{base}/chat/completions", timeout=120,
                       headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
                       json={"model": model, "messages": messages, "temperature": temperature, "top_p": top_p,
                             "max_tokens": max_new_tokens, "seed": seed + k})
        r.raise_for_status()
        j = r.json()
        use = j.get("usage", {})
        res.append({"text": j["choices"][0]["message"]["content"].strip(), "prompt_tokens": use.get("prompt_tokens"),
                    "new_tokens": use.get("completion_tokens"), "truncated": 0, "seconds": round(time.time() - t, 3)})
    return res


def try_live(fn, what: str = "the live run"):
    """Run fn() when a live model is set up; otherwise say so and return None."""
    if llm_backend() is None:
        print(f"(skipped {what}: no live model here; the supplied logged runs carry the lab)")
        return None
    try:
        return fn()
    except Exception as e:  # a wrong key, a model the endpoint does not list, no network
        print(f"(skipped {what}: the live model did not answer: {type(e).__name__}: {str(e)[:160]})")
        return None


def words(text: str) -> list[str]:
    return text.split()


def timeit(fn, repeat: int = 5) -> float:
    """The median time of fn() in milliseconds, after one warm-up call."""
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


_AR_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ـ": None,
                          **{chr(c): None for c in range(0x64B, 0x653)}})


def normalise_ar(text: str) -> str:
    """Fold the Arabic spellings that vary between writers: hamza on alef, ta marbuta, alef
    maqsura, tatweel and the short-vowel marks."""
    return text.translate(_AR_FOLD)


EXTRA_TEXTS = [  # the sentences of the similarity warm-up in Lab 3.3
    "a red striped rug", "سجادة مخططة باللون الأحمر", "a crimson carpet with lines", "a blue dotted shawl",
    "how do I return an order?", "زولية حمراء مقلمة",
]


def partial_backbone(m):
    """Wrap a ResNet so that only its last block and its head are trained: the earlier blocks
    run without gradients and stay in evaluation mode (their batch-norm statistics are kept)."""
    import torch
    import torch.nn as nn

    class Partial(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            for part in (m.conv1, m.bn1, m.layer1, m.layer2, m.layer3):
                part.eval()
            with torch.no_grad():
                x = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
            return m.fc(torch.flatten(m.avgpool(m.layer4(x)), 1))

    return Partial(m)


def thumbs(X):
    """Colour 24 x 24 thumbnails of images (N x 3 x 48 x 48), each scaled to mean 0 and length 1:
    a cheap fingerprint for finding the same photograph uploaded twice."""
    g = X.reshape(len(X), 3, 24, 2, 24, 2).mean((3, 5)).reshape(len(X), -1)
    g = g - g.mean(1, keepdim=True)
    return g / g.norm(dim=1, keepdim=True)


def pairs(n: int = 12) -> list[dict]:
    """The first n products of the catalogue with both of their descriptions:
    {"product", "en", "ar"}: the same product described in two languages."""
    by = {}
    for d in descriptions():
        by.setdefault(d["product"], {"product": d["product"]})[d["lang"]] = d["text"]
    return [by[k] for k in sorted(by)[:n]]


def keyword_index(texts: list[str]):
    """A keyword baseline: every text as its set of lower-case words, and how rare each word is."""
    import math
    import re
    sets = [set(re.findall(r"\w+", t.lower())) for t in texts]
    df = {}
    for s in sets:
        for w in s:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(len(texts) / n) for w, n in df.items()}
    return sets, idf


def keyword_top_k(query: str, index, k: int = 5) -> list[int]:
    """The k texts that share the rarest words with the query (ties: the earlier text)."""
    import re
    sets, idf = index
    q = set(re.findall(r"\w+", query.lower()))
    scores = np.array([sum(idf[w] for w in q & s) for s in sets])
    return np.argsort(-scores, kind="stable")[:k].tolist()
