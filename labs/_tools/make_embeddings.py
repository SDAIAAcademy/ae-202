"""Compute the supplied vectors of Day 3 with the embedding model (the course author runs this
once; it loads the model, about 470 MB).

    uv run python _tools/make_embeddings.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nawl as U  # noqa: E402

docs = [d["text"] for d in U.descriptions()]
qs = [q["text"] for q in U.queries()]
texts = (["passage: " + t for t in docs] + ["query: " + t for t in qs] + [t for t in qs]
         + ["query: " + U.normalise_ar(t) for t in qs] + ["passage: " + U.normalise_ar(t) for t in docs]
         + ["query: " + t for t in U.EXTRA_TEXTS])
texts = list(dict.fromkeys(texts))
from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(U.EMBED_MODEL, device="cpu")
V = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
out = U.DATA / "embeddings"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "vectors.npz", vectors=V.astype(np.float16))
(out / "keys.json").write_text(json.dumps([U._key(t) for t in texts]), encoding="utf-8")
(out / "MODEL.txt").write_text(f"{U.EMBED_MODEL}\n{len(texts)} texts, {V.shape[1]} dimensions, float16\n", encoding="utf-8")
print(len(texts), V.shape)
