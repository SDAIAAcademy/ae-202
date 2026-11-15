# %% [markdown]
# # Lab 3.3 · Catalogue search
#
# Day 3 · Session 3 (Sentence embeddings) · 25 min.
#
# Rayan Al-Sudais wants customers to find products by describing them. You build Khayt, the
# search over the cooperative's 4,000 descriptions: one vector per description, cosine
# similarity, a brute-force search, and recall at 5 on 50 queries that Rayan's team labelled.
# Then you compare it with the simpler alternative: matching keywords.
#
# **Tasks**
# - [ ] Embed and compare six sentences · 5 min
# - [ ] Brute-force search over the catalogue · 7 min
# - [ ] Recall at 5 on 50 queries · 8 min
# - [ ] Against a keyword baseline · 5 min
# - [ ] ✱ Stretch · Time the search and estimate it for a million descriptions
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# The lab runs on supplied vectors of a multilingual embedding model. In Colab (or with
# `AE202_MODELS=load`) it also loads the model and embeds live.

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
import nawl as U

docs = U.descriptions()
queries = U.queries()
doc_ids = np.array([d["id"] for d in docs])
print(f"{len(docs)} descriptions ({sum(d['lang'] == 'ar' for d in docs)} Arabic, {sum(d['lang'] == 'en' for d in docs)} English),"
      f" {len(queries)} labelled queries; embedding model: {U.EMBED_MODEL}; live model: {U.embedder_available()}")
print(queries[0])

# %% [markdown]
# ## 1 · Embed and compare six sentences
#
# `U.embed(texts)` returns one vector of 384 numbers per text, of length 1. For vectors of
# length 1 the cosine similarity is the dot product.
#
# ✏️ **Exercise** · Return the matrix of cosine similarities between the rows of `A` and the
# rows of `B` (do not assume length 1: divide by the lengths).

# %%
def cosine(A, B):
    # BEGIN SOLUTION
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
E = U.embed(U.EXTRA_TEXTS)
try:
    C = cosine(E, E)
except NotImplementedError:
    cosine = lambda A, B: (A / np.linalg.norm(A, axis=1, keepdims=True)) @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
    C = cosine(E, E)
assert C.shape == (6, 6) and np.allclose(np.diag(C), 1, atol=1e-3)
assert np.allclose(cosine(2 * E, 3 * E), C, atol=1e-4), "cosine similarity must not depend on the length of the vectors"
for i, t in enumerate(U.EXTRA_TEXTS):
    print(f"{i}  {t}")
print("\n   " + " ".join(f"{i:5d}" for i in range(6)))
for i, row in enumerate(C):
    print(f"{i}  " + " ".join(f"{v:5.2f}" for v in row))
U.save_reference("d3_3_six", {"texts": U.EXTRA_TEXTS, "cosine": [[round(float(v), 3) for v in row] for row in C]})

# %% [markdown]
# **Look at** two things. Every similarity is high, even between a rug and a question about
# returns: with this model, read similarities as a ranking, not as a scale from 0 to 1.
# And compare sentence 0 with its Arabic translation (1), its English paraphrase (2) and a
# different product (3).

# %% [markdown]
# ## 2 · Brute-force search over the catalogue
#
# `D` holds one vector per description. Searching is: similarity of the query to every
# description, then the `k` largest.
#
# ✏️ **Exercise** · Return the indices of the `k` rows of `D` most similar to the query
# vector `q`, most similar first.

# %%
D = U.embed([d["text"] for d in docs], prefix="passage: ")
if U.embedder_available():   # embed some descriptions yourself and compare with the supplied vectors
    live = U.embed([d["text"] for d in docs[:200]], prefix="passage: ", live=True)
    print(f"embedded 200 descriptions live; largest difference from the supplied vectors: {np.abs(live - D[:200]).max():.4f}")


def top_k(q, D, k=5):
    # BEGIN SOLUTION
    sims = D @ q
    return np.argsort(-sims)[:k]
    # END SOLUTION

# %%
Q = U.embed([q["text"] for q in queries])
try:
    top = top_k(Q[0], D)
except NotImplementedError:
    top_k = lambda q, D, k=5: np.argsort(-(D @ q))[:k]
    top = top_k(Q[0], D)
assert list(top) == list(np.argsort(-(D @ Q[0]))[:5]), "not the five most similar descriptions"
print("query:", queries[0]["text"], "| relevant:", queries[0]["relevant"])
for i in top:
    mark = "✓" if docs[i]["id"] in queries[0]["relevant"] else " "
    print(f"  {mark} {docs[i]['id']} {float(D[i] @ Q[0]):.3f}  {docs[i]['text']}")

# %% [markdown]
# ## 3 · Recall at 5 on 50 queries
#
# Each query has four relevant descriptions: two products, each described in both
# languages. Recall at 5 is the share of the relevant descriptions that appear among the
# first five results.
#
# ✏️ **Exercise** · Return recall at k for one query: `found` is the list of retrieved
# description ids, `relevant` the list of relevant ids.

# %%
def recall_at_k(found, relevant):
    # BEGIN SOLUTION
    return len(set(found) & set(relevant)) / len(relevant)
    # END SOLUTION

# %%
try:
    recall_at_k(["a"], ["a", "b"])
except NotImplementedError:
    recall_at_k = lambda found, relevant: len(set(found) & set(relevant)) / len(relevant)
assert recall_at_k(["a", "x", "y"], ["a", "b"]) == 0.5 and recall_at_k([], ["a"]) == 0.0
recalls = np.array([recall_at_k(doc_ids[top_k(qv, D)], q["relevant"]) for q, qv in zip(queries, Q)])
by_lang = {lang: float(recalls[[q["lang"] == lang for q in queries]].mean()) for lang in ("en", "ar", "mixed")}
print(f"recall at 5, all 50 queries: {recalls.mean():.3f}")
print("by query language:", {k: round(v, 3) for k, v in by_lang.items()})
print("queries with nothing relevant in the first five:", int((recalls == 0).sum()), "of", len(queries))

# %% [markdown]
# **Look at** the number without flinching. It is low. Before anyone tunes anything, two
# questions: is it better than something simpler, and where exactly does it fail?
# The first is the next task; the second is Lab 3.4.

# %% [markdown]
# ## 4 · Against a keyword baseline
#
# The simpler alternative: return the descriptions that share the rarest words with the
# query. `U.keyword_index` and `U.keyword_top_k` do that.
#
# ✏️ **Exercise** · Return the recall at 5 of the keyword search for every query, as an array.

# %%
index = U.keyword_index([d["text"] for d in docs])


def keyword_recalls():
    # BEGIN SOLUTION
    return np.array([recall_at_k(doc_ids[U.keyword_top_k(q["text"], index)], q["relevant"]) for q in queries])
    # END SOLUTION

# %%
try:
    kw = keyword_recalls()
except NotImplementedError:
    kw = np.array([recall_at_k(doc_ids[U.keyword_top_k(q["text"], index)], q["relevant"]) for q in queries])
kw_by_lang = {lang: float(kw[[q["lang"] == lang for q in queries]].mean()) for lang in ("en", "ar", "mixed")}
print(f"{'recall at 5':14s} {'all':>6s} {'en':>6s} {'ar':>6s} {'mixed':>6s}")
print(f"{'keywords':14s} {kw.mean():6.3f} {kw_by_lang['en']:6.3f} {kw_by_lang['ar']:6.3f} {kw_by_lang['mixed']:6.3f}")
print(f"{'embeddings':14s} {recalls.mean():6.3f} {by_lang['en']:6.3f} {by_lang['ar']:6.3f} {by_lang['mixed']:6.3f}")
better = int((recalls > kw).sum()), int((recalls < kw).sum())
print(f"embeddings better on {better[0]} queries, keywords better on {better[1]}, equal on {len(queries) - sum(better)}")
result = {"embeddings": {"all": float(recalls.mean()), **by_lang}, "keywords": {"all": float(kw.mean()), **kw_by_lang},
          "zero_recall_queries": int((recalls == 0).sum()), "embeddings_better": better[0], "keywords_better": better[1]}
U.save_run("d3_3_recall", result)
U.save_reference("d3_3_recall", result)
assert recalls.mean() > kw.mean(), "on these queries the embedding search should beat keywords overall"

# %% [markdown]
# **Look at** the comparison by language. A result is "good" or "bad" only against an
# alternative measured on the same queries. Where do keywords hold their own, and why
# there?

# %% [markdown]
# ## ✱ Stretch · How long does it take?
#
# Time `top_k(Q[0], D)` with `U.timeit`. Brute force compares the query with every
# description: estimate the time for a catalogue of a million. At what size would you move to
# an index that searches approximately (FAISS is one example), and what would you have to
# measure again after the move?

# %% [markdown]
# ## Check yourself
#
# 1. Two unrelated sentences had a similarity above 0.7. Is the model broken?
# 2. Why is recall at 5 here at most 1.0 only if both products are found in both languages?
# 3. Keywords did well on some English queries. What do those queries have in common?
#
# <details><summary>Answers</summary>
#
# 1. No. This model's similarities live in a narrow band near the top; only their order
#    carries meaning. A fixed threshold copied from another model would be wrong here.
# 2. A query has four relevant descriptions. Recall counts all four, so a search that only
#    ever returns the query's language cannot exceed 0.5.
# 3. They use the catalogue's own words ("striped", "rug"). Exact words are what keyword
#    search is good at; it has nothing for a synonym or for the other language.
# </details>
#
# ## Carry forward
#
# The vectors `D` and `Q` and your three functions are the starting point of Lab 3.4, which
# loads them again from the supplied files. Your result is in `runs/d3_3_recall.json`.
