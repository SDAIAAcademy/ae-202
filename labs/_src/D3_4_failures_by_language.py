# %% [markdown]
# # Lab 3.4 · Failures by language
#
# Day 3 · Session 4 (Multilingual search and its failures) · 25 min.
#
# Khayt's recall at 5 is low. A number does not say why. You split it by the language of
# the query, look at which language the results come back in, try two fixes and measure
# both, and write up three failure cases with a likely cause each: the evidence this
# outcome asks for.
#
# **Tasks**
# - [ ] Recall at 5 by query language · 5 min
# - [ ] Same-language and cross-language hits · 6 min
# - [ ] Two fixes: merge by language, pair by product · 6 min
# - [ ] Three failure cases with a likely cause · 8 min
# - [ ] ✱ Stretch · Fold Arabic spelling variants and measure again
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# This lab starts from the supplied vectors: it does not need your Lab 3.3.

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
doc_lang = np.array([d["lang"] for d in docs])
by_id = {d["id"]: d for d in docs}
D = U.embed([d["text"] for d in docs], prefix="passage: ")
Q = U.embed([q["text"] for q in queries])
top5 = [np.argsort(-(D @ qv))[:5] for qv in Q]
recall = lambda found, relevant: len(set(found) & set(relevant)) / len(relevant)
recalls = np.array([recall(doc_ids[t], q["relevant"]) for t, q in zip(top5, queries)])
print(f"recall at 5 over {len(queries)} queries: {recalls.mean():.3f}")

# %% [markdown]
# ## 1 · Recall at 5 by query language
#
# ✏️ **Exercise** · Return `{"en": ..., "ar": ..., "mixed": ...}`: the mean recall of the
# queries of each language, and print how many queries each mean rests on.

# %%
def recall_by_language(recalls, queries):
    # BEGIN SOLUTION
    out = {}
    for lang in ("en", "ar", "mixed"):
        mask = np.array([q["lang"] == lang for q in queries])
        out[lang] = float(recalls[mask].mean())
        print(f"  {lang:5s} {int(mask.sum()):2d} queries  recall at 5 {out[lang]:.3f}")
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    by_lang = recall_by_language(recalls, queries)
except NotImplementedError:
    by_lang = {lang: float(recalls[[q["lang"] == lang for q in queries]].mean()) for lang in ("en", "ar", "mixed")}
    print(by_lang)
assert set(by_lang) == {"en", "ar", "mixed"} and abs(by_lang["en"] - recalls[:20].mean()) < 1e-9

# %% [markdown]
# **Look at** the sizes as well as the means: 20, 20 and 10 queries. One query is 5 or 10
# points of its group. These numbers rank the three groups; they do not measure them finely.

# %% [markdown]
# ## 2 · Same-language and cross-language hits
#
# Every query has relevant descriptions in both languages. Which ones does the search find?
#
# ✏️ **Exercise** · Over the English and Arabic queries (leave the mixed ones out), return
# three numbers: the share of the top-5 results written in the query's own language; the
# share of same-language relevant descriptions that were found; the share of other-language
# relevant descriptions that were found.

# %%
def language_wall(top5, queries):
    # BEGIN SOLUTION
    own, same, cross = [], [], []
    for t, q in zip(top5, queries):
        if q["lang"] == "mixed":
            continue
        own += [doc_lang[i] == q["lang"] for i in t]
        found = set(doc_ids[t])
        for rid in q["relevant"]:
            (same if by_id[rid]["lang"] == q["lang"] else cross).append(rid in found)
    return float(np.mean(own)), float(np.mean(same)), float(np.mean(cross))
    # END SOLUTION

# %%
def _wall(top5, queries):
    own, same, cross = [], [], []
    for t, q in zip(top5, queries):
        if q["lang"] == "mixed":
            continue
        own += [doc_lang[i] == q["lang"] for i in t]
        found = set(doc_ids[t])
        for rid in q["relevant"]:
            (same if by_id[rid]["lang"] == q["lang"] else cross).append(rid in found)
    return float(np.mean(own)), float(np.mean(same)), float(np.mean(cross))

try:
    own, same, cross = language_wall(top5, queries)
except NotImplementedError:
    own, same, cross = _wall(top5, queries)
assert (own, same, cross) == _wall(top5, queries)
print(f"share of the top-5 results in the query's own language: {own:.2f}")
print(f"relevant descriptions found: same language {same:.2f}, other language {cross:.2f}")

# where does the best other-language relevant description rank?
ranks = []
for qv, q in zip(Q, queries):
    if q["lang"] == "mixed":
        continue
    order = np.argsort(-(D @ qv))
    position = {doc_ids[i]: r + 1 for r, i in enumerate(order)}
    ranks.append(min(position[r] for r in q["relevant"] if by_id[r]["lang"] != q["lang"]))
print(f"rank of the best other-language relevant description: median {int(np.median(ranks))}, best {min(ranks)} (of {len(docs)})")

# %% [markdown]
# **Look at** the three lines together. The model is multilingual: it embeds both languages
# in one space. Yet for this catalogue it puts almost every description of the query's own
# language ahead of every description of the other one. Half of the relevant descriptions
# are behind a wall, so recall at 5 could never pass 0.5.

# %% [markdown]
# ## 3 · Two fixes: merge by language, pair by product
#
# **Fix A: merge by language.** Search the Arabic and the English descriptions separately
# and return three results from the query's language and two from the other. It is given.
#
# **Fix B: pair by product.** The catalogue knows that two descriptions belong to one
# product. Take the best three results and add, after each, the other-language description
# of the same product, until there are five.
#
# ✏️ **Exercise** · Write fix B: return five document indices for the query vector `qv`.
# `sibling[i]` is the index of the other description of the same product as document `i`.

# %%
product_rows = {}
for i, d in enumerate(docs):
    product_rows.setdefault(d["product"], []).append(i)
sibling = np.zeros(len(docs), dtype=int)
for a, b in product_rows.values():
    sibling[a], sibling[b] = b, a


def merge_by_language(qv, lang):
    sims = D @ qv
    first = "en" if lang == "en" else "ar"
    out = []
    for which, n in ((first, 3), ("ar" if first == "en" else "en", 2)):
        rows = np.where(doc_lang == which)[0]
        out += list(rows[np.argsort(-sims[rows])[:n]])
    return np.array(out)


def pair_by_product(qv):
    # BEGIN SOLUTION
    out = []
    for i in np.argsort(-(D @ qv)):
        for j in (i, sibling[i]):
            if j not in out:
                out.append(int(j))
        if len(out) >= 5:
            break
    return np.array(out[:5])
    # END SOLUTION

# %%
def _pair(qv):
    out = []
    for i in np.argsort(-(D @ qv)):
        for j in (i, sibling[i]):
            if j not in out:
                out.append(int(j))
        if len(out) >= 5:
            break
    return np.array(out[:5])

try:
    pair_by_product(Q[0])
except NotImplementedError:
    pair_by_product = _pair
assert len(pair_by_product(Q[0])) == 5 and list(pair_by_product(Q[0])) == list(_pair(Q[0]))
rec_a = np.array([recall(doc_ids[merge_by_language(qv, q["lang"])], q["relevant"]) for qv, q in zip(Q, queries)])
rec_b = np.array([recall(doc_ids[pair_by_product(qv)], q["relevant"]) for qv, q in zip(Q, queries)])
group = lambda r: {lang: round(float(r[[q["lang"] == lang for q in queries]].mean()), 3) for lang in ("en", "ar", "mixed")}
print(f"{'recall at 5':22s} {'all':>6s}   by language")
for name, r in (("plain search", recalls), ("A: merge by language", rec_a), ("B: pair by product", rec_b)):
    print(f"{name:22s} {r.mean():6.3f}   {group(r)}")
assert rec_b.mean() > recalls.mean()

# %% [markdown]
# **Look at** fix A first: it does not help. Giving the other language two seats is useless
# when the right description is not among that language's best two either. Fix B works
# because it does not ask the model to cross the wall: it uses a fact the catalogue already
# holds. The cheapest improvement to a learned system is often a piece of structure you
# already own.

# %% [markdown]
# ## 4 · Three failure cases with a likely cause
#
# Fix B leaves many queries at zero. The cell below lists them. Read the query, read what
# came back, and decide on a likely cause from this list (or name your own):
#
# - `synonym`: the query uses a word the catalogue does not ("carpet" for "rug")
# - `dialect`: a Gulf dialect word for the product
# - `arabizi`: Arabic written in Latin letters and digits
# - `spelling`: the same word written differently (hamza, ta marbuta)
# - `wrong attribute`: the right kind of product comes back with another colour or pattern
# - `negation or number`: the query excludes something or names a size

# %%
failed = [k for k in range(len(queries)) if rec_b[k] == 0]
print(f"{len(failed)} of {len(queries)} queries still find nothing relevant with fix B\n")
for k in failed:
    q = queries[k]
    want = by_id[q["relevant"][0]]
    got = [f"{docs[i]['type']}/{docs[i]['pattern']}/{docs[i]['colour']}" for i in pair_by_product(Q[k])[:3]]
    print(f"{q['id']} [{q['lang']:5s}] {q['text']}\n     wanted {want['type']}/{want['pattern']}/{want['colour']}   got {got}")

# %% [markdown]
# ✏️ **Exercise** · Choose three of the failed queries with three **different** causes and
# fill in the list: the query id, what came back in your own words, the likely cause, and
# one fix you would try. This list is your evidence.

# %%
def three_failure_cases():
    # BEGIN SOLUTION
    return [
        {"query": "Q11", "what_came_back": "cushion covers, but with dots and triangles in other colours: the kind of product is right, the pattern and colour are not",
         "likely_cause": "synonym", "fix_to_try": "expand the query with the catalogue's own words (zigzag to chevron, indigo to blue) before embedding it"},
        {"query": "Q37", "what_came_back": "wall hangings, belts and blankets that are green: the colour matched, the product did not",
         "likely_cause": "dialect", "fix_to_try": "a small table from dialect words to catalogue words, built from the shop's search log"},
        {"query": "Q49", "what_came_back": "shawls in other patterns and colours: nothing of the query was understood",
         "likely_cause": "arabizi", "fix_to_try": "detect Latin-script Arabic and transliterate it to Arabic script before embedding"},
    ]
    # END SOLUTION

# %%
CAUSES = {"synonym", "dialect", "arabizi", "spelling", "wrong attribute", "negation or number"}
try:
    cases = three_failure_cases()
except NotImplementedError:
    cases = U.reference("d3_4_cases")["cases"]
failed_ids = {queries[k]["id"] for k in failed}
assert len(cases) == 3, "three cases"
assert all(c["query"] in failed_ids for c in cases), "each case must be one of the failed queries listed above"
assert len({c["likely_cause"] for c in cases}) == 3, "three different causes"
assert all(len(c["what_came_back"]) > 20 and len(c["fix_to_try"]) > 20 for c in cases), "say what came back and what you would try"
for c in cases:
    print(f"{c['query']}: {c['likely_cause']}\n   came back: {c['what_came_back']}\n   to try:    {c['fix_to_try']}")
report = {"recall_at_5": {"plain": {"all": float(recalls.mean()), **group(recalls)}, "merge_by_language": {"all": float(rec_a.mean()), **group(rec_a)},
                          "pair_by_product": {"all": float(rec_b.mean()), **group(rec_b)}},
          "own_language_share_of_top5": own, "same_language_found": same, "other_language_found": cross,
          "median_rank_of_best_other_language_relevant": int(np.median(ranks)), "still_failing": len(failed), "cases": cases}
U.save_run("d3_4_report", report)
U.save_reference("d3_4_cases", report)

# %% [markdown]
# **Look at** your three causes next to each other. None of them is "the model is bad". Each
# names a gap between how customers write and how the catalogue is written, and each
# suggests a different, small, testable fix.

# %% [markdown]
# ## ✱ Stretch · Fold the spelling variants
#
# `U.normalise_ar` folds hamza forms, ta marbuta, alef maqsura, tatweel and the short-vowel
# marks. Vectors for the normalised queries and descriptions are supplied: compute
# `U.embed([U.normalise_ar(q["text"]) for q in queries])` and
# `U.embed([U.normalise_ar(d["text"]) for d in docs], prefix="passage: ")` and measure recall
# again. Which queries moved? Did any get worse?

# %% [markdown]
# ## Check yourself
#
# 1. Recall at 5 was capped at 0.5 before fix B. Why?
# 2. Fix A gave the other language two of five seats and recall did not rise. What does
#    that tell you about the model, beyond the wall?
# 3. Rayan asks for "the recall for Arabic customers" as one number for a slide. What do you
#    tell him about the 20 queries behind it?
#
# <details><summary>Answers</summary>
#
# 1. Half of the relevant descriptions are in the other language, and the search returned
#    the query's own language only.
# 2. Its cross-language matching is too coarse for this catalogue: it finds the right kind
#    of product in the other language, not the right pattern and colour among its best two.
# 3. That 20 queries rank groups but do not measure them: one query is five points. And
#    that the queries were written by the team, not sampled from real customers' searches.
# </details>
#
# ## Carry forward
#
# Bilingual search is one of the three capstone options. Its README must report recall at 5
# for Arabic, English and mixed queries, a simpler alternative, and failure cases like yours.
