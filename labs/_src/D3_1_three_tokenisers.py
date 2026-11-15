# %% [markdown]
# # Lab 3.1 · Three tokenisers
#
# Day 3 · Session 1 (Tokens, including Arabic) · 25 min.
#
# Every product of the cooperative is described twice, in English and in Arabic. A language
# model reads neither: it reads tokens. You cut the same product sentences with three
# tokenisers and count what comes out, because the count decides how much fits in a
# model's window, how long it takes and what it costs.
#
# **Tasks**
# - [ ] One sentence pair, three tokenisers · 5 min
# - [ ] Token counts for twelve pairs · 6 min
# - [ ] Tokens per word, Arabic against English · 5 min
# - [ ] Diacritics, Arabizi and digits · 5 min
# - [ ] What the count changes · 4 min
# - [ ] ✱ Stretch · Measure what tatweel and spelling variants do to the count
#
# AI policy for this lab: an assistant may explain and hint, one step at a time.
# Only vocabulary files are fetched (a few megabytes each), never a model.

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

import nawl as U

NAMES = {"gpt2": "GPT-2 (byte-level BPE, English text, 2019)",
         "e5": "multilingual-e5 (SentencePiece, 100 languages)",
         "qwen": "Qwen2.5 (byte-level BPE, multilingual, 2024)"}
tok = {name: U.tokeniser(name) for name in NAMES}
pairs = U.pairs(12)
for name, t in tok.items():
    print(f"{name:5s} vocabulary of {t.get_vocab_size():,} tokens   {NAMES[name]}")
print("\n", pairs[0]["en"], "\n", pairs[0]["ar"])

# %% [markdown]
# ## 1 · One sentence pair, three tokenisers
#
# ✏️ **Exercise** · Return the pieces a tokeniser cuts `text` into, as a list of strings: encode
# with `t.encode(text, add_special_tokens=False)`, then decode each id by itself with
# `t.decode([i])`.

# %%
def pieces(t, text):
    # BEGIN SOLUTION
    ids = t.encode(text, add_special_tokens=False).ids
    return [t.decode([i]) for i in ids]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    pieces(tok["qwen"], "rug")
except NotImplementedError:
    pieces = lambda t, text: [t.decode([i]) for i in t.encode(text, add_special_tokens=False).ids]
short_en, short_ar = "A red striped rug", "سجادة مخططة حمراء"
for name, t in tok.items():
    print(f"{name:5s} EN {len(pieces(t, short_en)):2d} {pieces(t, short_en)}")
    print(f"{'':5s} AR {len(pieces(t, short_ar)):2d} {pieces(t, short_ar)}")
assert "".join(pieces(tok["qwen"], short_en)) == short_en

# %% [markdown]
# **Look at** the Arabic pieces of GPT-2: most print as `�`. Its vocabulary was built from
# English text, so it falls back to single bytes, and an Arabic letter is two bytes. Half a
# letter is not a character, which is why it cannot be shown.

# %% [markdown]
# ## 2 · Token counts for twelve pairs
#
# ✏️ **Exercise** · Return a table as a list of rows, one per product pair:
# `{"product": ..., "words_en": ..., "words_ar": ..., "gpt2_en": ..., "gpt2_ar": ..., "e5_en": ..., ...}`
# with the number of whitespace-separated words and the number of tokens per tokeniser.

# %%
def count_table(pairs):
    # BEGIN SOLUTION
    rows = []
    for p in pairs:
        row = {"product": p["product"], "words_en": len(p["en"].split()), "words_ar": len(p["ar"].split())}
        for name, t in tok.items():
            for lang in ("en", "ar"):
                row[f"{name}_{lang}"] = len(t.encode(p[lang], add_special_tokens=False).ids)
        rows.append(row)
    return rows
    # END SOLUTION

# %%
def _count_table(pairs):
    rows = []
    for p in pairs:
        row = {"product": p["product"], "words_en": len(p["en"].split()), "words_ar": len(p["ar"].split())}
        for name, t in tok.items():
            for lang in ("en", "ar"):
                row[f"{name}_{lang}"] = len(t.encode(p[lang], add_special_tokens=False).ids)
        rows.append(row)
    return rows

try:
    table = count_table(pairs)
except NotImplementedError:
    table = _count_table(pairs)
assert table == _count_table(pairs), "the counts differ from the reference"
cols = ["words_en", "words_ar"] + [f"{n}_{lang}" for n in tok for lang in ("en", "ar")]
print(f"{'product':8s} " + " ".join(f"{c:>8s}" for c in cols))
for r in table:
    print(f"{r['product']:8s} " + " ".join(f"{r[c]:8d}" for c in cols))
totals = {c: sum(r[c] for r in table) for c in cols}
print(f"{'total':8s} " + " ".join(f"{totals[c]:8d}" for c in cols))

# %% [markdown]
# ## 3 · Tokens per word, Arabic against English
#
# The two descriptions of a product say the same thing. Compare what each language costs.
#
# ✏️ **Exercise** · From `totals`, return for one tokeniser `name` the tuple
# `(tokens per English word, tokens per Arabic word, Arabic tokens / English tokens)`.

# %%
def ratios(totals, name):
    # BEGIN SOLUTION
    en = totals[f"{name}_en"] / totals["words_en"]
    ar = totals[f"{name}_ar"] / totals["words_ar"]
    return en, ar, totals[f"{name}_ar"] / totals[f"{name}_en"]
    # END SOLUTION

# %%
try:
    ratios(totals, "qwen")
except NotImplementedError:
    ratios = lambda totals, name: (totals[f"{name}_en"] / totals["words_en"], totals[f"{name}_ar"] / totals["words_ar"],
                                   totals[f"{name}_ar"] / totals[f"{name}_en"])
summary = {}
print(f"{'tokeniser':10s} {'tokens/word EN':>15s} {'tokens/word AR':>15s} {'AR tokens / EN tokens':>22s}")
for name in tok:
    en, ar, rel = ratios(totals, name)
    summary[name] = {"tokens_per_word_en": round(en, 2), "tokens_per_word_ar": round(ar, 2), "ar_over_en": round(rel, 2)}
    print(f"{name:10s} {en:15.2f} {ar:15.2f} {rel:22.2f}")
assert summary["gpt2"]["ar_over_en"] > summary["e5"]["ar_over_en"], "the English-only tokeniser must pay more for Arabic"
U.save_reference("d3_1_counts", {"totals": totals, "summary": summary})

# %% [markdown]
# **Look at** the last column. The same product, said in Arabic, costs a multilingual
# tokeniser about as many tokens as in English or somewhat more, and an English-built one
# several times more. A tokeniser is part of the model: you cannot swap it afterwards.

# %% [markdown]
# ## 4 · Diacritics, Arabizi and digits
#
# Customers do not write like the catalogue.

# %%
variants = {
    "catalogue Arabic": "سجادة مخططة حمراء",
    "with short vowels": "سَجَّادَة مُخَطَّطَة حَمْرَاء",
    "Arabizi": "sajjada mukhattata 7amra",
    "English": "red striped rug",
    "a size": "120 x 180 cm",
    "Arabic-Indic digits": "١٢٠ × ١٨٠ سم",
}
variant_counts = {}
print(f"{'text':22s} " + " ".join(f"{n:>6s}" for n in tok))
for label, text in variants.items():
    variant_counts[label] = {n: len(t.encode(text, add_special_tokens=False).ids) for n, t in tok.items()}
    print(f"{label:22s} " + " ".join(f"{variant_counts[label][n]:6d}" for n in tok))
print("\nQwen pieces of the vowelled text:", pieces(tok["qwen"], variants["with short vowels"]))
print("Qwen pieces of the Arabizi text: ", pieces(tok["qwen"], variants["Arabizi"]))
assert variant_counts["with short vowels"]["qwen"] > variant_counts["catalogue Arabic"]["qwen"]
U.save_reference("d3_1_variants", variant_counts)

# %% [markdown]
# **Look at** the vowelled row: each short-vowel mark is a character of its own and breaks
# the words into more pieces. The same three words cost two to three times as much, and a
# model has seen far less text written that way.

# %% [markdown]
# ## 5 · What the count changes
#
# Wasf, the description drafter of Day 4, runs with a context window of `U.N_CTX` tokens.
#
# ✏️ **Exercise** · Return how many whole descriptions of the given language fit in a window
# of `window` tokens with the Qwen tokeniser: use the mean number of tokens per description
# over the twelve pairs.

# %%
def descriptions_that_fit(lang, window=U.N_CTX):
    # BEGIN SOLUTION
    mean_tokens = totals[f"qwen_{lang}"] / len(table)
    return int(window // mean_tokens)
    # END SOLUTION

# %%
try:
    fit = {lang: descriptions_that_fit(lang) for lang in ("en", "ar")}
except NotImplementedError:
    fit = {lang: int(U.N_CTX // (totals[f"qwen_{lang}"] / len(table))) for lang in ("en", "ar")}
print(f"a window of {U.N_CTX} tokens holds about {fit['en']} English descriptions or {fit['ar']} Arabic ones")
assert fit["en"] > 0 and fit["ar"] > 0
U.save_reference("d3_1_fit", fit)

# %% [markdown]
# **Look at** the two numbers and think of three bills that are counted in tokens: the
# window (how much the model can read at once), the time (a model produces one token per
# step) and, for a hosted model, the price.

# %% [markdown]
# ## ✱ Stretch · Tatweel and spelling variants
#
# Count the tokens of `"سجادة"`, `"سجاده"` (ta marbuta written as ha), `"سجـــادة"` (with
# tatweel) and of `U.normalise_ar` applied to each. Which variants would a search treat as
# different words unless the text is normalised first?

# %% [markdown]
# ## Check yourself
#
# 1. Why does GPT-2's tokeniser need about two tokens per Arabic letter?
# 2. The multilingual tokenisers still need more tokens per Arabic word than per English
#    word. Give one reason that has nothing to do with the tokeniser's quality.
# 3. A customer writes with full short vowels. Name two consequences.
#
# <details><summary>Answers</summary>
#
# 1. Its vocabulary was learnt from English text and has no Arabic pieces, so it falls back
#    to bytes; an Arabic letter is two bytes in UTF-8.
# 2. An Arabic word carries more: the article, prepositions and pronouns attach to it
#    ("باللون" is "in the colour"), so one word is several units of meaning.
# 3. The text costs more tokens, and it is cut into pieces the model has rarely seen, so
#    quality can drop. Normalising (removing the marks) before the model is a common step.
# </details>
#
# ## Carry forward
#
# The Qwen tokeniser is the one Wasf uses on Day 4; the e5 tokeniser is the one behind the
# search of Labs 3.3 and 3.4.
