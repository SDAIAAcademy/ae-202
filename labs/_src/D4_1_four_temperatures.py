# %% [markdown]
# # Lab 4.1 · Four temperatures
#
# Day 4 · Session 1 (Next-token prediction and sampling) · 25 min.
#
# Rayan Al-Sudais wants Wasf to draft one line for each of the shop's 2,000 products. Before
# anyone picks a setting, you measure what the settings do. You apply temperature and
# nucleus sampling by hand to one real next-token distribution of the small model, then you
# measure variation, length and rule violations over 800 logged drafts: 20 products, five
# samples each, eight settings.
#
# **Tasks**
# - [ ] Temperature on a real distribution · 5 min
# - [ ] Nucleus sampling by hand · 4 min
# - [ ] Variation across five samples · 6 min
# - [ ] Length, rule violations and one surprise · 6 min
# - [ ] A live run of your own · 4 min
# - [ ] ✱ Stretch · Draw 1,000 next tokens from your own distribution and compare the frequencies
#
# AI policy for this lab: an assistant may explain and hint, one step at a time. The model
# under test is the object you measure, not a source of answers.
# The lab reads supplied logged runs of the model (`data/runs/d4_1.jsonl`); with a live
# model (Colab, or a free hosted endpoint) the last task adds a run of your own.

# %%
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

import json
import numpy as np
import nawl as U
import llm_tasks as T

rows = U.runs("d4_1")
nxt = json.loads((U.DATA / "runs" / "d4_1_next_token.json").read_text(encoding="utf-8"))
logits = np.array(nxt["logits"])
tokens = nxt["tokens"]
print(f"{len(rows)} logged drafts of {rows[0]['model']}; {rows[0]['source']}")
print(f"next-token scores: the {len(tokens)} highest of {nxt['vocabulary_size']:,} tokens "
      f"(they hold {nxt['mass_of_these_at_temperature_1']:.4f} of the probability at temperature 1)")
print("the prompt:", nxt["prompt"].replace("\n", " | "))

# %% [markdown]
# ## 1 · Temperature on a real distribution
#
# The model does not write a sentence. It gives every token of its vocabulary a score (a
# logit) for the next position; a token is drawn; the model is run again. `logits` holds
# the scores for the first token of the product line above.
#
# ✏️ **Exercise** · Turn scores into probabilities at temperature `t`: divide the scores by `t`,
# then apply the softmax. Subtract the largest score first so that the exponential stays small.

# %%
def probabilities(logits, t):
    # BEGIN SOLUTION
    z = (logits - logits.max()) / t
    e = np.exp(z)
    return e / e.sum()
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _probabilities(logits, t):
    e = np.exp((logits - logits.max()) / t)
    return e / e.sum()

try:
    probabilities(logits, 1.0)
except NotImplementedError:
    probabilities = _probabilities
assert np.allclose(probabilities(logits, 0.7), _probabilities(logits, 0.7)), "divide by t, then softmax"
assert abs(probabilities(logits, 1.5).sum() - 1) < 1e-9


def needed_for(p, share=0.9):
    """How many of the most likely tokens are needed to hold `share` of the probability."""
    return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), share) + 1)


print(f"{'temperature':>11s}  {'first':>7s} {'second':>7s} {'third':>7s}   tokens holding 90%")
for t in (0.2, 0.7, 1.0, 1.5):
    p = probabilities(logits, t)
    print(f"{t:11.1f}  {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}   {needed_for(p):4d}")
print("the three most likely tokens:", [tokens[i] for i in range(3)])

# %% [markdown]
# **Look at** the last column. The order of the tokens never changes: temperature changes
# how much of the probability the first ones hold. Close to 0 the first token takes
# everything, which is why "temperature 0" means "always take the most likely token". Above
# 1 the tail gains: tokens the model scored as unlikely are drawn.

# %% [markdown]
# ## 2 · Nucleus sampling by hand
#
# Nucleus sampling (top-p) cuts the tail: keep the most likely tokens until their
# probabilities add up to `top_p`, drop the rest, and share the probability again among
# those kept.
#
# ✏️ **Exercise** · `p` is sorted from the most to the least likely token. Return an array of
# the same length: the kept tokens renormalised to sum to 1, zeros for the dropped ones. The
# token that crosses `top_p` is kept.

# %%
def nucleus(p, top_p):
    # BEGIN SOLUTION
    keep = np.cumsum(p) - p < top_p
    q = np.where(keep, p, 0.0)
    return q / q.sum()
    # END SOLUTION

# %%
def _nucleus(p, top_p):
    q = np.where(np.cumsum(p) - p < top_p, p, 0.0)
    return q / q.sum()

try:
    nucleus(probabilities(logits, 1.0), 0.9)
except NotImplementedError:
    nucleus = _nucleus
assert np.allclose(nucleus(np.array([0.5, 0.3, 0.15, 0.05]), 0.7), [0.625, 0.375, 0, 0]), "keep the token that crosses top_p"
p15 = probabilities(logits, 1.5)
for top_p in (1.0, 0.9, 0.5):
    q = nucleus(p15, top_p)
    kept = int((q > 0).sum())
    print(f"temperature 1.5, top_p {top_p:3.1f}: {kept:3d} tokens can be drawn; the first has {q[0]:.3f}; the last kept: {tokens[kept - 1]!r}")

# %% [markdown]
# **Look at** the number of tokens that can still be drawn. Temperature 1.5 opened the tail;
# the nucleus closes it again. The two settings pull in opposite directions, and a request
# always carries both.

# %% [markdown]
# ## 3 · Variation across five samples
#
# Each of the 20 products was drafted five times at each setting. One number for variation:
# how many of the five drafts are different texts.
#
# ✏️ **Exercise** · Return the number of distinct texts in a list of drafts.

# %%
def distinct(texts):
    # BEGIN SOLUTION
    return len(set(texts))
    # END SOLUTION

# %%
try:
    distinct(["a", "a"])
except NotImplementedError:
    distinct = lambda texts: len(set(texts))
assert distinct(["a", "b", "a"]) == 2
settings = list(T.SETTINGS)
cases = list(dict.fromkeys(r["case"] for r in rows))


def drafts(setting, case):
    return [r["text"] for r in rows if r["setting"] == setting and r["case"] == case]


def shared_words(texts):
    """How alike the drafts of one product are: the mean share of words that two drafts have in common."""
    sets = [set(t.lower().split()) for t in texts]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in pairs]))


variation = {s: float(np.mean([distinct(drafts(s, c)) for c in cases])) for s in settings}
alike = {s: float(np.mean([shared_words(drafts(s, c)) for c in cases])) for s in settings}
for s in settings[:4]:
    print(f"{s:14s} distinct drafts of five: {variation[s]:.2f}   words shared by two drafts: {alike[s]:.2f}   (means over {len(cases)} products)")
print()
for s in ("T=0", "T=0.7", "T=1.5"):
    print(s)
    for text in drafts(s, cases[0])[:3]:
        print("   ", text.replace("\n", " / ")[:150])

# %% [markdown]
# **Look at** the two columns. At temperature 0 the five drafts are one draft: the same
# prompt gives the same text. Above 0 the count is 5 of 5 everywhere: a measure that is at
# its ceiling tells you nothing, so a second, finer one is needed. The share of words that
# two drafts have in common falls from 0.7 to 1.5: the drafts differ in wording first, then
# in kind. Read the three drafts at 1.5: a wide tail is not "creative", it is noise.

# %% [markdown]
# ## 4 · Length, rule violations and one surprise
#
# The task has four rules: one sentence, at most 20 words, English only, no price.
# `T.violations(text)` checks them by code and returns `{rule: True when broken}`.
#
# ✏️ **Exercise** · For one setting, return a dictionary with `words` (the mean number of words
# in a draft), one entry per rule (the share of drafts that break it), and `any_violation`
# (the share of drafts that break at least one rule).

# %%
def measure(setting):
    texts = [r["text"] for r in rows if r["setting"] == setting]
    # BEGIN SOLUTION
    checks = [T.violations(t) for t in texts]
    out = {"words": float(np.mean([len(t.split()) for t in texts]))}
    for rule in checks[0]:
        out[rule] = float(np.mean([c[rule] for c in checks]))
    out["any_violation"] = float(np.mean([any(c.values()) for c in checks]))
    return out
    # END SOLUTION

# %%
def _measure(setting):
    texts = [r["text"] for r in rows if r["setting"] == setting]
    checks = [T.violations(t) for t in texts]
    out = {"words": float(np.mean([len(t.split()) for t in texts]))}
    for rule in checks[0]:
        out[rule] = float(np.mean([c[rule] for c in checks]))
    out["any_violation"] = float(np.mean([any(c.values()) for c in checks]))
    return out

try:
    measure(settings[0])
except NotImplementedError:
    measure = _measure
table = {s: {**measure(s), "distinct_of_5": variation[s], "shared_words": alike[s]} for s in settings}
assert all(abs(table[s]["any_violation"] - _measure(s)["any_violation"]) < 1e-9 for s in settings), "any_violation: at least one rule broken"
rules = [k for k in table[settings[0]] if k not in ("words", "any_violation", "distinct_of_5", "shared_words")]
short = {"more than one sentence": ">1 sentence", "more than 20 words": ">20 words", "not English only": "not English", "mentions a price": "price"}
print(f"{'setting':14s} {'shared words':>12s} {'words':>6s} " + " ".join(f"{short[r]:>12s}" for r in rules) + f" {'any rule':>9s}")
for s in settings:
    m = table[s]
    print(f"{s:14s} {m['shared_words']:12.2f} {m['words']:6.1f} " + " ".join(f"{m[r]:12.2f}" for r in rules) + f" {m['any_violation']:9.2f}")
U.save_run("d4_1_table", {"table": table})
U.save_reference("d4_1_table", {"table": table})

# %% [markdown]
# **Look at** four rows, and write down which one surprised you.
#
# - `T=0` breaks a rule in a quarter of its drafts: every one of them is too long. A fixed
#   output is repeatable; it is not thereby right.
# - `T=1.5 p=0.9` is hardly better than `T=1.5`. In task 2 you saw why: at this temperature
#   the nucleus of 0.9 still holds more than a hundred tokens. Only `p=0.5` cuts deep enough.
# - `T=0.7 strict`: the strict instruction takes the violations from 0.44 to 0.05 and halves
#   the length. The instruction did more than any sampling setting.
# - `T=1.5 strict` is as bad as `T=1.5`. An instruction cannot repair a distribution that
#   the temperature has flattened: the rules are in the prompt, the noise is in the draw.
#
# And one thing the table does not show. Read the second draft at 0.7 above: "35 inches".
# The product is 35 cm wide. The four checks are rules that a program can test; whether a
# draft is true to the product is not among them.

# %%
# The drafts behind one cell of the table: change the setting and the rule, and read
worst = [r for r in rows if r["setting"] == "T=1.5" and T.violations(r["text"])["not English only"]]
print(len(worst), "drafts at T=1.5 are not English only; three of them:")
for r in worst[:3]:
    print("   ", r["text"].replace("\n", " / ")[:170])

# %% [markdown]
# ## 5 · A live run of your own
#
# With a live model, draft one product five times at a temperature of your choice and put
# your row next to the table. Without one, the cell says so and the lab is complete.

# %%
MY_TEMPERATURE, MY_TOP_P = 1.0, 0.9
product = T.products()[3]
mine = U.try_live(lambda: U.generate(T.description_messages(product), temperature=MY_TEMPERATURE, top_p=MY_TOP_P,
                                     max_new_tokens=48, n=U.pick(5, 2), seed=7), "your own five drafts")
if mine:
    texts = [m["text"] for m in mine]
    for text in texts:
        print("-", text.replace("\n", " / ")[:150])
    print(f"distinct: {distinct(texts)} of {len(texts)}; drafts breaking a rule: {sum(any(T.violations(t).values()) for t in texts)}")

# %% [markdown]
# ## ✱ Stretch · Draw 1,000 tokens
#
# Draw 1,000 next tokens from `nucleus(probabilities(logits, 1.5), 0.9)` with
# `np.random.default_rng(0).choice(len(tokens), size=1000, p=...)` and compare the
# frequencies of the five most likely tokens with their probabilities. How far apart are they
# after 1,000 draws, and after 50?

# %% [markdown]
# ## Check yourself
#
# 1. A colleague sets temperature 0 "so that the model is always right". What does
#    temperature 0 give, and what does it not give?
# 2. At temperature 1.5 and top_p 0.5, is the output closer to temperature 1.5 or to
#    temperature 0.7? Use your table.
# 3. Each cell of the table rests on 100 drafts of 20 products. Name one thing the table
#    cannot tell you.
#
# <details><summary>Answers</summary>
#
# 1. The same text for the same prompt, which makes a result repeatable. It does not make the
#    text right: the most likely token can be wrong, and the drafts at temperature 0 break
#    rules too.
# 2. Closer to 0.7: the nucleus removes most of the tail that temperature 1.5 opened, and
#    the violations fall from 0.93 to 0.56, near the 0.44 of temperature 0.7.
# 3. Whether a draft is true to the product, or good. The four checks are rules a program can
#    test; nobody has read the drafts. And 20 products in English say nothing about Arabic.
# </details>
#
# ## Carry forward
#
# Your table is in `runs/d4_1_table.json`. Lab 4.4 reads the share of clean drafts from it
# when Wasf is compared with a template and with larger models.
