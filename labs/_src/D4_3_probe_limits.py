# %% [markdown]
# # Lab 4.3 · Probe the limits
#
# Day 4 · Session 3 (Limits: fabrication, cut-off and context) · 25 min.
#
# Before Wasf writes anything a customer reads, Bushra Al-Aqeel wants its limits on paper.
# You probe the small model in three places: questions it cannot know, questions beyond
# its training cut-off, and inputs longer than its context window, in English and in
# Arabic. You record what it does, and you write the limits note that travels with its
# output.
#
# **Tasks**
# - [ ] Questions it cannot know · 7 min
# - [ ] Beyond the cut-off · 4 min
# - [ ] Beyond the context window · 7 min
# - [ ] Arabic against English · 4 min
# - [ ] The limits note · 3 min
# - [ ] ✱ Stretch · Put the missing fact in the prompt and measure again
#
# AI policy for this lab: an assistant may explain and hint, one step at a time. The model
# under test is the object you measure, not a source of answers.
# The lab reads supplied logged runs of the model (`data/runs/d4_3.jsonl`); with a live
# model (Colab, or a free hosted endpoint) the last cell adds a run of your own.

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

import re
import nawl as U
import llm_tasks as T

rows = U.runs("d4_3")
print(f"{len(rows)} logged runs of {rows[0]['model']}; {rows[0]['source']}")
print(f"the runtime's context window: {U.N_CTX} tokens; live model here: {U.llm_backend()}")

# %% [markdown]
# ## 1 · Questions it cannot know
#
# Al-Nawl Crafts Cooperative does not exist: no model has read anything about it. Eight
# questions ask for facts about it, six in English and two in Arabic. A model can do two
# things: say that it does not know, or produce an answer anyway.
#
# Read the eight answers first.

# %%
unanswerable = [r for r in rows if r["setting"] == "unanswerable"]
for r in unanswerable:
    print(f"{r['case']}  Q: {r['prompt']}\n     A: {r['text'][:230]}\n")

# %% [markdown]
# ✏️ **Exercise** · Write a rule that returns `True` when an answer says it does not know:
# look for phrases such as "I don't have", "I'm sorry", "unable", and their Arabic
# counterparts ("عذر", "ليس لدي", "لا يمكنني"). Lower-case the text first.

# %%
def says_it_does_not_know(text):
    # BEGIN SOLUTION
    phrases = ["i don't have", "i do not have", "i'm sorry", "i am sorry", "unable", "i can't", "i cannot",
               "عذر", "ليس لدي", "لا يمكنني", "لا أملك"]
    low = text.lower()
    return any(p in low for p in phrases)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    says_it_does_not_know("x")
except NotImplementedError:
    says_it_does_not_know = T.abstains
labels = {r["case"]: says_it_does_not_know(r["text"]) for r in unanswerable}
fabricated = [r for r in unanswerable if not labels[r["case"]]]
print(f"{sum(labels.values())} of {len(unanswerable)} answers say they do not know; {len(fabricated)} answer anyway:\n")
for r in fabricated:
    print(f"{r['case']}  Q: {r['prompt']}\n     A: {r['text'][:200]}\n")
assert labels == {r["case"]: T.abstains(r["text"]) for r in unanswerable}, "your rule disagrees with a reading of the eight answers"

# %% [markdown]
# **Look at** the answers that came anyway. They are fluent, specific and false: a founding
# year, founders, a country. The same question, asked in Arabic, produced a different false
# story with the same year. Nothing in the text marks it as invented. And look at your rule:
# it counts phrases. A polite refusal followed by an invented "general" answer passes it.

# %% [markdown]
# ## 2 · Beyond the cut-off
#
# A model's knowledge ends where its training text ends. Four questions ask about dates and
# about events after that point.
#
# ✏️ **Exercise** · Return every four-digit year (1900 to 2099) that appears in a text, as a
# list of integers.

# %%
def years_in(text):
    # BEGIN SOLUTION
    return [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", text)]
    # END SOLUTION

# %%
try:
    years_in("in 2023")
except NotImplementedError:
    years_in = lambda text: [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", text)]
assert years_in("released in 2019, updated 2023.") == [2019, 2023]
cutoff = [r for r in rows if r["setting"] == "cutoff"]
claimed = []
for r in cutoff:
    if "last update" in r["text"]:
        claimed += years_in(r["text"])
    print(f"{r['case']}  Q: {r['prompt']}\n     A: {r['text'][:210]}\n     years named: {years_in(r['text'])}\n")
print("the model says its knowledge ends in:", sorted(set(claimed)), "| its weights were published in September 2024 (model card)")

# %% [markdown]
# **Look at** three things. It does not know today's date and says so. It treats a
# tournament of 2026 as the future, because for it the present is where its text ended. And
# the year it gives for its own last update is itself generated text: a year earlier than
# the release of its weights. Its statement about its cut-off is a guess like any other.

# %% [markdown]
# ## 3 · Beyond the context window
#
# The shop policy is a short document. Four questions are answered from it in three
# arrangements: the policy and the question (`fits`); the policy, then a long catalogue
# extract, then the question (`policy first, too long`); and the extract, then the policy,
# then the question (`policy last, too long`). The last two are longer than the window of
# 2,048 tokens. A local runtime does not refuse such a prompt: it keeps the end and drops
# the start, silently.
#
# ✏️ **Exercise** · Return `{setting: share of right answers}` over the policy runs. An answer
# is right when `T.has_number(text, expect)` is true, with `expect` from `T.POLICY_QUESTIONS`.

# %%
expect = {q["id"]: q["expect"] for q in T.POLICY_QUESTIONS}
policy = [r for r in rows if r["case"] in expect]


def share_right(policy):
    # BEGIN SOLUTION
    out = {}
    for setting in dict.fromkeys(r["setting"] for r in policy):
        group = [r for r in policy if r["setting"] == setting]
        out[setting] = sum(T.has_number(r["text"], expect[r["case"]]) for r in group) / len(group)
    return out
    # END SOLUTION

# %%
def _share_right(policy):
    out = {}
    for setting in dict.fromkeys(r["setting"] for r in policy):
        group = [r for r in policy if r["setting"] == setting]
        out[setting] = sum(T.has_number(r["text"], expect[r["case"]]) for r in group) / len(group)
    return out

try:
    shares = share_right(policy)
except NotImplementedError:
    shares = _share_right(policy)
assert shares == _share_right(policy)
print(f"{'arrangement':24s} {'prompt tokens':>13s} {'tokens dropped':>15s} {'seconds':>8s}   answers (expected {[expect[k] for k in expect]})")
for setting in shares:
    group = [r for r in policy if r["setting"] == setting]
    answers = [r["text"].replace("\n", " ")[:18] for r in group]
    print(f"{setting:24s} {group[0]['prompt_tokens']:13d} {group[0]['truncated']:15d} {sum(r['seconds'] for r in group) / len(group):8.1f}   {answers}")
print("\nshare right:", {k: round(v, 2) for k, v in shares.items()})
assert shares["fits"] == 1.0 and shares["policy first, too long"] < shares["policy last, too long"]

# %% [markdown]
# **Look at** the middle row. About 1,400 tokens were dropped from the start, the policy
# among them. The model was not told. It answered every question with a confident number,
# and every number is wrong. Same model, same question, same document: what changed is
# where the document sat in a prompt that did not fit.

# %% [markdown]
# ## 4 · Arabic against English
#
# The Arabic policy and the Arabic questions say the same as the English ones, and both
# fit in the window.

# %%
for q in T.POLICY_QUESTIONS:
    en = next(r for r in policy if r["case"] == q["id"] and r["setting"] == "fits")
    ar = next(r for r in policy if r["case"] == q["id"] and r["setting"] == "fits, arabic")
    print(f"{q['id']} expected {q['expect']:>3s} | English: {en['text'][:20]!r:24s} {'right' if T.has_number(en['text'], q['expect']) else 'WRONG':6s}"
          f"| Arabic: {ar['text'][:24]!r:28s} {'right' if T.has_number(ar['text'], q['expect']) else 'WRONG'}")
print(f"\ngrounded answers right: English {shares['fits']:.2f}, Arabic {shares['fits, arabic']:.2f}")
ar_unknown = [r for r in unanswerable if r["case"] in ("U07", "U08")]
print("Arabic questions it cannot know:", {r["case"]: "says it does not know" if labels[r["case"]] else "answers anyway" for r in ar_unknown})

# %% [markdown]
# **Look at** the Arabic miss: the model picked another number from the same paragraph of the
# policy. Four questions are far too few to put a figure on the gap between the languages.
# They are enough to know that the Arabic output must be measured by itself, on more
# questions, before a customer reads it.

# %% [markdown]
# ## 5 · The limits note
#
# ✏️ **Exercise** · Write the note that goes with Wasf's output before it may be shown to a
# customer: at least three sentences, each naming one limit you measured in this lab and what
# is done about it. Return them as a list of strings.

# %%
def limits_note():
    # BEGIN SOLUTION
    return [
        "This text was drafted by a small language model and is shown as a draft: a member of the shop team reads it before it is published.",
        "The model invents facts it does not have (it gave the cooperative a founding year and founders): it is given the product's attributes and may state nothing else.",
        "Its knowledge ends before its release and it does not know today's date: it is never asked about dates, prices, stock or delivery.",
        "A prompt longer than its window loses its beginning without warning: prompts are counted in tokens before they are sent, and one that does not fit is refused.",
        "Its Arabic answers were wrong more often than its English ones in a small test: Arabic output is checked on its own test set before use.",
    ]
    # END SOLUTION

# %%
try:
    note = limits_note()
except NotImplementedError:
    note = U.reference("d4_3_report")["limits_note"]
assert len(note) >= 3 and all(len(s) > 40 for s in note), "at least three full sentences"
for s in note:
    print("-", s)
report = {"model": rows[0]["model"], "cannot_know": {"questions": len(unanswerable), "said_it_does_not_know": sum(labels.values()),
                                                     "answered_anyway": [r["case"] for r in fabricated]},
          "years_claimed": sorted(set(claimed)), "context": shares, "limits_note": note}
U.save_run("d4_3_report", report)
U.save_reference("d4_3_report", report)

# %%
# A probe of your own (runs only with a live model)
mine = U.try_live(lambda: U.generate("Who is the director of Al-Nawl Crafts Cooperative, and since when?", max_new_tokens=60), "your own probe")
if mine:
    print(mine[0]["text"])

# %% [markdown]
# ## ✱ Stretch · Put the fact in the prompt
#
# With a live model: ask question U01 again with `T.POLICY_EN` placed before it and the line
# "Answer from the document above." Does the fabricated or refused answer become a right
# one? This is the idea behind grounded answers, which AE-203 builds and evaluates.

# %% [markdown]
# ## Check yourself
#
# 1. The model fabricated two of eight answers and refused six. Is "it refuses 75% of
#    unanswerable questions" a safe sentence for a report?
# 2. Why did `policy last, too long` still give right answers?
# 3. A hosted endpoint returns an error for a prompt that is too long; the local runtime
#    here did not. Which behaviour would you rather have, and why?
#
# <details><summary>Answers</summary>
#
# 1. No. Eight questions, one wording each, one model setting: it shows that fabrication
#    happens, not how often. And the rule that counted refusals looks for phrases only.
# 2. The runtime keeps the end of a prompt that does not fit. The policy sat near the end,
#    inside the part that was kept.
# 3. The error. A silent cut gives a confident wrong answer that nobody can trace; an error
#    is a fault you see. With a local runtime, count the tokens yourself before sending.
# </details>
#
# ## Carry forward
#
# Your limits note is in `runs/d4_3_report.json`. The capstone rubric gives 15 points to
# licence and limits: this is what the limits half looks like.
