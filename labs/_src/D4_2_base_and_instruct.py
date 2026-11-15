# %% [markdown]
# # Lab 4.2 · Base against instruction-tuned
#
# Day 4 · Session 2 (From base models to assistants) · 25 min.
#
# Wasf's model is published twice: as a base model, trained only to continue text, and as
# an instruction-tuned model, trained further on conversations. Zaid Al-Luhaidan downloaded
# the wrong one last week and concluded that "the model is broken". You give both the same
# ten small tasks, score them by code, and then make the base model do the work with a
# few-shot prompt.
#
# **Tasks**
# - [ ] What each model receives · 5 min
# - [ ] Ten tasks, scored by code · 8 min
# - [ ] A few-shot prompt for the base model · 7 min
# - [ ] Where the two still differ · 5 min
# - [ ] ✱ Stretch · Write the few-shot prompt for an eleventh task
#
# AI policy for this lab: an assistant may explain and hint, one step at a time. The models
# under test are the objects you measure, not a source of answers.
# The lab reads supplied logged runs of both models (`data/runs/d4_2.jsonl`), all at
# temperature 0 with at most 40 new tokens; with a live model the last cell adds your own.

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

import nawl as U
import llm_tasks as T

rows = U.runs("d4_2")
tasks = {t["id"]: t for t in T.TASKS}
print(f"{len(rows)} logged runs: {len(tasks)} tasks, settings {list(dict.fromkeys(r['setting'] for r in rows))}")
print("models:", sorted(set(r["model"] for r in rows)))


def row(case, setting):
    return next(r for r in rows if r["case"] == case and r["setting"] == setting)

# %% [markdown]
# ## 1 · What each model receives
#
# You type the same prompt for both models. They do not receive the same text. Each logged
# run keeps `prompt` (what was typed) and `rendered` (the text that went into the model).
#
# ✏️ **Exercise** · Return the two pieces of text that were added around the prompt: what comes
# before it in `rendered`, and what comes after it.

# %%
def added_around(r):
    # BEGIN SOLUTION
    before, after = r["rendered"].split(r["prompt"], 1)
    return before, after
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    added_around(row("T08", "base"))
except NotImplementedError:
    added_around = lambda r: tuple(r["rendered"].split(r["prompt"], 1))
assert tuple(added_around(row("T08", "base"))) == ("", ""), "the base model receives the prompt and nothing else"
before, after = added_around(row("T08", "instruct"))
print("typed:           ", repr(row("T08", "instruct")["prompt"]))
print("added before it: ", repr(before))
print("added after it:  ", repr(after))
print()
for setting in ("instruct", "base"):
    print(f"{setting:9s} answers: {row('T08', setting)['text'][:160]!r}")

# %% [markdown]
# **Look at** what was added. The instruction-tuned model was trained on text of exactly this
# shape: a system turn, a user turn, then the opening of an assistant turn. It continues
# that text, and the continuation of "the assistant's turn starts here" is an answer. The
# base model received a bare question. A question is often followed, in ordinary text, by
# more questions.

# %% [markdown]
# ## 2 · Ten tasks, scored by code
#
# `T.task_score(task, r)` gives three checks for one run: `right` (the expected answer is in
# the first line), `stopped` (the model ended by itself before the 40-token limit) and
# `clean` (right, stopped, and one line only: usable by a program as it is).
#
# ✏️ **Exercise** · For one setting, return `{"right": share, "stopped": share, "clean": share}`
# over the ten tasks.

# %%
def score(setting):
    # BEGIN SOLUTION
    checks = [T.task_score(tasks[r["case"]], r) for r in rows if r["setting"] == setting]
    return {k: sum(c[k] for c in checks) / len(checks) for k in ("right", "stopped", "clean")}
    # END SOLUTION

# %%
def _score(setting):
    checks = [T.task_score(tasks[r["case"]], r) for r in rows if r["setting"] == setting]
    return {k: sum(c[k] for c in checks) / len(checks) for k in ("right", "stopped", "clean")}

try:
    score("instruct")
except NotImplementedError:
    score = _score
assert score("base") == _score("base")
scores = {s: score(s) for s in ("instruct", "base")}
for s, m in scores.items():
    print(f"{s:14s} right {m['right']:.1f}   stopped by itself {m['stopped']:.1f}   clean {m['clean']:.1f}")
print()
for case in ("T01", "T07"):
    print(case, "typed:", repr(tasks[case]["prompt"][:70]))
    for s in ("instruct", "base"):
        print(f"   {s:9s} {row(case, s)['text'][:150]!r}")

# %% [markdown]
# **Look at** the two rows. The instruction-tuned model gets 7 of 10 right and 7 of 10 clean.
# The base model gets 3 right and 2 clean, from the same number of weights. Read what it
# wrote: nothing at all for the colour, a garbled "WoRlG rug" for the capitals. It did not
# fail to know the answer. It did not know that an answer was wanted: nothing in a bare
# line of text says "your turn".

# %% [markdown]
# ## 3 · A few-shot prompt for the base model
#
# A base model continues text. So you write a text whose most likely continuation is the
# answer: two solved examples in a fixed layout, then the new case with the answer left open.
#
# ✏️ **Exercise** · Build that text. Each example is a pair `(input, output)`. The layout is
#
# ```
# Country: France
# Capital: Paris
#
# Country: Japan
# Capital: Tokyo
#
# Country: Saudi Arabia
# Capital:
# ```
#
# with one empty line between examples and nothing after the last colon.

# %%
def few_shot(examples, query, in_label, out_label):
    # BEGIN SOLUTION
    blocks = [f"{in_label}: {x}\n{out_label}: {y}" for x, y in examples]
    blocks.append(f"{in_label}: {query}\n{out_label}:")
    return "\n\n".join(blocks)
    # END SOLUTION

# %%
def _few_shot(examples, query, in_label, out_label):
    return "\n\n".join([f"{in_label}: {x}\n{out_label}: {y}" for x, y in examples] + [f"{in_label}: {query}\n{out_label}:"])

try:
    few_shot([("a", "b")], "c", "In", "Out")
except NotImplementedError:
    few_shot = _few_shot
mine = few_shot([("France", "Paris"), ("Japan", "Tokyo")], "Saudi Arabia", "Country", "Capital")
assert mine == tasks["T08"]["few_shot"], "compare your text with the layout above, character by character"
print(mine)
print("\nthe base model continues:", repr(row("T08", "base few-shot")["text"][:120]))
scores["base few-shot"] = score("base few-shot")
print()
for s, m in scores.items():
    print(f"{s:14s} right {m['right']:.1f}   stopped by itself {m['stopped']:.1f}   clean {m['clean']:.1f}")
U.save_run("d4_2_scores", scores)
U.save_reference("d4_2_scores", scores)

# %% [markdown]
# **Look at** the third row. With two examples in front of each case the base model is right
# on 10 of 10, more than the instruction-tuned model on its plain prompt. And it is clean on
# 0 of 10: it never stops. After "Riyadh" it writes a fourth country, and a fifth. The
# examples told it what the text looks like; nothing told it where the text ends.

# %% [markdown]
# ## 4 · Where the two still differ
#
# One row per task. Read the rows where the settings disagree.

# %%
mark = lambda c: ("right" if c["right"] else "WRONG") + ("" if c["stopped"] else ", ran on")
print(f"{'task':5s} {'expected':18s} {'instruct':18s} {'base':18s} {'base few-shot':18s}")
for case, task in tasks.items():
    cells = [mark(T.task_score(task, row(case, s))) for s in ("instruct", "base", "base few-shot")]
    print(f"{case:5s} {task['expect'][:16]:18s} " + " ".join(f"{c:18s}" for c in cells))
print()
for case in ("T04", "T10"):
    print(case, "base few-shot wrote:", repr(row(case, "base few-shot")["text"][:170]))

# %% [markdown]
# **Look at** tasks T03, T04 and T05. The instruction-tuned model was wrong on its plain
# prompt; the base model with examples was right. The examples showed what kind of answer
# is wanted (a language, a type of product, one word) better than the instruction did. And look at the last
# column as a whole: every answer is right, and every one "ran on". A program that takes the
# first line gets ten usable answers; a program that takes the whole text gets none. Ten
# tasks, one wording each, are enough to see the two behaviours. They are not enough to
# rank the two models.

# %%
# With a live model: the same task, your own wording (the instruction-tuned model answers here)
out = U.try_live(lambda: U.generate("Which material is this product made of? Answer with one word.\nProduct: A green basket woven from palm leaf.",
                                    max_new_tokens=20), "your own task")
if out:
    print(repr(out[0]["text"]))

# %% [markdown]
# ## ✱ Stretch · An eleventh task
#
# Write a few-shot prompt, with `few_shot`, that makes a base model return the size of a
# product in centimetres from its description. Which two examples do you choose so that the
# layout of the answer is fixed? With a live runtime, run it with `kind="base"`.

# %% [markdown]
# ## Check yourself
#
# 1. The base model and the instruction-tuned model have the same number of weights and the
#    same architecture. What is different?
# 2. Why does the base model keep writing after a right answer?
# 3. A hosted endpoint offers only instruction-tuned models. When would you still write
#    examples into the prompt?
#
# <details><summary>Answers</summary>
#
# 1. The values of the weights: the instruction-tuned model was trained further on
#    conversations wrapped in a chat template, and it receives that template around your
#    prompt. Nothing was added to the network.
# 2. It was trained to continue text, and after two examples and an answer the most likely
#    continuation is a further example. It has not learned to end its turn; the calling
#    program must cut the text at the first line, or set a stop sequence.
# 3. Whenever the layout of the answer matters: a fixed label, a JSON shape, a house style.
#    Examples fix the format for instruction-tuned models too.
# </details>
#
# ## Carry forward
#
# Your scores are in `runs/d4_2_scores.json`. In Lab 4.4 the two published variants of a
# model are one line of the shortlist each: this lab is why the line says which one.
