# %% [markdown]
# # Lab 4.4 · The model-choice table
#
# Day 4 · Session 4 (Choosing a model family) · 25 min.
#
# Mariam Al-Hudaithi, the director, wants one page: for each of the three use cases, which
# model, why, and what could go wrong. You have everything for it: the results you measured
# in the labs of four days, a shortlist of open-weight and hosted models with their
# licences, and the constraints of the shop. You screen the options on licence, data and
# size first, compare what remains on quality, latency and cost, and write two lines per
# use case. Then you choose your capstone task.
#
# **Tasks**
# - [ ] Read the shortlist and the measurements · 4 min
# - [ ] Licence and data constraints first · 5 min
# - [ ] Quality, size, latency and cost per use case · 8 min
# - [ ] Two lines of justification each · 5 min
# - [ ] Choose the capstone task · 3 min
# - [ ] ✱ Stretch · Change one constraint and see which choice flips
#
# AI policy for this lab: an assistant may explain and hint, one step at a time. The
# recommendation is yours: you defend it in the capstone.

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
import nawl as U

short = json.loads((U.DATA / "shortlist.json").read_text(encoding="utf-8"))
meas = json.loads((U.DATA / "measurements.json").read_text(encoding="utf-8"))
models = {m["id"]: m for m in short["models"]}
C = short["constraints"]
cases = meas["use_cases"]
print("measured on", meas["measured_on"])
print("constraints:")
for k, v in C.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 1 · Read the shortlist and the measurements
#
# Two tables. The shortlist says what each model **is**: family, size, licence, where it
# runs. The measurements say what each option **did** on the cooperative's own data, in the
# labs. An option is a model used in one way for one use case.

# %%
print(f"{'model':24s} {'family':28s} {'weights (M)':>11s} {'file (MB)':>9s}  {'commercial use':15s} {'leaves server':13s} licence")
for m in short["models"]:
    w = "" if m["weights_millions"] is None else f"{m['weights_millions']:,}"
    print(f"{m['id']:24s} {m['family']:28s} {w:>11s} {m['file_mb']:9,}  {m['commercial_use']:15s} {'yes' if m['leaves_the_server'] else 'no':13s} {m['licence'][:60]}")

# %%
show = lambda v, f: "not measured" if v is None else format(v, f)
for name, case in cases.items():
    print(f"\n{name.upper()} · {case['task']}\n  quality = {case['quality_measure']}; latency budget {C['latency_budget_ms'][name]:,} ms")
    for o in case["options"]:
        print(f"  {o['id']:26s} {o['what'][:56]:56s} quality {show(o['quality'], '.3f'):>12s}   latency {show(o['latency_ms'], ',.1f'):>12s} ms")

# %% [markdown]
# **Look at** the rows that say "not measured". The larger models and the hosted models are
# on the shortlist because somebody proposed them. Nobody has run them on the cooperative's
# data. And look at one licence: the model of 3 billion weights has other terms than its
# smaller and its larger sibling of the same family.

# %% [markdown]
# ## 2 · Licence and data constraints first
#
# The order of the questions matters. A model that may not be used is out, whatever its
# quality. So the screen comes before the comparison.
#
# ✏️ **Exercise** · Return the list of reasons why an option is **not** eligible for its use
# case (an empty list means eligible). Test in this order, and use these words:
#
# - `"licence"`: the model's `commercial_use` is `"no"`;
# - `"size"`: its `file_mb` is above `C["max_model_file_mb"]`;
# - `"not measured"`: the option's `quality` or `latency_ms` is `None`;
# - `"latency"`: its `latency_ms` is above the budget of the use case.

# %%
def screen(option, use_case):
    model = models[option["model"]]
    # BEGIN SOLUTION
    reasons = []
    if model["commercial_use"] == "no":
        reasons.append("licence")
    if model["file_mb"] > C["max_model_file_mb"]:
        reasons.append("size")
    if option["quality"] is None or option["latency_ms"] is None:
        reasons.append("not measured")
    elif option["latency_ms"] > C["latency_budget_ms"][use_case]:
        reasons.append("latency")
    return reasons
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _screen(option, use_case):
    model = models[option["model"]]
    reasons = ["licence"] if model["commercial_use"] == "no" else []
    reasons += ["size"] if model["file_mb"] > C["max_model_file_mb"] else []
    if option["quality"] is None or option["latency_ms"] is None:
        reasons.append("not measured")
    elif option["latency_ms"] > C["latency_budget_ms"][use_case]:
        reasons.append("latency")
    return reasons

try:
    screen(cases["naqsh"]["options"][0], "naqsh")
except NotImplementedError:
    screen = _screen


def sign_offs(option):
    """Who must sign before this option is used (it stays eligible; the recommendation says so)."""
    model = models[option["model"]]
    return (["licensing officer"] if model["commercial_use"] == "sign-off" else []) + \
           (["data protection officer"] if model["leaves_the_server"] else [])


for name, case in cases.items():
    print(name.upper())
    for o in case["options"]:
        reasons = screen(o, name)
        assert sorted(reasons) == sorted(_screen(o, name)), f"{o['id']}: compare with the four rules"
        verdict = "eligible" if not reasons else "out: " + ", ".join(reasons)
        print(f"  {o['id']:26s} {verdict:34s} " + ("sign-off: " + ", ".join(sign_offs(o)) if sign_offs(o) else ""))
eligible = {name: [o for o in case["options"] if not screen(o, name)] for name, case in cases.items()}

# %% [markdown]
# **Look at** what went out, and why. No option left because its quality was low: the screen
# does not look at quality. One model is out on its licence alone. One is out on its size:
# it does not fit on the shop's server. And every hosted option is out for one reason only:
# nobody measured it. That is a task for next week, not a verdict on hosted models.
#
# One eligible option carries a sign-off. The weights of the pretrained image model have a
# permissive licence; the photographs it was trained on were collected for non-commercial
# research. Whether that matters for a shop is a question for Bushra Al-Aqeel, the
# licensing officer. Your part is to put it in front of her.

# %% [markdown]
# ## 3 · Quality, size, latency and cost per use case
#
# Among the eligible options you compare four numbers. Three are in the tables. The fourth,
# cost, you compute. A model on the shop's own server costs nothing more per request: the
# server is there. A hosted model is paid per token.
#
# ✏️ **Exercise** · Return the cost in riyals of `requests` requests to an option. For a hosted
# model it is the tokens of all requests, in millions, times the price per million, for
# input and for output. For any other model it is 0. `tokens` is
# `{"prompt": ..., "output": ...}` per request.

# %%
def cost(option, requests, tokens):
    price = models[option["model"]]["price_per_million_tokens"]
    # BEGIN SOLUTION
    if price is None:
        return 0.0
    return requests * (tokens["prompt"] * price["input"] + tokens["output"] * price["output"]) / 1e6
    # END SOLUTION

# %%
def _cost(option, requests, tokens):
    price = models[option["model"]]["price_per_million_tokens"]
    return 0.0 if price is None else requests * (tokens["prompt"] * price["input"] + tokens["output"] * price["output"]) / 1e6

by_id = {o["id"]: o for case in cases.values() for o in case["options"]}
try:
    cost(by_id["wasf-hosted"], 1000, {"prompt": 100, "output": 20})
except NotImplementedError:
    cost = _cost
assert abs(cost(by_id["wasf-hosted"], 1000, {"prompt": 100, "output": 20}) - 0.36) < 1e-9, "1,000 requests of 100 and 20 tokens at 2 and 8 riyals per million cost 0.36 riyals"
assert cost(by_id["wasf-small"], 1000, {"prompt": 100, "output": 20}) == 0

VOLUME = {"wasf": (2000, "one product line for each of the 2,000 products"), "khayt": (30000, "a month of 30,000 searches")}
for name, (requests, what) in VOLUME.items():
    tokens = cases[name]["tokens_per_request"]
    hosted = next(o for o in cases[name]["options"] if models[o["model"]]["kind"] == "hosted")
    print(f"{name}: {what}, {tokens['prompt']} + {tokens['output']} tokens each, on {hosted['model']}: {cost(hosted, requests, tokens):.2f} riyals (exercise price list)")

print()
table = {}
for name, options in eligible.items():
    print(f"{name.upper()} · eligible options · budget {C['latency_budget_ms'][name]:,} ms")
    for o in options:
        m = models[o["model"]]
        extra = "".join(f"   {k.replace('_', ' ')}: {v}" for k, v in o.items() if k.startswith("quality_"))
        print(f"  {o['id']:26s} quality {o['quality']:.3f}   latency {o['latency_ms']:>8,.1f} ms   file {m['file_mb']:>6,} MB   {m['licence'][:28]:28s}{extra}")
    table[name] = [{"option": o["id"], "quality": o["quality"], "latency_ms": o["latency_ms"], "file_mb": models[o["model"]]["file_mb"],
                    "licence": models[o["model"]]["licence"], "sign_offs": sign_offs(o)} for o in options]

# %% [markdown]
# **Look at** each block with one question: what decides?
#
# - **Naqsh.** Both pretrained options beat the small network by 30 points. Between them
#   the gap is 0.788 against 0.774: 14 photographs in 1,000, and 0.779 once the
#   near-duplicates are out. Latency does not decide either: 35 ms is far inside 200 ms.
# - **Khayt.** Pairing the results by product takes recall at 5 from 0.225 to 0.335 at the
#   same latency and with the same model. The best row still finds the product in one
#   search of three, and half as often in Arabic as in English.
# - **Wasf.** The template scores 1.000: it cannot break a rule, by construction. The
#   measure counts rule violations; it does not see that 2,000 product lines read alike. The
#   small model with the strict instruction is at 0.950 and takes about 6.5 seconds a draft
#   on a processor: 2,000 drafts are a night's batch of under four hours.
# - **Cost.** At these volumes the hosted options would cost under one riyal on the
#   exercise price list. Cost does not decide here. What keeps them out is that nobody has
#   measured them, and that the text would leave the shop's server.

# %% [markdown]
# ## 4 · Two lines of justification each
#
# ✏️ **Exercise** · For each use case, choose one eligible option and write two lines:
# `because` (the numbers that decide, against the simpler alternative) and `but` (the risk
# or the limit, and who signs off when a sign-off is needed). Replace the reference text with
# your own.

# %%
def recommend():
    # BEGIN SOLUTION
    return {
        "naqsh": {"choice": "naqsh-frozen",
                  "because": "Frozen features reach 0.774 on the 1,000 test photographs against 0.476 for the small network trained from scratch, in a 45 MB file and well inside the latency budget. Fine-tuning adds 14 photographs in 1,000, which this test cannot tell from none.",
                  "but": "The backbone was trained on ImageNet photographs: the licensing officer signs off before the shop uses it, and the small network is the fallback. The accuracy is reported without the near-duplicates."},
        "khayt": {"choice": "khayt-embeddings-paired",
                  "because": "Recall at 5 of 0.335 against 0.175 for keyword search, with an MIT licence, on the shop's server and inside the latency budget.",
                  "but": "Two searches in three still miss, Arabic queries do worse than English ones, and the pairing was designed on the same 50 queries: it is confirmed on new queries before launch."},
        "wasf": {"choice": "wasf-small-strict",
                 "because": "With the strict instruction 95% of the drafts break no rule, against 56% without it, from a model of 1 GB with an Apache 2.0 licence that runs on the shop's server.",
                 "but": "The checks do not test whether a draft is true to the product, and Arabic was not measured: a member of the team reads every line, the limits note of Lab 4.3 goes with it, and the template is the fallback."},
    }
    # END SOLUTION

# %%
try:
    rec = recommend()
except NotImplementedError:
    rec = U.reference("d4_4_table")["recommendation"]
for name, r in rec.items():
    option = by_id[r["choice"]]
    assert option in eligible[name], f"{name}: {r['choice']} did not pass the screen"
    assert any(ch.isdigit() for ch in r["because"]), f"{name}: 'because' names at least one measured number"
    assert len(r["but"]) > 40, f"{name}: 'but' names a risk or a limit"
    assert not sign_offs(option) or "sign" in r["but"].lower(), f"{name}: this option needs a sign-off; say who signs"
    print(f"{name.upper()}: {r['choice']}\n  because {r['because']}\n  but {r['but']}\n")
U.save_run("d4_4_table", {"table": table, "recommendation": rec})
U.save_reference("d4_4_table", {"table": table, "recommendation": rec})

# %% [markdown]
# **Look at** your three choices next to a neighbour's. Where you differ, the difference is
# seldom in the numbers: it is in which risk each of you would rather carry. That is the
# part a table cannot decide, and the part you defend.

# %% [markdown]
# ## 5 · Choose the capstone task
#
# Tomorrow you build one of the three, alone, and defend it. Choose now, and name the
# simpler alternative you will compare with and the number you will report.

# %%
capstone = {
    "task": "naqsh",                       # "naqsh", "khayt" or "wasf"
    "simpler_alternative": "the small convolutional network trained from scratch",
    "measure": "test accuracy without near-duplicates, per class",
    "first_hour": "uv run python capstone/starter.py naqsh",
}
assert capstone["task"] in cases, "one of the three use cases"
assert all(len(str(v)) > 3 for v in capstone.values())
print(U.save_run("d4_4_capstone", capstone).name, "written:", capstone)
print("the brief and the rubric: capstone/README.md")

# %% [markdown]
# ## ✱ Stretch · Change one constraint
#
# Set `C["max_model_file_mb"] = 20000` and run the screen again. Which options change their
# reason, and does any become eligible? Then set a latency budget of 5 ms for Naqsh. Which
# choice flips, and what would you tell the director?

# %% [markdown]
# ## Check yourself
#
# 1. Why is the model with a research licence screened out before anyone looks at its quality?
# 2. The hosted options are "not measured". Does that mean a hosted model is a bad choice?
# 3. Two eligible options differ by 1.4 points of accuracy on 1,000 photographs. What decides
#    between them?
#
# <details><summary>Answers</summary>
#
# 1. A model that may not be used cannot be chosen. Looking at its quality first makes it the
#    favourite, and then the licence looks like an obstacle to argue away.
# 2. No. It means there is no evidence yet on this task. Measuring it needs the data
#    protection officer's sign-off first, because the text leaves the shop's server.
# 3. Not the accuracy: 14 photographs are within the noise of this test. Cost, simplicity
#    and risk decide, or a larger test.
# </details>
#
# ## Carry forward
#
# `runs/d4_4_table.json` is the model-choice table of your capstone README, and
# `runs/d4_4_capstone.json` is your task. The capstone brief is in `capstone/README.md`.
