# %% [markdown]
# # Lab 3.2 · Attention by hand
#
# Day 3 · Session 2 (Attention step by step) · 25 min.
#
# Attention is four lines of tensor code. You write them for a toy sequence of five tokens,
# add the mask that turns an encoder into a decoder, and check your function against the one
# in the library. Then you read an attention pattern and see why a transformer needs to be
# told the order of its tokens.
#
# **Tasks**
# - [ ] Scores: queries against keys · 6 min
# - [ ] Weights: a softmax per row · 4 min
# - [ ] The causal mask · 6 min
# - [ ] The output, checked against the library · 5 min
# - [ ] Read an attention pattern · 4 min
# - [ ] ✱ Stretch · Split the vectors into two heads
#
# AI policy for this lab: **AI-off for your first attempt.** Write the four lines yourself;
# an assistant may explain an error message afterwards.

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

import math
import torch
import torch.nn.functional as F
import nawl as U

tokens = ["red", "striped", "wool", "rug", "."]
torch.manual_seed(7)
d = 8
X = torch.randn(len(tokens), d)             # one vector of 8 numbers per token (its embedding)
Wq, Wk, Wv = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv            # queries, keys and values: three views of the same tokens
print("X", tuple(X.shape), " Q", tuple(Q.shape), " K", tuple(K.shape), " V", tuple(V.shape))

# %% [markdown]
# ## 1 · Scores: queries against keys
#
# Each token asks a question (its query) and offers a label (its key). The score of token
# `i` for token `j` is the dot product of query `i` with key `j`, divided by `sqrt(d)` so
# that the numbers do not grow with the size of the vectors.
#
# ✏️ **Exercise** · Return the 5 x 5 matrix of scores.

# %%
def scores(Q, K):
    # BEGIN SOLUTION
    return Q @ K.T / math.sqrt(K.shape[-1])
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    S = scores(Q, K)
except NotImplementedError:
    scores = lambda Q, K: Q @ K.T / math.sqrt(K.shape[-1])
    S = scores(Q, K)
assert S.shape == (5, 5)
assert torch.allclose(S[1, 3], (Q[1] * K[3]).sum() / math.sqrt(d)), "row i is the query, column j is the key"
print(S.round(decimals=2))

# %% [markdown]
# ## 2 · Weights: a softmax per row
#
# ✏️ **Exercise** · Turn each row of scores into weights that are positive and add up to 1.

# %%
def weights(S):
    # BEGIN SOLUTION
    return torch.softmax(S, dim=-1)
    # END SOLUTION

# %%
try:
    A = weights(S)
except NotImplementedError:
    weights = lambda S: torch.softmax(S, dim=-1)
    A = weights(S)
assert torch.allclose(A.sum(dim=-1), torch.ones(5)), "every row must add up to 1 (softmax over the last axis)"
print(A.round(decimals=2))
print("row sums:", A.sum(dim=-1).round(decimals=4).tolist())

# %% [markdown]
# ## 3 · The causal mask
#
# A model that generates text predicts token `i + 1` from tokens `1..i`. While it is trained
# on a whole sentence at once, token `i` must not look at the tokens after it: they are the
# answer. The mask sets those scores to minus infinity before the softmax, so their weights
# become exactly 0.
#
# ✏️ **Exercise** · Return the scores with every entry above the diagonal replaced by
# `float("-inf")` (`torch.triu(..., diagonal=1)` selects them).

# %%
def causal(S):
    # BEGIN SOLUTION
    n = S.shape[-1]
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return S.masked_fill(future, float("-inf"))
    # END SOLUTION

# %%
def _causal(S):
    n = S.shape[-1]
    return S.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1), float("-inf"))

try:
    Ac = weights(causal(S))
except NotImplementedError:
    causal = _causal
    Ac = weights(causal(S))
assert torch.allclose(Ac.sum(dim=-1), torch.ones(5)) and float(torch.triu(Ac, diagonal=1).abs().sum()) == 0.0
assert float(Ac[0, 0]) == 1.0, "the first token can only attend to itself"
print(Ac.round(decimals=2))

# %% [markdown]
# **Look at** the first row: one token, weight 1. And the last row: it may look at all five.

# %% [markdown]
# ## 4 · The output, checked against the library
#
# The output for token `i` is the weighted mean of all value vectors, with row `i` of the
# weights.
#
# ✏️ **Exercise** · Put the pieces together: scores, the causal mask when `is_causal` is
# true, weights, then the weighted sum of `V`. Return `(output, weights)`.

# %%
def attention(Q, K, V, is_causal=False):
    # BEGIN SOLUTION
    S = scores(Q, K)
    if is_causal:
        S = causal(S)
    A = weights(S)
    return A @ V, A
    # END SOLUTION

# %%
def _attention(Q, K, V, is_causal=False):
    S = Q @ K.T / math.sqrt(K.shape[-1])
    A = torch.softmax(_causal(S) if is_causal else S, dim=-1)
    return A @ V, A

try:
    attention(Q, K, V)
except NotImplementedError:
    attention = _attention
results = {}
for flag in (False, True):
    out, A_ = attention(Q, K, V, is_causal=flag)
    lib = F.scaled_dot_product_attention(Q, K, V, is_causal=flag)
    results[flag] = float((out - lib).abs().max())
    assert out.shape == (5, d)
    assert torch.allclose(out, lib, atol=1e-5), "your output differs from the library function"
    print(f"is_causal={flag!s:5s} largest difference from F.scaled_dot_product_attention: {results[flag]:.1e}")
U.save_reference("d3_2_check", {"max_diff_plain": results[False], "max_diff_causal": results[True]})

# %% [markdown]
# ## 5 · Read an attention pattern
#
# Two questions about what you built.

# %%
out, A_ = attention(Q, K, V, is_causal=True)
print("weights with the causal mask (rows: the token that looks; columns: the token looked at)\n")
print(" " * 9 + " ".join(f"{t:>8s}" for t in tokens))
for t, row in zip(tokens, A_):
    print(f"{t:>8s} " + " ".join(f"{w:8.2f}" for w in row.tolist()))
looks_at = tokens[int(A_[3].argmax())]
print(f"\n'rug' puts its largest weight on '{looks_at}' (these embeddings are random: in a trained model"
      " this is where 'rug' would collect its colour and its pattern)")

# the order of the tokens: shuffle the inputs and see what happens to the outputs
perm = torch.tensor([3, 0, 4, 1, 2])
out_plain, _ = attention(Q, K, V)
out_shuffled, _ = attention(Q[perm], K[perm], V[perm])
same = torch.allclose(out_shuffled, out_plain[perm], atol=1e-6)
print("\nshuffle the tokens, and each token's output is unchanged (it only moved with its token):", same)
assert same

# %% [markdown]
# **Look at** the last line. Without the mask, attention treats its input as a bag of
# tokens: "red rug striped wool" gives every token the same output as "red striped wool
# rug". This is why a transformer adds a **position** signal to each embedding before the
# first attention layer.

# %% [markdown]
# ## ✱ Stretch · Two heads
#
# Split `Q`, `K` and `V` into two halves of 4 numbers each (`Q[:, :4]` and `Q[:, 4:]`), run
# `attention` on each half and concatenate the two outputs. That is multi-head attention
# with two heads. Why can two heads look at two different things for the same token?

# %% [markdown]
# ## Check yourself
#
# 1. Why divide the scores by `sqrt(d)`?
# 2. What would a decoder learn if you trained it without the causal mask?
# 3. Attention alone ignores word order. Name the two things in a transformer that put
#    order back.
#
# <details><summary>Answers</summary>
#
# 1. Dot products of longer vectors are larger. Large scores push the softmax towards a
#    single 1 and many 0s, where gradients vanish; the division keeps the scale steady.
# 2. To copy: every position could read the next token, which is its target. The loss would
#    go to zero and the model could generate nothing.
# 3. The position signal added to each embedding, and, in a decoder, the causal mask, which
#    makes the result depend on what comes before.
# </details>
#
# ## Carry forward
#
# The four lines are the whole mechanism. A transformer repeats them in several heads and
# layers, with a small fully connected network after each. Day 4 looks at what comes out of
# the last layer: a score for every token of the vocabulary.
