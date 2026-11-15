---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/slides/day3.html"
---

# Deep Learning and Generative AI Fundamentals

## Day 3: Transformers and embeddings

How text becomes tokens and vectors, what attention computes, and where bilingual search fails

#### [day3#title] Transformers and embeddings

*SDAIA Academy · Deep Learning and Generative AI Fundamentals*

Day 3: How text becomes tokens and vectors, what attention computes, and where bilingual search fails

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-202

#### [day3#agenda] Transformers and embeddings

*Today*

- **Session 1 · Tokens, including Arabic**: How a tokeniser cuts text, why Arabic costs more tokens, and what that changes downstream
- **Session 2 · Attention step by step**: Scaled dot-product attention as four lines of tensor code, with the mask that makes a decoder
- **Session 3 · Sentence embeddings**: One vector per text, cosine similarity, brute-force search and recall at k
- **Session 4 · Multilingual search and its failures**: What a multilingual embedding model does with Arabic, English and mixed queries, and how to read its failures

#### [day3#review-day2] Which row would you keep?

*Review · Day 2*

**Question:** Frozen features reach 0.774 in 24 seconds; fine-tuning the last block reaches 0.788 in 102 seconds, on the same 1,000 test photographs. Which one goes on the upload page?

- A. The fine-tuned model, because its accuracy is higher
- B. The frozen-feature model, because it trains four times faster
- C. Either: the test cannot tell them apart, so decide on cost and simplicity, or test more
- D. Neither: with 800 labels a model should be trained from scratch

#### [day3#transformer-map] From text in to one token out: five stations

*The map for two days*

1. **1 · Tokens** text cut into pieces, each a number
2. **2 · Embeddings** a vector per token, plus its position
3. **3 · Attention** each token gathers from the others, layer after layer
4. **4 · Scores** one number for every token of the vocabulary
5. **5 · Sampling** one token is chosen and appended

All of it must fit in the **context window**: the largest number of tokens the model reads at once. Today: stations 1 to 3. Tomorrow: 4 and 5.

### Session 1: Tokens, including Arabic

#### [day3#s1] Tokens, including Arabic

*Session 1*

- Tokenise a sentence and read its pieces
- Compare token counts of Arabic and English text across tokenisers
- Explain what the token count changes: the window, the time and the price

#### [day3#model-reads-tokens] A model never sees text; it sees token numbers

*Station 1*

"A red striped rug" for Qwen2.5: 4 tokens

`A` ` red` ` striped` ` rug`

The same in Arabic, سجادة مخططة حمراء: 9 tokens

`س` `ج` `ادة` ` م` `خط` `طة` ` ح` `مر` `اء`

- A **tokeniser** cuts text into pieces from a fixed vocabulary and returns their numbers
- Frequent words are one token; rare words are several pieces
- The tokeniser is built before training and **belongs to the model**

#### [day3#subword-vocabulary] Subword vocabularies are learnt from text by merging frequent pairs

*How the vocabulary is built*

- Start from single bytes or characters
- Merge the most frequent neighbouring pair into a new piece; repeat tens of thousands of times
- What was frequent in the **training text** becomes one token
- Anything can still be written: in the worst case, byte by byte

| Tokeniser in the lab | Vocabulary | Built from |
| --- | --- | --- |
| GPT-2 | 50,257 | English web text |
| Qwen2.5 | 151,665 | Multilingual text |
| multilingual-e5 | 250,002 | About 100 languages |

#### [day3#arabic-costs-more] The same product costs more tokens in Arabic, by a factor that depends on the tokeniser

*What the lab measured*

| Twelve products, each described in both languages | Tokens per English word | Tokens per Arabic word | Arabic tokens ÷ English tokens |
| --- | --- | --- | --- |
| GPT-2 (built from English) | 1.54 | 4.71 | 3.14 |
| Qwen2.5 (multilingual) | 1.86 | 2.51 | 1.39 |
| multilingual-e5 | 1.68 | 1.94 | 1.18 |

> **Two causes.** A vocabulary with few Arabic pieces falls back to bytes, two per letter. And an Arabic word carries more: باللون is "in the colour" in one word.

#### [day3#how-customers-write] Customers do not write like the catalogue

*Beyond the catalogue*

| The same three words | Text | GPT-2 | e5 | Qwen2.5 |
| --- | --- | --- | --- | --- |
| Catalogue Arabic | سجادة مخططة حمراء | 20 | 7 | 9 |
| With short vowels | سَجَّادَة مُخَطَّطَة حَمْرَاء | 35 | 23 | 25 |
| Arabizi | sajjada mukhattata 7amra | 11 | 8 | 12 |
| English | red striped rug | 3 | 4 | 3 |

> **What goes wrong.** Each short-vowel mark is a character of its own. Vowelled text is cut into single letters and marks: three times the tokens, in pieces the model has rarely seen.

#### [day3#three-bills] Three bills are paid in tokens

*Why you count*

- **The window.** A model reads a limited number of tokens at once. A window of 2,048 holds about 86 English descriptions or 62 Arabic ones (Lab 3.1).
- **The time.** A generating model produces one token per step. More tokens for the same sentence is more waiting.
- **The price.** Hosted models charge per token read and per token written. The same content can cost more in one language.

#### [day3#embedding-and-position] Each token number becomes a vector, and gets its position added

*Station 2*

- The **embedding table** has one row per token of the vocabulary: a vector of a few hundred to a few thousand numbers
- The rows are weights: learnt by the same five lines as everything else
- A **position** signal is added, so that "rug" at position 4 differs from "rug" at position 1

- **In numbers.** Qwen2.5-0.5B: 151,665 rows of 896 numbers, 136 million of its 494 million weights. A small model spends a quarter of itself on the table.

#### [day3#check-tokens] Why did the Arabic prompt not fit?

*Checkpoint*

**Question:** Rayan pastes 70 product descriptions into a small model with a window of 2,048 tokens. The English batch is answered; the Arabic batch of the same 70 products loses its beginning. What explains it?

- A. The Arabic descriptions have more characters than the English ones
- B. The Arabic descriptions need more tokens for the same content
- C. The tokeniser has no Arabic pieces and skips unknown words
- D. The model's window is smaller for right-to-left languages

#### [day3#lab-3-1] Three tokenisers

Lab 3.1 · 25 min

Tokenise the same Arabic and English product sentences with three tokenisers and compare the token counts.

[D3_1_three_tokenisers.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D3_1_three_tokenisers.ipynb)

- [ ] One sentence pair, three tokenisers (5 min)
- [ ] Token counts for twelve pairs (6 min)
- [ ] Tokens per word, Arabic against English (5 min)
- [ ] Diacritics, Arabizi and digits (5 min)
- [ ] What the count changes (4 min)
- [ ] ✱ Stretch: Measure what tatweel and spelling variants do to the count

### Session 2: Attention step by step

#### [day3#s2] Attention step by step

*Session 2*

- Compute attention scores, weights and outputs for a toy sequence
- Apply a causal mask and say what it prevents
- Check an implementation against the library function

#### [day3#why-attention] A word means what its neighbours make it mean

*Station 3*

- "red striped wool **rug**": the vector of "rug" should end up knowing its colour, pattern and material
- A convolution looks at fixed neighbours; text needs to reach **any** other position
- Attention lets every token collect from every other, with weights computed from the tokens themselves

- **One sentence.** Attention replaces each token's vector by a weighted mean of all tokens' vectors, and learns where the weights should be large.

#### [day3#query-key-value] Every token asks a question, offers a label and carries a content

*Three views of one token*

- **Query.** What this token is looking for. `Q = X @ Wq`
- **Key.** What this token can be found by. `K = X @ Wk`
- **Value.** What this token hands over when it is found. `V = X @ Wv`

- Three weight matrices, learnt; the rest of attention has **no weights at all**
- The score of token i for token j is the dot product of query i and key j

#### [day3#four-lines] Scaled dot-product attention is four lines

*The mechanism*

```python
S = Q @ K.T / math.sqrt(d)          # 1 scores: every query against every key
S = S.masked_fill(future, -inf)     # 2 the causal mask (decoders only)
A = torch.softmax(S, dim=-1)        # 3 weights: each row adds up to 1
out = A @ V                         # 4 a weighted mean of the values
```

> **Shapes.** Five tokens with vectors of 8: Q, K and V are (5, 8), the scores and weights (5, 5), the output (5, 8) again. A layer of attention keeps the shape of what it reads.

#### [day3#causal-mask] A decoder may look back, never ahead

*The mask*

|  | red | striped | wool | rug | . |
| --- | --- | --- | --- | --- | --- |
| red | 1.00 | 0 | 0 | 0 | 0 |
| striped | 0.31 | 0.69 | 0 | 0 | 0 |
| wool | 0.15 | 0.09 | 0.75 | 0 | 0 |
| rug | 0.18 | 0.19 | 0.15 | 0.48 | 0 |
| . | 0.18 | 0.16 | 0.38 | 0.05 | 0.22 |

- Rows: the token that looks. Columns: the token looked at
- Scores above the diagonal are set to minus infinity; their weights become exactly 0
- In training, the token after position i is the **answer** for position i: it must not be visible

#### [day3#order-needs-position] Attention alone does not know the order of its tokens

*A consequence*

- Shuffle the five tokens: each token's output is unchanged, it only moves with its token (Lab 3.2)
- "rug red" and "red rug" would be the same to it
- Hence the **position signal** added to each embedding at station 2

> **Two things put order back.** The position signal, and in a decoder the causal mask, which makes each output depend on what came before it.

#### [day3#block-and-stack] A transformer repeats one block many times

*From one layer to a model*

1. **Attention** several heads side by side: tokens exchange information
2. **Feed-forward** a small two-layer network on each token by itself
3. **Repeat** 24 blocks in Qwen2.5-0.5B

- **Heads**: the vectors are split into parts that attend separately, so one token can look for several things at once
- Around each part: a skip connection and a normalisation, as in yesterday's ResNet, for the same reason: depth that still trains
- The cost of attention grows with the **square** of the number of tokens: one more reason windows are limited

#### [day3#three-architectures] Encoder, decoder, or both: the mask decides what the model is for

*Three shapes of transformer*

- **Encoder.** No mask: every token sees the whole text. Output: one vector per token or per text. For classifying, tagging, **embedding**. This afternoon's model.
- **Decoder.** Causal mask: each token sees the past. Output: scores for the next token. For **generating**. Tomorrow's model, and every chat assistant.
- **Encoder-decoder.** An encoder reads the input; a decoder writes the output while attending to it. For translation and summarising with a fixed input.

#### [day3#check-attention] What does the mask protect?

*Checkpoint*

**Question:** Zaid trains a small decoder to predict the next token of product descriptions and forgets the causal mask. The training loss falls to almost zero within one epoch. When he asks the model to write a description, it produces nonsense. Why?

- A. It memorised the training descriptions and cannot generalise
- B. It never learnt the order of the tokens in a sentence
- C. Its learning rate was too high for a model of that size
- D. In training each position could read the token it had to predict

#### [day3#lab-3-2] Attention by hand

Lab 3.2 · 25 min

Implement scaled dot-product attention for a toy sequence and check it against the library function.

[D3_2_attention.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D3_2_attention.ipynb)

- [ ] Scores: queries against keys (6 min)
- [ ] Weights: a softmax per row (4 min)
- [ ] The causal mask (6 min)
- [ ] The output, checked against the library (5 min)
- [ ] Read an attention pattern (4 min)
- [ ] ✱ Stretch: Split the vectors into two heads

### Session 3: Sentence embeddings

#### [day3#s3] Sentence embeddings

*Session 3*

- Embed texts and compare them with cosine similarity
- Build a brute-force search over 4,000 vectors
- Measure recall at 5 on labelled queries against a keyword baseline

#### [day3#one-vector-per-text] An embedding model turns a whole text into one vector

*Sentence embeddings*

1. **Text** a description or a query
2. **Encoder** one vector per token
3. **Pool** the mean over the tokens
4. **One vector** 384 numbers, scaled to length 1

- Trained on pairs of texts that belong together (a question and its answer, a sentence and its translation): **their vectors are pulled together, others pushed apart**
- Texts with similar meaning then lie close; you search by distance

#### [day3#cosine-similarity] Cosine similarity compares directions, not lengths

*Similarity*

- `cos(a, b) = a · b / (|a| |b|)`: 1 for the same direction, 0 for a right angle
- For vectors of length 1 it is the **dot product**: one matrix product compares a query with every description
- Other measures (dot product on raw vectors, Euclidean distance) rank the same when all vectors have length 1

```python
D = U.embed(texts, "passage: ")  # (4000, 384)
q = U.embed([query])[0]          # (384,)
sims = D @ q                     # (4000,)
top = np.argsort(-sims)[:5]
```

#### [day3#similarities-are-a-ranking] Read similarities as a ranking, not as a scale

*What the lab measured*

| Compared with "a red striped rug" | Cosine similarity |
| --- | --- |
| Its Arabic translation, سجادة مخططة باللون الأحمر | 0.87 |
| An English paraphrase, "a crimson carpet with lines" | 0.83 |
| Another product, "a blue dotted shawl" | 0.79 |
| An unrelated question, "how do I return an order?" | 0.78 |

> **What goes wrong.** A threshold such as "similar above 0.7" copied from another model. Here everything is above 0.7; the whole ranking happens between 0.76 and 0.90.

#### [day3#brute-force-and-indexes] Brute force is exact and enough for thousands; an index trades exactness for speed

*Searching*

- **Brute force.** Compare the query with every vector: one matrix product. Exact. 4,000 descriptions take well under a millisecond in NumPy.
- **An approximate index.** Organise the vectors so that a query visits only some of them (FAISS is one example). Needed from millions; may miss a true neighbour, so recall must be measured again.

> **Start exact.** Measure quality with brute force first. Then an index has something to be compared with.

#### [day3#recall-at-k] Recall at 5: of what should come back, how much is in the first five?

*The measure*

- Rayan's team wrote 50 queries and marked the **relevant** descriptions of each
- Each query names a kind of product, a pattern and a colour: two products match, each described in both languages, so **four** relevant descriptions
- Recall at 5 = relevant found in the first five ÷ 4, averaged over the queries

- query: red wavy table runner
- 1 ✓: This table runner is wavy and red. Material: camel hair.
- 2 ✓: Wavy table runner made of linen. Colour: red.
- 3: Wavy table runner made of palm fibre. Colour: orange.
- 4: Wool table runner with a wavy design in white.

#### [day3#recall-result] Recall at 5 is 0.23: better than keywords, and not good

*What the lab measured*

| Recall at 5 | All 50 | English (20) | Arabic (20) | Mixed (10) |
| --- | --- | --- | --- | --- |
| Keywords (the simpler alternative) | 0.175 | 0.263 | 0.113 | 0.125 |
| Embeddings | 0.225 | 0.300 | 0.188 | 0.150 |

- Embeddings better on 15 queries, keywords on 4, equal on 31
- 21 of 50 queries find nothing relevant in the first five

> **A number means something only next to an alternative** measured on the same queries. Then ask where it fails.

#### [day3#check-recall] What does this recall mean?

*Checkpoint*

**Question:** A query has four relevant descriptions. The search returns five results; two of them are relevant. A colleague says: "two out of five, so recall at 5 is 0.4". What is the right figure?

- A. 0.5: two of the four relevant descriptions were found
- B. 0.4: two of the five results are relevant
- C. 1.0: at least one relevant description is in the first five
- D. 0.8: four relevant descriptions for five places

#### [day3#lab-3-3] Catalogue search

Lab 3.3 · 25 min

Embed the product descriptions, build a brute-force search and measure recall at 5 on the 50 labelled queries.

[D3_3_catalogue_search.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D3_3_catalogue_search.ipynb)

- [ ] Embed and compare six sentences (5 min)
- [ ] Brute-force search over the catalogue (7 min)
- [ ] Recall at 5 on 50 queries (8 min)
- [ ] Against a keyword baseline (5 min)
- [ ] ✱ Stretch: Time the search and estimate it for a million descriptions

### Session 4: Multilingual search and its failures

#### [day3#s4] Multilingual search and its failures

*Session 4*

- Compare recall by query language
- Separate a failure of language from a failure of wording
- Write up three failure cases with a likely cause and a fix to try

#### [day3#split-the-number] A low number is a question: split it until it answers

*Failure analysis*

1. **1 · Split** by query language, by how the query is worded
2. **2 · Look** at what came back for the failed queries
3. **3 · Name** a likely cause for each kind of failure
4. **4 · Try** one small fix per cause, and measure it

> **Mind the group sizes.** 20, 20 and 10 queries: one query is 5 or 10 points of its group. These numbers rank the groups; they do not measure them finely.

#### [day3#language-wall] The search never crossed from one language to the other

*What the lab measured*

- **100%** of the top-5 results are in the query's own language
- **0.47** of the same-language relevant descriptions were found
- **0.00** of the other-language relevant descriptions were found

- The best other-language relevant description sits at a median rank of **1,988 of 4,000**
- Half of what should come back is behind a wall: recall at 5 could never pass 0.5

#### [day3#two-fixes] The fix that worked used a fact the catalogue already had

*Two fixes, both measured*

| Recall at 5 | All 50 | English | Arabic | Mixed |
| --- | --- | --- | --- | --- |
| Plain search | 0.225 | 0.300 | 0.188 | 0.150 |
| A: three results from the query's language, two from the other | 0.210 | 0.300 | 0.150 | 0.150 |
| B: after each result, add the same product's other description | 0.335 | 0.475 | 0.250 | 0.225 |

> **Fix A asked the model to cross the wall and it could not.** Fix B did not ask: the catalogue knows which two descriptions are one product.

#### [day3#how-queries-fail] The remaining failures are gaps between how customers write and how the catalogue is written

*Reading the failures*

| Query | What came back | Likely cause |
| --- | --- | --- |
| I am looking for a indigo pillow cover with a zigzag design | Cushion covers with dots and triangles, other colours | Synonyms: the catalogue says "chevron" and "blue" |
| ودي بجزدان بنقاط أخضر زينة | Green wall hangings, belts and blankets | A dialect word for the product; only the colour matched |
| abgha salla mu3ayyanat aswad | Shawls in other patterns and colours | Arabizi: Arabic in Latin letters and digits |
| أبحث عن فاصل كتاب بنقشة النجوم لونها أحمر | Star-patterned bookmarks in orange and black | Wrong attribute: right product and pattern, another colour |

#### [day3#multilingual-is-uneven] "Multilingual" is a claim to test on your own languages and your own text

*Multilingual models*

- Quality differs by language: by how much text of it was in training, and of which kind
- Standard Arabic is covered far better than **dialects**; **Arabizi** and **code-switching** are rarer still
- Spelling varies between writers: hamza forms, ta marbuta, short vowels. Normalise both the catalogue and the query, then measure
- Report results **per language**; an average hides the group that is served worst

> **Responsible use.** A service that works better in one language than in another treats its users unequally. Measuring it per language is the first step, and it goes in the README.

#### [day3#what-to-write-down] A failure case has four parts

*The write-up*

1. **The query**, as the customer wrote it
2. **What came back**, in your words
3. **A likely cause**: say "likely", you have not proven it
4. **One fix to try**, small enough to measure

- **Why "likely".** One query does not prove a cause. The fix is the experiment: if recall on the queries of that kind rises, the cause was right.

#### [day3#check-wall] What do you tell Rayan?

*Checkpoint*

**Question:** Rayan reads "recall at 5: English 0.48, Arabic 0.25" after fix B and proposes to translate every Arabic query to English first. Which reply uses what the lab measured?

- A. "No need: the two numbers are within noise of each other"
- B. "Better to replace the model by a larger one before anything else"
- C. "Worth a test: the Arabic failures were dialect, Arabizi and spelling, which translation may or may not fix"
- D. "It will work: the wall between the languages was the whole problem"

#### [day3#lab-3-4] Failures by language

Lab 3.4 · 25 min

Compare Arabic, English and mixed queries, find where the relevant descriptions rank and collect three failure cases with a likely cause.

[D3_4_failures_by_language.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D3_4_failures_by_language.ipynb)

- [ ] Recall at 5 by query language (5 min)
- [ ] Same-language and cross-language hits (6 min)
- [ ] Two fixes: merge by language, pair by product (6 min)
- [ ] Three failure cases with a likely cause (8 min)
- [ ] ✱ Stretch: Fold Arabic spelling variants and measure again

#### [day3#wrap-up] Three things to keep

*Day 3 · wrap-up*

- **Count tokens.** The window, the time and the price are paid in tokens, and Arabic costs 1.2 to 3 times more depending on the tokeniser.
- **Attention is four lines.** Scores, mask, softmax, weighted mean. The mask makes a decoder; the position signal gives it order.
- **Measure search per language.** 0.23 overall hid a wall between two languages. A baseline, a split and three failure cases found it.

#### [day3#next] Day 4: Generation and model choice

*Next*

Stations 4 and 5: scores over the vocabulary, and how one token is chosen. Then the limits of a small language model, and a model choice made from this week's measurements.
