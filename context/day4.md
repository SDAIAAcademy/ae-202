---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/slides/day4.html"
---

# Deep Learning and Generative AI Fundamentals

## Day 4: Generation and model choice

How a language model writes one token at a time, what tuning adds, where it fails, and how to choose a model

#### [day4#title] Generation and model choice

*SDAIA Academy · Deep Learning and Generative AI Fundamentals*

Day 4: How a language model writes one token at a time, what tuning adds, where it fails, and how to choose a model

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-202

#### [day4#agenda] Generation and model choice

*Today*

- **Session 1 · Next-token prediction and sampling**: A language model as a distribution over the next token, and what temperature and nucleus sampling do to it
- **Session 2 · From base models to assistants**: What instruction tuning and preference tuning add to a base model, and what reasoning effort costs
- **Session 3 · Limits: fabrication, cut-off and context**: Why a model answers when it cannot know, what a knowledge cut-off is, and what happens past the context window
- **Session 4 · Choosing a model family**: Encoders, decoders, diffusion and vision-language models, open-weight against hosted, and a choice made from measurements

#### [day4#review-day3] Which sentence is safe in a README?

*Review · Day 3*

**Question:** Pairing each result with the same product's other description took recall at 5 from 0.225 to 0.335 on the 50 labelled queries. Which sentence reports that without claiming more?

- A. "Pairing fixed the model's cross-language search"
- B. "Pairing raised recall at 5 from 0.225 to 0.335 on the queries it was designed on; new queries will confirm it"
- C. "The search now serves Arabic and English customers equally well"
- D. "A larger embedding model would have done better still"

#### [day4#stations-four-five] Today: from the last vector to one chosen token

*The map, second half*

1. **1 · Tokens** Day 3
2. **2 · Embeddings** Day 3
3. **3 · Attention** Day 3
4. **4 · Scores** one number for every token of the vocabulary
5. **5 · Sampling** one token is chosen and appended

Everything must fit in the **context window**. Session 3 shows what happens when it does not.

### Session 1: Next-token prediction and sampling

#### [day4#s1] Next-token prediction and sampling

*Session 1*

- Turn logits into probabilities at a given temperature
- Apply nucleus sampling to a distribution by hand
- Measure variation, length and rule violations across samples

#### [day4#one-token-at-a-time] A language model does not write a sentence; it scores the next token

*Station 4*

1. **Read** all tokens so far, through every block
2. **Score** one number (a logit) for each of 151,936 tokens
3. **Choose** one token
4. **Append** and run the model again

- The last layer turns the vector at the **last position** into one score per token of the vocabulary
- A product line of 19 tokens is 19 runs of the model, each on a text one token longer
- Training was the same task: given real text, predict its next token. Cross-entropy loss, the five lines of Day 1

#### [day4#scores-to-probabilities] The scores for the first token of one product line

*What the lab loads*

| Item | Value |
| --- | --- |
| Co | 0.296 |
| " | 0.128 |
| Explore | 0.098 |

Probabilities at temperature 1, for the product "coasters, check, blue, cotton".

- The **softmax** turns scores into probabilities that add up to 1
- The three most likely tokens hold about half of the probability
- The 200 most likely hold 0.92; the other 151,736 tokens share the rest
- That long, thin **tail** is what the sampling settings act on

#### [day4#greedy-decoding] Greedy decoding takes the most likely token, every time

*The simplest choice*

- The same prompt gives the **same text**: five drafts of a product were one draft
- "Temperature 0" is the usual name for it
- Useful where a result must be repeatable: tests, extraction, the ten tasks of Lab 4.2

- **What the lab measured.** At temperature 0, 25 of 100 drafts still broke a rule: every one of them was longer than 20 words.

> **What goes wrong.** "Temperature 0, so the model is right." Greedy decoding makes an output repeatable. It does not make it correct.

#### [day4#temperature] Temperature divides the scores before the softmax

*Station 5*

```python
z = (logits - logits.max()) / t
p = np.exp(z) / np.exp(z).sum()
```

- The **order** of the tokens never changes
- Below 1 the first tokens gain; above 1 the tail gains

| Temperature | First token | Tokens holding 90% |
| --- | --- | --- |
| 0.2 | 0.981 | 1 |
| 0.7 | 0.529 | 8 |
| 1.0 | 0.296 | 53 |
| 1.5 | 0.119 | 126 |

#### [day4#nucleus-sampling] Nucleus sampling cuts the tail that temperature opens

*Station 5, second setting*

- Sort the tokens; keep the most likely until their probabilities add up to `top_p`
- Drop the rest and share the probability again among those kept
- The two settings pull in opposite directions, and every request has a value for both, if only the default

| At temperature 1.5 | Tokens that can be drawn | First token |
| --- | --- | --- |
| top_p 1.0 | all | 0.119 |
| top_p 0.9 | 126 | 0.132 |
| top_p 0.5 | 19 | 0.233 |

> **Read the middle row.** At a high temperature a nucleus of 0.9 still holds more than a hundred tokens. It cuts little.

#### [day4#three-drafts] A wide tail is not creative; it is noise

*What the drafts look like*

| Setting | One draft for "coasters, check, blue, cotton, 35 x 150 cm" |
| --- | --- |
| Temperature 0 | Coasters, check, blue, cotton, 35 x 150 cm, perfect for a cozy summer getaway. |
| Temperature 0.7 | Coasters, a unique collection of blue blue cotton check coasters measuring 35 inches x 150 cm, perfect for adding a touch of nature to any room. |
| Temperature 1.5 | Coasters adorned in the muted blue color are coupé them beyond narrvcournoise vaccination strain elite align heading luminance |

- Words shared by two drafts of one product: 1.00 at temperature 0, 0.20 at 0.7, 0.01 at 1.5
- At 1.5, 89 of 100 drafts are no longer English only

#### [day4#eight-settings] The instruction moved the numbers more than any sampling setting

*What the lab measured*

| 100 drafts per setting | Words per draft | Over 20 words | Not English only | Break any rule |
| --- | --- | --- | --- | --- |
| Temperature 0 | 18.1 | 0.25 | 0.00 | 0.25 |
| Temperature 0.7 | 19.9 | 0.43 | 0.01 | 0.44 |
| Temperature 1.5 | 24.3 | 0.69 | 0.89 | 0.93 |
| Temperature 1.5, top_p 0.5 | 21.5 | 0.52 | 0.05 | 0.56 |
| Temperature 0.7, strict instruction | 11.2 | 0.01 | 0.02 | 0.05 |
| Temperature 1.5, strict instruction | 21.8 | 0.55 | 0.88 | 0.95 |

> **Two readings.** The instruction changes which tokens are likely; sampling only chooses among them. And an instruction cannot repair a distribution that temperature has flattened.

#### [day4#what-checks-miss] A rule that code can check is not the same as a draft that is true

*The limit of the table*

**Product:** coasters, check, blue, cotton, 35 x 150 cm

**Draft at 0.7 (weak example):** ... blue cotton check coasters measuring 35 inches x 150 cm ...

- The four checks count sentences, words, scripts and price words
- None of them reads the draft against the product
- 100 drafts of 20 English products say nothing about Arabic

> **Carry forward.** "Share of drafts that break no rule" is Wasf's quality column in Lab 4.4. Know what it does not measure before you defend it.

#### [day4#check-sampling] Which change does the table support?

*Checkpoint*

**Question:** Rayan runs Wasf at temperature 0.7 with the plain request, and 44 of 100 drafts break a rule. He wants fewer violations by tomorrow. Which change do the lab's measurements support first?

- A. Set the temperature to 0, so that the output is fixed and correct
- B. Raise top_p to 1.0, so that the model has more words to choose from
- C. Sample five drafts for each product and keep the longest one
- D. Keep temperature 0.7 and add the strict instruction with the four rules

#### [day4#lab-4-1] Four temperatures

Lab 4.1 · 25 min

Apply temperature and nucleus sampling to a real next-token distribution, then measure variation, length and rule violations of a small model across five samples at four temperatures.

[D4_1_four_temperatures.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D4_1_four_temperatures.ipynb)

- [ ] Temperature on a real distribution (5 min)
- [ ] Nucleus sampling by hand (4 min)
- [ ] Variation across five samples (6 min)
- [ ] Length, rule violations and one surprise (6 min)
- [ ] A live run of your own (4 min)
- [ ] ✱ Stretch: Draw 1,000 next tokens from your own distribution and compare the frequencies

### Session 2: From base models to assistants

#### [day4#s2] From base models to assistants

*Session 2*

- Show what a chat template adds to a prompt
- Compare a base and an instruction-tuned model on the same tasks
- Make a base model do a task with a few-shot prompt

#### [day4#base-model] A base model has learnt one thing: to continue text

*Where an assistant starts*

- **Pretraining**: predict the next token over a very large amount of text. No questions, no answers, no roles
- What comes out knows a great deal about language and the world as written
- It has no notion that a question is addressed **to it**

**Typed:** Write this in capital letters: wool rug

**Base model (weak example):** WoRlG rug

**Instruction-tuned (good example):** WOOL RUG

#### [day4#chat-template] You type the same prompt; the two models do not read the same text

*What each model receives*

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of Saudi Arabia? Answer with one word.<|im_end|>
<|im_start|>assistant
```

- The **chat template** wraps your prompt in turns marked by special tokens
- The instruction-tuned model continues this text, and the continuation of "the assistant's turn starts here" is an answer
- The base model receives the bare line

#### [day4#instruction-tuning] Instruction tuning is more training on conversations, with nothing added to the network

*Concept*

1. **Base model** next-token training on text
2. **Instruction tuning** the same training, on instructions with good answers, wrapped in the template
3. **Preference tuning** pushed toward the answers people rated better

- Same architecture, same number of weights: only their **values** change
- The model learns the turn structure, to answer the request, and to **end its turn**
- It is the fine-tuning of Day 2, applied to a language model

#### [day4#ten-tasks] On ten small tasks the base model fails, until the prompt is written for it

*What the lab measured*

| Ten tasks, temperature 0, at most 40 new tokens | Right | Stopped by itself | Clean |
| --- | --- | --- | --- |
| Instruction-tuned, plain prompt | 7 of 10 | 9 of 10 | 7 of 10 |
| Base, plain prompt | 3 of 10 | 7 of 10 | 2 of 10 |
| Base, few-shot prompt | 10 of 10 | 0 of 10 | 0 of 10 |

**Right**: the expected answer is in the first line. **Clean**: right, stopped, one line only, so a program can use it as it is.

#### [day4#few-shot-prompt] A few-shot prompt is a text whose most likely continuation is the answer

*Writing for a model that continues text*

```
Country: France
Capital: Paris

Country: Japan
Capital: Tokyo

Country: Saudi Arabia
Capital:
```

- Two solved examples in a fixed layout, then the new case with the answer left open
- The examples fix the **format** and show what kind of answer is wanted: three tasks the instruction-tuned model missed came out right
- Examples help instruction-tuned models too, whenever the shape of the answer matters

#### [day4#when-to-stop] Knowing the answer and knowing when to stop are learnt separately

*The other half of an answer*

- prompt ends: Country: Saudi Arabia / Capital:
- answer: Riyadh
- runs on: Country: South Korea / Capital: Seoul / Country: Thailand / Capital: Bangkok / Country: Turkey ...

- Generation ends when the model draws an **end token**, or at the limit of new tokens
- An instruction-tuned model learnt to end its turn; after two examples, a base model's likeliest next text is a third example
- The calling program sets a **stop sequence** or keeps the first line

#### [day4#preference-tuning] Preference tuning shapes how an assistant behaves, not what it knows

*Concept*

- People, or a model trained on their ratings, compare answers; the model is pushed toward the preferred ones
- This is where tone, helpfulness, refusals and "I'm sorry, but I don't have that information" come from
- It adds **no facts**: a learnt habit of declining is not knowledge of what the model knows

- **Why it matters this afternoon.** In Lab 4.3 the model declines six questions it cannot know and answers two with invented facts. Both behaviours come from training, and neither is a check.

#### [day4#reasoning-effort] Reasoning effort buys quality on hard tasks with tokens you pay for and wait for

*Concept*

- Some models write intermediate **reasoning tokens** before the answer; a setting says how much
- They are generated like any other token: one run of the model each
- They help on multi-step problems; a one-word extraction gains nothing

- **In the week's numbers.** The small model wrote about 3 tokens a second on a laptop processor. At that speed, 300 reasoning tokens before a 19-token product line are well over a minute of waiting.

> **Not measured here.** The lab's model has no reasoning setting. Treat effort as one more column of a model-choice table: quality gained against tokens spent.

#### [day4#check-base] How do you get an answer out of a base model?

*Checkpoint*

**Question:** Zaid sends "Which colour is this product? Answer with one word." to the base model at temperature 0 and gets nothing useful. He must keep this model. What should he change?

- A. Write two solved examples, leave the new answer open, and keep the first line of the output
- B. Lower the temperature further
- C. Put "You are a helpful assistant" in front of the question
- D. Raise the limit of new tokens

#### [day4#lab-4-2] Base against instruction-tuned

Lab 4.2 · 25 min

Compare a base model and an instruction-tuned model of the same size on ten tasks, then rescue the base model with a few-shot prompt.

[D4_2_base_and_instruct.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D4_2_base_and_instruct.ipynb)

- [ ] What each model receives (5 min)
- [ ] Ten tasks, scored by code (8 min)
- [ ] A few-shot prompt for the base model (7 min)
- [ ] Where the two still differ (5 min)
- [ ] ✱ Stretch: Write the few-shot prompt for an eleventh task

### Session 3: Limits: fabrication, cut-off and context

#### [day4#s3] Limits: fabrication, cut-off and context

*Session 3*

- Separate an abstention from a fabricated answer
- Show what a runtime does with a prompt longer than its window
- Write the limits of a generated output before a customer sees it

#### [day4#likely-is-not-true] The model produces a likely continuation; nothing in it checks that the text is true

*Fabrication*

**Question:** In which year was Al-Nawl Crafts Cooperative founded, and by whom?

**Small model (weak example):** Al-Nawl Crafts Cooperative was founded in 1998. The cooperative was established by the Al-Nawl family ... a prominent family in the textile industry of Iran.

- The cooperative does not exist: no model has read a word about it
- The answer is **fluent, specific and false**, and nothing in the text marks it as invented
- Asked in Arabic: the same year, founded by "a group of engineers in Egypt"

#### [day4#refuse-or-invent] Eight questions it cannot know: six declined, two answered anyway

*What the lab measured*

- **6** answers say "I don't have that information"
- **2** answers invent a year and founders
- **8** questions, one wording each: far too few for a rate

- Declining is a habit from tuning. It held for a phone number and a price, and failed for a founding year
- The lab's rule counts phrases: a polite refusal followed by invented "general information" passes it

> **What goes wrong.** "It refuses 75% of unanswerable questions" in a report. The lab shows that fabrication happens, not how often.

#### [day4#knowledge-cut-off] A model's present is where its training text ended

*The cut-off*

| Asked | The small model answered |
| --- | --- |
| What is today's date? | It cannot see the date, and says so |
| Which team won the football World Cup held in 2026? | It cannot "predict future events" |
| What is the most recent year you have information about? | "My last update was in 2023" |

- Its weights were published in September 2024. The year it gives for itself is **generated text**, like any other answer
- Anything newer than the cut-off, or private, has to arrive in the prompt

#### [day4#context-window] The context window counts everything: instructions, documents, question and answer

*The window*

- A fixed number of **tokens** the model can read at once; what is outside does not exist for it
- The output is written into the same window
- The model has a limit; the **runtime** is started with one too, often smaller, to save memory

- **In the lab.** The runtime's window is 2,048 tokens. The shop policy and a question are 164 tokens. The policy, 110 catalogue lines and the question are about 3,400.

#### [day4#silent-truncation] A prompt that did not fit lost its beginning, and nobody was told

*What the lab measured*

| Four questions on the shop policy | Tokens dropped | Right | Answers (expected 14, 10, 300, 2) |
| --- | --- | --- | --- |
| Policy, question: fits | 0 | 4 of 4 | 14 days, 10 working days, 300 riyals, 2 working days |
| Policy, long extract, question | 1,402 | 0 of 4 | 60 days, 4, 1, 3 |
| Long extract, policy, question | 1,402 | 4 of 4 | 14 days, 10, 300, 2 |

> **What goes wrong.** The runtime kept the end of the prompt and dropped the start, policy included. The model answered each question with a confident number.

#### [day4#count-before-sending] Count the tokens before you send, and fail loudly

*What to do about it*

- **Count.** Tokenise the prompt with the model's own tokeniser (Lab 3.1) and compare with the window minus the room for the answer.
- **Refuse or shorten on purpose.** An error is a fault you can see. If you shorten, you choose what goes, not the runtime.
- **Know your runtime.** Some return an error for a prompt that is too long; some cut silently, from the start or from the end. Test it once.

> **A silent cut is worse than a crash.** It gives a confident wrong answer that nobody can trace.

#### [day4#arabic-and-english] The same policy, the same questions: 4 of 4 in English, 3 of 4 in Arabic

*Two languages*

| Expected | English | Arabic |
| --- | --- | --- |
| 14 | 14 days | 14 يوما |
| 10 | 10 working days | 10 |
| 300 | 300 riyals | 25 ريال |
| 2 | 2 working days | 2 |

- The Arabic miss is another number from the same paragraph of the policy
- Four questions cannot put a figure on the gap; they can tell you to measure Arabic **by itself**
- Of the two invented founding stories, one was the Arabic question

> **Responsible use.** A model that serves one language worse treats its users unequally. Per-language results go in the README, as for search.

#### [day4#limits-note] Write the limits down, each with what is done about it

*Before a customer reads it*

| Measured limit | What is done about it |
| --- | --- |
| It invents facts it does not have | It is given the product's attributes and may state nothing else |
| It does not know the date or anything recent | It is never asked about dates, prices, stock or delivery |
| A prompt beyond the window loses its beginning | Prompts are counted first; one that does not fit is refused |
| Arabic answers were wrong more often | Arabic output gets its own test set |

> **And say what it is.** A draft written by a language model, read by the shop team before it is published.

#### [day4#check-limits] Where did "60 days" come from?

*Checkpoint*

**Question:** A script sends the shop policy, then 110 catalogue lines, then a question on returns, about 3,400 tokens in all, to a runtime with a window of 2,048 at temperature 0. The answer is "60 days"; the policy says 14. What happened?

- A. The policy was written after the model's training cut-off
- B. The sampling settings let an unlikely token through
- C. The runtime dropped the start of the prompt, the policy with it, and the model answered without it
- D. The model cannot read numbers reliably in a long text

#### [day4#lab-4-3] Probe the limits

Lab 4.3 · 25 min

Probe the model with questions it cannot know, questions beyond its cut-off and inputs beyond its context window, in Arabic and English, and record the failures.

[D4_3_probe_limits.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D4_3_probe_limits.ipynb)

- [ ] Questions it cannot know (7 min)
- [ ] Beyond the cut-off (4 min)
- [ ] Beyond the context window (7 min)
- [ ] Arabic against English (4 min)
- [ ] The limits note (3 min)
- [ ] ✱ Stretch: Put the missing fact in the prompt and measure again

### Session 4: Choosing a model family

#### [day4#s4] Choosing a model family

*Session 4*

- Match a model family to a task
- Apply licence and data constraints before comparing quality
- Justify a model choice with measured quality, size, latency and cost

#### [day4#families] The family follows from what goes in and what must come out

*Model families*

| Family | Reads | Produces | This week |
| --- | --- | --- | --- |
| Convolutional network | An image | A class or a feature vector | Naqsh: ResNet-18, 11.7 million weights |
| Encoder | A text | One vector, or a label | Khayt: multilingual-e5-small, 118 million |
| Decoder | A text | The next token, again and again | Wasf: Qwen2.5-0.5B, 494 million |
| Diffusion model | Noise and a text | An image | Not used |
| Vision-language model | Images and text | Text | Shortlisted for Naqsh, not measured |

> **Start from the task, not from the newest model.** A search needs vectors; a classifier needs a label; only drafting needs generation.

#### [day4#diffusion-models] A diffusion model makes an image by removing noise, step by step

*Concept*

1. **Start** an image of pure noise
2. **Denoise** a network predicts the noise; a little is removed
3. **Repeat** tens of steps, each guided by the text
4. **Image** a new picture that matches the text

- Trained on images with added noise: the task is to predict the noise, a regression loss
- One image costs **many runs** of the network, as one sentence costs many runs of a decoder
- For the cooperative: catalogue illustrations, never a photograph of a product that exists

#### [day4#vision-language-models] A vision-language model turns image patches into tokens a decoder can read

*Concept*

- An **image encoder** cuts the picture into patches and gives a vector per patch
- The vectors enter the language model beside the text tokens; attention does the rest
- Image encoders of this kind are trained on pictures with their captions: matching pairs pulled together, as for yesterday's sentence embeddings

- **For Naqsh.** A hosted vision-language model could be asked "which pattern is this?" with no training at all. Nobody has measured it on the cooperative's photographs, and the photographs would leave the shop's server.

#### [day4#open-weight-or-hosted] Open-weight or hosted: who runs it, and where the data goes

*Where the model runs*

- **Open-weight, on your server.** You download the weights and run them. The data stays. The version never changes unless you change it. You pay for the machine and the work of running it, and the licence is yours to read.
- **Hosted, behind an interface.** The provider runs it. Much larger models than your server holds. You pay per token. The text or image leaves your server, and the provider updates and retires versions on its own calendar.

> **In the case.** Anything that leaves the shop's server needs the sign-off of Bushra Al-Aqeel, the data protection officer, before a single test request.

#### [day4#size-and-latency] Size and latency span four orders of magnitude across the three models

*What the week measured*

| One request on a laptop processor | File | Latency |
| --- | --- | --- |
| Small convolutional network (Lab 2.1) | 0.1 MB | 1.9 ms |
| ResNet-18, one photograph | 45 MB | 35 ms |
| multilingual-e5-small, one query and the search | 470 MB | 39 ms |
| Qwen2.5-0.5B, one product line of 19 tokens | 990 MB | 6,500 ms |

- An encoder runs **once** per request; a decoder runs once per token written
- A latency budget belongs to the use: a person waiting for a search, or a batch of 2,000 drafts overnight

#### [day4#licences-and-provenance] Read the licence of each size, and ask what the model was trained on

*May we use it?*

| Model | Licence of the weights | For a shop |
| --- | --- | --- |
| multilingual-e5-small | MIT | Yes |
| Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct | Apache 2.0 | Yes |
| Qwen2.5-3B-Instruct | A research licence of its own | No |
| ResNet-18, ImageNet weights | BSD 3-Clause | Sign-off: trained on ImageNet, whose terms of access name non-commercial research and education |

> **What goes wrong.** "Same family, same licence." One size of a family can carry other terms than its smaller and larger siblings.

#### [day4#screen-then-compare] Screen first, compare second, then write two lines

*The method*

1. **1 · Screen** licence, data leaving the server, size, measured at all, inside the latency budget
2. **2 · Compare** quality against the simpler alternative, latency, size, cost
3. **3 · Justify** "because": the numbers that decide. "But": the risk, and who signs

- A model that may not be used cannot be chosen, so quality comes **after** the screen
- "Not measured" removes an option from this week's table. It is not a verdict on the model
- Cost: a model on your own server adds nothing per request; a hosted one is paid per token

#### [day4#what-decides] In each use case something other than the top score decides

*The table you fill*

| Use case | Eligible options and their quality | What decides |
| --- | --- | --- |
| **Naqsh** · test accuracy | From scratch 0.476 · frozen features 0.774 · fine-tuned 0.788 | 14 photographs in 1,000 separate the best two |
| **Khayt** · recall at 5 | Keywords 0.175 · embeddings 0.225 · paired by product 0.335 | The best row still misses two searches in three |
| **Wasf** · drafts breaking no rule | Template 1.00 · small model 0.56 · with strict instruction 0.95 | The measure does not see that every template line reads alike |

> **Keep the row without a model.** It is the simpler alternative the rubric asks for.

#### [day4#check-choice] What is the first reply?

*Checkpoint*

**Question:** A colleague proposes Qwen2.5-3B-Instruct for Wasf: "six times the weights, same family, surely better lines". What do you say first?

- A. "Let us measure its drafts on the 20 products, then decide"
- B. "Licence first: this size has a research licence, unlike its siblings, so a shop may not use it"
- C. "Same family as our Apache 2.0 model, so only the file size needs checking"
- D. "A larger model is too slow for us, so there is no need to look further"

#### [day4#lab-4-4] The model-choice table

Lab 4.4 · 25 min

Fill a model-choice table for three cooperative use cases from supplied lab measurements of latency, size and quality and each model's licence, then choose the capstone task.

[D4_4_model_choice.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D4_4_model_choice.ipynb)

- [ ] Read the shortlist and the measurements (4 min)
- [ ] Licence and data constraints first (5 min)
- [ ] Quality, size, latency and cost per use case (8 min)
- [ ] Two lines of justification each (5 min)
- [ ] Choose the capstone task (3 min)
- [ ] ✱ Stretch: Change one constraint and see which choice flips

#### [day4#capstone-brief] One small model for one use case, built alone and defended live

*Tomorrow*

- **Naqsh.** An image classifier for the eight patterns, against the small network trained from scratch.
- **Khayt.** A bilingual search over the catalogue, against keyword search.
- **Wasf.** Constrained text generation for product lines, against a sentence template.

- A **public repository**: code, training logs, commit history, and a README from the supplied template
- The README reports curves, test results, the simpler alternative, latency on a laptop, and every licence
- Eight minutes of defence, four of questions; you change **one hyperparameter live** and explain what moves

#### [day4#wrap-up] Three things to keep

*Day 4 · wrap-up*

- **Generation is a draw.** Scores, then one token chosen. Temperature and the nucleus shape the draw; the instruction shapes the scores, and did more: 0.44 to 0.05.
- **Likely is not true.** A founding year for a cooperative that does not exist; "60 days" from a policy the model never saw. Write the limits before a customer reads the output.
- **Screen, compare, justify.** Licence and data first, then measured quality against the simpler alternative, then two lines you can defend.

#### [day4#next] Capstone day

*Next*

Build your model, write the README, and defend one curve or one failure the assessor chooses.
