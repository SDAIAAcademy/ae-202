---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.2.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/slides/capstone.html"
---

# Deep Learning and Generative AI Fundamentals

## Capstone: Capstone: a small model for the cooperative, defended live

An image classifier, a bilingual search or a constrained text generator, with its curves, its comparison, its latency and its licences

#### [capstone#title] Capstone: a small model for the cooperative, defended live

*SDAIA Academy · Deep Learning and Generative AI Fundamentals*

Capstone: An image classifier, a bilingual search or a constrained text generator, with its curves, its comparison, its latency and its licences

AI Engineering Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-202

#### [capstone#agenda] Check, build, publish, defend

*Capstone day*

- **The supervised exit check**: one hour, on paper, without AI assistance
- **Build and measure**: your model, the simpler alternative, the same test for both
- **Publish**: the public repository, with logs and a README from the template
- **Defend**: eight minutes each, then four minutes of questions with one live change

#### [capstone#exit-check] The supervised check: four things you can do with a pen

*20% of the result*

- **Shapes and gradients by hand**: a small network, one step of the chain rule
- **Reading a learning curve**: what the curve says, and what you would change
- **The transformer diagram**: five blanks and two generated outputs to explain
- **Sampling settings**: what temperature and the nucleus do to an output

- **Conditions.** Proctored, on paper, no AI assistance, no notes. To pass the course: at least 60% on this check, at least 60% on the capstone rubric, and 70% overall.

#### [capstone#the-brief] One small model for one use case of the cooperative

*What you build*

| Task | You build | The simpler alternative | Measured on |
| --- | --- | --- | --- |
| **Naqsh** | A classifier of the eight patterns from the 800 labelled photographs | The small network trained from scratch | The 1,000 test photographs, near-duplicates removed |
| **Khayt** | A search over the 4,000 descriptions | Keyword search | Recall at 5 per language: the 50 queries and 20 of your own |
| **Wasf** | One product line of at most 20 words from a small open-weight model | A sentence template | Rule violations and variation on 20 products, and 20 drafts you read yourself |

> **Alone, with any assistant.** AI assistance is fully allowed. The commit history, the logs and the defence show that the work is yours.

#### [capstone#readme-reports] Five things, each with a number and how it was measured

*What the README reports*

- **Training curves or runs.** At least two measured runs, and what you read from each.
- **Test results.** Per class or per language, on a test you can call fair, with three failures and a likely cause each.
- **The simpler alternative.** The same test for both. The difference in items as well as in points.

- **Latency and size.** Median time of one request on a laptop processor, and the size of the model file.
- **Licences and limits.** The licence and provenance of every model and dataset, with its address, and the limits written for the person who reads the output.

#### [capstone#rubric] Five criteria, 100 points

*How the repository is scored*

| Item | Value |
| --- | --- |
| Working model and training | 25 points |
| Evaluation against a simpler alternative | 25 points |
| Measurement of latency and size | 15 points |
| Licence and limits | 15 points |
| Reproducibility | 20 points |

> **The result.** The repository is 60%, the live defence 20%, the supervised check 20%.

#### [capstone#format] 8 minutes to present, 4 to answer, one change made live

*The defence*

- **1 · The task and the model.** Which use case, what you built, one decision you rejected.
- **2 · What you measured.** Against the simpler alternative, on the same test. Latency, size, licences.
- **3 · Where it fails.** Three failures with a likely cause, and the limits you wrote.

> **In the questions.** The assessor picks one curve or one failure for you to explain, and names one hyperparameter. Say what will happen, change it, run, and explain what came out.

#### [capstone#running-order] Who defends when

*Running order*

| # | Participant | Task and repository |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Clone it clean · 15 minutes

*Before the first defence*

Clone your public repository into a new folder and run one command from its README. If that fails here, it fails for the reviewer.

#### [capstone#now-presenting] Defence

*Now defending*

##### Questions and the live change

#### [capstone#review] What the room learned

*After the defences*

- **The strongest comparison.** Whose simpler alternative was hardest to beat, and what did that show?
- **The most common gap.** Which section of the README was thinnest across the room?
- **One surprise.** Which live change did not do what its author predicted?

#### [capstone#what-next] The follow-up task, and where this leads

*After the course*

- **Follow-up task, within two weeks.** Run your capstone model again on a second dataset, unchanged first, then with one change, and add both tables to your README
- **Write your expectation before you run.** The commit history shows the order
- **Next in the track.** AE-203 builds applications on language models: prompts, structured output, grounded answers. AE-205 serves and monitors a model
- **Later, for depth.** AE-301 for language engineering and tuning; AE-309 for vision

#### [capstone#thanks] Thank you

*SDAIA Academy*

Deep Learning and Generative AI Fundamentals
