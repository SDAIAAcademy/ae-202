# Capstone: a small model for the cooperative, defended live

You build, measure and defend one small model for Al-Nawl Crafts Cooperative, alone. Eight minutes to
present, four minutes of questions. In the questions the assessor names one hyperparameter; you say what the
curve or the table will do, change it in front of the room, and explain what came out.

## Choose a task

You chose in Lab 4.4. The three tasks use the data of the labs, and each one names the simpler alternative
that your model must be compared with.

| Task | You build | The simpler alternative | Measured on |
|---|---|---|---|
| Naqsh | A classifier of textile patterns from the 800 labelled photographs | The small convolutional network trained from scratch | The 1,000 test photographs, with the near-duplicates removed and the wrong labels reported |
| Khayt | A search over the 4,000 Arabic and English descriptions | Keyword search | Recall at 5, for English, Arabic and mixed queries apart: the 50 labelled queries, and 20 queries that you write and label before you run them |
| Wasf | One product line of at most 20 words from a product's attributes, by a small open-weight language model | A sentence template filled with the attributes | Rule violations, length and variation on 20 products with five samples each, and 20 drafts that you read and judge yourself |

Wasf needs a live model: Colab, or a free hosted endpoint for a small open-weight model. Naqsh and Khayt run
on a laptop processor.

`starter.py` loads the data of a task and prints what the first hour needs:

```
uv run python capstone/starter.py naqsh
```

## What you submit

A link to a **public repository**, your own work, with:

- the code, a lock file and fixed seeds, so that a reviewer gets your numbers from a clean clone;
- the training logs or run logs, and a commit history that shows the work as it happened;
- a `README.md` written from `TEMPLATE_README.md`, with every section filled.

## How it is scored

| Criterion | Weight | Full marks look like |
|---|---|---|
| Working model and training | 25 | The model runs from a clean clone. For a trained model: training and validation curves from at least two runs, read correctly. For a search or a generator: the settings tried, with a log of each run |
| Evaluation against a simpler alternative | 25 | The same test data for both; the difference stated in items (photographs, queries, drafts) as well as in points; results per class or per language; three failures with a likely cause each |
| Measurement of latency and size | 15 | Median time of one request on a laptop processor, the size of the model file, and how both were measured |
| Licence and limits | 15 | The licence and the provenance of every model and dataset used, with the address of the source; what the output may be used for; the limits you measured, written for the person who will read the output |
| Reproducibility | 20 | One command per number; seeds fixed; the lock file committed; the reviewer's numbers match yours within the tolerance you state |

The capstone repository is 60% of the grade, the live defence 20%, and the supervised check 20%. To pass:
70% overall, with at least 60% on this rubric and at least 60% on the supervised check.

## AI policy

AI-full: use any assistant. You defend every number yourself, your commit history and your logs are part of
the evidence, and the change made in the defence is yours alone.

## Responsible use

Every model and dataset goes through the licence and provenance check of Lab 4.4, and the result is in your
README. Text written by a model is shown as a draft with its limits note (Lab 4.3) and is read by a person
before a customer sees it. Report results for Arabic and for English apart. The cooperative and its data are
invented; if you add data of your own, it holds nothing about a real person.

## After the capstone day

The follow-up task is in [`../followup/`](../followup/): within two weeks, run your capstone model again on a
second dataset and report what changed.
