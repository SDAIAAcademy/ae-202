# Deep Learning and Generative AI Fundamentals: labs

Hands-on notebooks for the SDAIA Academy course **AE-202 Deep Learning and Generative AI Fundamentals (أساسيات التعلّم العميق والذكاء الاصطناعي التوليدي)**. Every lab works on **Al-Nawl Crafts Cooperative**, a fictional cooperative of weavers and basket makers with one online shop, and on its three small models: **Naqsh (نقش)**, which names the pattern of a textile from a photograph; **Khayt (خيط)**, the search over the bilingual catalogue; and **Wasf (وصف)**, a small open-weight language model that drafts one product line. Each lab starts from supplied files, so none depends on an earlier one.

Everything runs on a **laptop processor**, or in a Google Colab runtime. No key is needed and no language model is downloaded to a laptop: Day 3 reads supplied vectors of the embedding model and Day 4 reads supplied logged runs of the two language models. The one model file a laptop fetches is the pretrained image model of Day 2, about 45 MB.

## Quick start

**a · Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/ae-202/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/<notebook>.ipynb`.
2. Run the first cell. It clones this repository and installs `requirements-colab.txt`.

**b · Local**
```bash
git clone https://github.com/SDAIAAcademy/ae-202
cd ae-202/labs
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version; PyTorch for a processor)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1: seven check marks mean that every lab will run. The textile images are drawn on first use (about 20 seconds, nothing is downloaded) into `cache/`.

## Settings

| Setting (environment variable) | Default | What it does |
|---|---|---|
| `AE202_QUICK` | not set | `1` runs every lab on less data and fewer epochs, in seconds |
| `AE202_MODELS` | not set | `load` loads the embedding model (about 470 MB) on a laptop, for the live part of Labs 3.3 and 3.4; in Colab it loads by itself |
| `AE202_LLM` | not set | `runtime` loads the language model (about 1 GB) in the notebook, which Colab does by itself; `hosted` uses the endpoint below |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | not set | A free hosted endpoint that serves the OpenAI Chat Completions shape, and a small open-weight model it lists as free, for the live cells of Day 4 |
| `AE202_LLM_INSTRUCT`, `AE202_LLM_BASE`, `AE202_EMBED_MODEL` | Qwen2.5-0.5B-Instruct, Qwen2.5-0.5B, multilingual-e5-small | Other models of the same kind; the supplied vectors and logged runs were made with the defaults |

A live language model is optional everywhere in the labs: a cell that needs one says that it was skipped when there is none. The capstone's text-generation task is the one place that needs a live model (Colab, or a free hosted endpoint).

## Notebooks

| Day | Session | Notebook | Runs on |
|---|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/00_setup_check.ipynb) | processor |
| 0 | Pre-work: the array-programming refresher | [`00_prework_arrays.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/00_prework_arrays.ipynb) | processor, NumPy only |
| 1 | Gradient by hand and by autograd | [`D1_1_gradient_by_hand.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_1_gradient_by_hand.ipynb) | processor |
| 1 | The training loop (first attempt without AI assistance) | [`D1_2_training_loop.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_2_training_loop.ipynb) | processor |
| 1 | Three learning rates, with and without dropout | [`D1_3_learning_rates_dropout.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_3_learning_rates_dropout.ipynb) | processor |
| 1 | Three faults in a broken script | [`D1_4_three_faults.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_4_three_faults.ipynb) | processor |
| 2 | A small convolutional network and its filters | [`D2_1_small_cnn.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_1_small_cnn.ipynb) | processor |
| 2 | Frozen features and a linear classifier | [`D2_2_frozen_features.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_2_frozen_features.ipynb) | processor; fetches ResNet-18 (45 MB) |
| 2 | Fine-tune the last block on 800 images | [`D2_3_fine_tune.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_3_fine_tune.ipynb) | processor; a graphics processor is optional |
| 2 | Confusion matrix, confident errors, near-duplicates | [`D2_4_errors_duplicates.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_4_errors_duplicates.ipynb) | processor |
| 3 | Three tokenisers on Arabic and English | [`D3_1_three_tokenisers.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_1_three_tokenisers.ipynb) | processor; fetches three vocabulary files |
| 3 | Attention by hand (first attempt without AI assistance) | [`D3_2_attention.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_2_attention.ipynb) | processor |
| 3 | Catalogue search and recall at 5 | [`D3_3_catalogue_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_3_catalogue_search.ipynb) | supplied vectors |
| 3 | Failures by language | [`D3_4_failures_by_language.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_4_failures_by_language.ipynb) | supplied vectors |
| 4 | Four temperatures | [`D4_1_four_temperatures.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_1_four_temperatures.ipynb) | supplied logged runs |
| 4 | Base against instruction-tuned | [`D4_2_base_and_instruct.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_2_base_and_instruct.ipynb) | supplied logged runs |
| 4 | Probe the limits | [`D4_3_probe_limits.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_3_probe_limits.ipynb) | supplied logged runs |
| 4 | The model-choice table | [`D4_4_model_choice.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_4_model_choice.ipynb) | supplied measurements |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in. Each one is followed by a check with a safety net, so the cells below it still run when an exercise is not done. Complete reference versions are in `solutions/`. Cells marked **✱ Stretch** are optional.

The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py`), which also writes the solutions. Edit the sources, not the notebooks.

What your own runs write goes to `runs/`, and the drawn images to `cache/`; neither is committed. Delete both before running the course's `academy check`, which reports them as files under `labs/`.

## The case study: Al-Nawl Crafts Cooperative

Synthetic and deterministic (fixed seeds); no real organisation and no personal data.

| Data | File | Contents |
|---|---|---|
| Textile photographs | `cache/textiles.npz`, drawn by `prepare_data.py` | 8,000 images of 48 by 48 pixels in eight pattern classes (stripe, chevron, diamond, check, triangle, dot, wave, star): 6,000 to train, 1,000 to validate, 1,000 to test, and the 800 labelled images of Day 2 (100 per class) |
| Swatches | `cache/swatches.npz` | 4,000 grey swatches of 24 by 24 pixels laid straight, for the small fully connected network of Day 1 |
| Catalogue | `data/descriptions.jsonl` | 4,000 descriptions: 2,000 products, each described in Arabic and in English |
| Queries | `data/queries.jsonl` | 50 labelled queries (20 English, 20 Arabic, 10 mixed) with synonyms, dialect, spelling variants, Arabizi and code-switching |
| Supplied vectors | `data/embeddings/` | every description and query embedded with multilingual-e5-small (384 numbers each), written by `_tools/make_embeddings.py` |
| Logged runs | `data/runs/d4_*.jsonl` | 858 generations of Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct, **made by running the models** on a laptop processor with `_tools/make_runs.py`; one saved next-token distribution |
| Day 2 results | `data/runs/day2_*`, `data/checkpoints/` | the supplied results of training from scratch and of frozen features, the fine-tuned model's test probabilities, a warm start for Lab 2.1 |
| Shortlist and measurements | `data/shortlist.json`, `data/measurements.json` | the models of Lab 4.4 with size, licence and provenance; quality from the labs' reference results and latencies measured one request at a time (`_tools/make_measurements.py`). The hosted entries are generic and their prices are an exercise price list |
| Reference results | `data/reference/` | what each solution printed in a full run; the slides quote these numbers |
| A broken script | `broken/train_swatches.py` | the training script of Lab 1.4, with three seeded faults |

`nawl.py` (imported as `U`) holds the helpers: data loading, the small networks, `fit`, `evaluate`, the plots, the pretrained backbone, `embed`, the keyword search, `generate` and `try_live`. `llm_tasks.py` (imported as `T`) holds the tasks of Day 4: the product-line request and its four rules, the ten small tasks, the probing questions and the shop policy in both languages.

Planted problems, for the trainer:

- Lab 1.4: the loss receives probabilities where it expects scores; the learning rate is far too high; the data is never shuffled and is sorted by class.
- Lab 2.4: 40 test photographs are near-duplicates of training photographs, and 12 test labels are wrong; the key is `cache/planted_key.json`, written with the images.
- Labs 3.3 and 3.4: the search never returns a description in the other language than the query's; the queries hold dialect words, Arabizi and synonyms that the catalogue does not use (`solutions/queries_key.json`).
- Lab 4.3: the cooperative does not exist, so every fact a model states about it is invented; two prompts are longer than the runtime's window of 2,048 tokens.
- Lab 4.4: one model of the shortlist has a research licence, unlike the smaller and the larger model of its family; one is too large for the shop's server; the pretrained image model was trained on photographs collected for non-commercial research.

## Pre-work, capstone and follow-up

| What | Where | When |
|---|---|---|
| Pre-work: a two-hour array-programming refresher, checked by the entry quiz | `00_prework_arrays.ipynb` | before Day 1 |
| Capstone: the brief, the rubric, the README template and a starter | `capstone/` (`uv run python capstone/starter.py naqsh`, `khayt` or `wasf`) | the capstone day |
| Follow-up: the capstone model re-run on a second dataset | `followup/` (`uv run python capstone/starter.py naqsh --second`) | within two weeks |

**Windows and long folder paths.** If an import fails with "The filename or extension is too long", keep the virtual environment in a short folder: set `UV_PROJECT_ENVIRONMENT` to a short path (for example a folder at the root of your drive) before `uv sync`.
