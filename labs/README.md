# Deep Learning and Generative AI Fundamentals: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Tensors, shapes and gradients | [`D1_1_compute_gradient_hand.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_1_compute_gradient_hand.ipynb) |
| Day 1 | Session 2 · Writing the training loop | [`D1_2_write_training_loop.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_2_write_training_loop.ipynb) |
| Day 1 | Session 3 · Learning rates, batches and regularisation | [`D1_3_run_same_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_3_run_same_model.ipynb) |
| Day 1 | Session 4 · When training goes wrong | [`D1_4_find_three_seeded.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D1_4_find_three_seeded.ipynb) |
| Day 2 | Session 1 · Convolutions and what they detect | [`D2_1_train_small_convolutional.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_1_train_small_convolutional.ipynb) |
| Day 2 | Session 2 · Pretrained backbones | [`D2_2_extract_features_pretrained.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_2_extract_features_pretrained.ipynb) |
| Day 2 | Session 3 · Fine-tuning with little data | [`D2_3_fine_tune_last.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_3_fine_tune_last.ipynb) |
| Day 2 | Session 4 · Evaluating a deep model | [`D2_4_build_confusion_matrix.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D2_4_build_confusion_matrix.ipynb) |
| Day 3 | Session 1 · Tokens, including Arabic | [`D3_1_tokenise_same_arabic.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_1_tokenise_same_arabic.ipynb) |
| Day 3 | Session 2 · Attention step by step | [`D3_2_implement_scaled_dot.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_2_implement_scaled_dot.ipynb) |
| Day 3 | Session 3 · Sentence embeddings | [`D3_3_embed_product_descriptions.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_3_embed_product_descriptions.ipynb) |
| Day 3 | Session 4 · Multilingual search and its failures | [`D3_4_compare_arabic_english.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D3_4_compare_arabic_english.ipynb) |
| Day 4 | Session 1 · Next-token prediction and sampling | [`D4_1_generate_from_small.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_1_generate_from_small.ipynb) |
| Day 4 | Session 2 · From base models to assistants | [`D4_2_compare_base_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_2_compare_base_model.ipynb) |
| Day 4 | Session 3 · Limits: fabrication, cut-off and context | [`D4_3_probe_model_questions.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_3_probe_model_questions.ipynb) |
| Day 4 | Session 4 · Choosing a model family | [`D4_4_fill_model_choice.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-202/blob/main/labs/D4_4_fill_model_choice.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
