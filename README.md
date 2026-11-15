<!-- academy:title -->
# Deep Learning and Generative AI Fundamentals

أساسيات التعلم العميق والذكاء الاصطناعي التوليدي

AI Engineering Track · SDAIA Academy · AE-202 · Level 200

REPLACE: one line under the title: what participants can do after the course

Participants go from tensors to a working generative model and learn what each layer of the stack does. For a fictional crafts cooperative they write a training loop by hand, debug a network that will not learn, adapt a pretrained image model to synthetic textile images, implement attention for a toy sequence, build bilingual semantic search with sentence embeddings, and generate text with a small open-weight model while varying sampling settings. They finish by comparing model families on quality, size, latency, licence and cost for three of the cooperative's use cases.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-202/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Explain how a transformer turns input tokens into next-token probabilities, naming the role of embeddings, attention, the context window and sampling, given an annotated architecture diagram with five blanks and two generated outputs | Awareness | a completed diagram and a short written explanation scored against a rubric in the supervised check |
| LO2 | Implement a training loop with a loss, an optimiser, mini-batches and validation checks, and use it to train a small network, given a tensor library, synthetic image tensors and a starter script with no training code | Applied | a script whose loss and validation curves match the reference behaviour within a stated tolerance |
| LO3 | Apply a debugging checklist to a network whose loss does not fall and correct each fault found, given a training script with three seeded faults in the loss input, the learning rate and data shuffling | Applied | a log showing the overfit-one-batch test, each fault and the corrected learning curve |
| LO4 | Use transfer learning to adapt a pretrained image model to a new label set with little data, and measure the improvement over training from scratch, given 800 labelled textile images in eight classes and a laptop CPU time budget | Applied | a results table comparing frozen features, partial fine-tuning and training from scratch on the same test set |
| LO5 | Use sentence embeddings to build and test a bilingual semantic search over product descriptions, given 4,000 Arabic and English descriptions and 50 labelled queries | Applied | recall at 5 for Arabic, English and mixed queries, with three failure cases and a likely cause for each |
| LO6 | Measure how temperature, nucleus sampling and the instruction change the outputs of a small open-weight language model, given a locally run model and 20 fixed prompts sampled five times each | Applied | a table of variation, length and rule violations per setting, with an explanation of one surprising result |
| LO7 | Evaluate model families for three stated use cases on quality, size, latency, licence terms and cost, and recommend one per use case, given a supplied shortlist of open-weight and hosted models and results measured in the labs | Performance | a model-choice table with a two-line justification per use case, defended in the capstone |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Tensors and training | Tensors, shapes and gradients; Writing the training loop; Learning rates, batches and regularisation; When training goes wrong | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Vision and transfer | Convolutions and what they detect; Pretrained backbones; Fine-tuning with little data; Evaluating a deep model | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Transformers and embeddings | Tokens, including Arabic; Attention step by step; Sentence embeddings; Multilingual search and its failures | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Generation and model choice | Next-token prediction and sampling; From base models to assistants; Limits: fabrication, cut-off and context; Choosing a model family | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Compute a gradient by hand and with automatic differentiation for a two-layer network and compare the numbers | [`D1_1_compute_gradient_hand.ipynb`](labs/D1_1_compute_gradient_hand.ipynb) |
| 1.2 | Day 1 · Session 2 | Write a training loop with batching, a loss and an optimiser for the textile images and plot the loss curves | [`D1_2_write_training_loop.ipynb`](labs/D1_2_write_training_loop.ipynb) |
| 1.3 | Day 1 · Session 3 | Run the same model at three learning rates, with and without dropout, and read the curves | [`D1_3_run_same_model.ipynb`](labs/D1_3_run_same_model.ipynb) |
| 1.4 | Day 1 · Session 4 | Find three seeded faults in a broken script, starting with an overfit-one-batch test | [`D1_4_find_three_seeded.ipynb`](labs/D1_4_find_three_seeded.ipynb) |
| 2.1 | Day 2 · Session 1 | Train a small convolutional network and visualise its first-layer filters | [`D2_1_train_small_convolutional.ipynb`](labs/D2_1_train_small_convolutional.ipynb) |
| 2.2 | Day 2 · Session 2 | Extract features with a pretrained image model and train a linear classifier on them | [`D2_2_extract_features_pretrained.ipynb`](labs/D2_2_extract_features_pretrained.ipynb) |
| 2.3 | Day 2 · Session 3 | Fine-tune the last blocks with augmentation on 800 images and compare with supplied frozen-feature results | [`D2_3_fine_tune_last.ipynb`](labs/D2_3_fine_tune_last.ipynb) |
| 2.4 | Day 2 · Session 4 | Build a confusion matrix, inspect the ten most confident errors and check the splits for near-duplicates | [`D2_4_build_confusion_matrix.ipynb`](labs/D2_4_build_confusion_matrix.ipynb) |
| 3.1 | Day 3 · Session 1 | Tokenise the same Arabic and English sentences with three tokenisers and compare token counts | [`D3_1_tokenise_same_arabic.ipynb`](labs/D3_1_tokenise_same_arabic.ipynb) |
| 3.2 | Day 3 · Session 2 | Implement scaled dot-product attention for a toy sequence and check it against the library function | [`D3_2_implement_scaled_dot.ipynb`](labs/D3_2_implement_scaled_dot.ipynb) |
| 3.3 | Day 3 · Session 3 | Embed the product descriptions and measure recall at 5 on 50 labelled queries | [`D3_3_embed_product_descriptions.ipynb`](labs/D3_3_embed_product_descriptions.ipynb) |
| 3.4 | Day 3 · Session 4 | Compare Arabic, English and mixed queries and collect three failure cases with a likely cause | [`D3_4_compare_arabic_english.ipynb`](labs/D3_4_compare_arabic_english.ipynb) |
| 4.1 | Day 4 · Session 1 | Generate from a small local model at four temperatures and measure variation across five samples | [`D4_1_generate_from_small.ipynb`](labs/D4_1_generate_from_small.ipynb) |
| 4.2 | Day 4 · Session 2 | Compare a base model and an instruction-tuned model of the same size on ten tasks | [`D4_2_compare_base_model.ipynb`](labs/D4_2_compare_base_model.ipynb) |
| 4.3 | Day 4 · Session 3 | Probe the model with questions beyond its training cut-off and inputs beyond its context window and record the failures | [`D4_3_probe_model_questions.ipynb`](labs/D4_3_probe_model_questions.ipynb) |
| 4.4 | Day 4 · Session 4 | Fill a model-choice table for three cooperative use cases from supplied lab measurements of latency, size and quality and each model's licence, then choose the capstone task | [`D4_4_fill_model_choice.ipynb`](labs/D4_4_fill_model_choice.ipynb) |
<!-- /academy:labs -->

## Capstone

A small model for one cooperative use case (image classifier, bilingual search or constrained text generation), submitted as a public repository whose README reports training curves, test results, a comparison with a simpler alternative, latency on a laptop and the licence of every model and dataset used, followed by an eight-minute defence and four minutes of questions

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- AE-201 Machine Learning Foundations and Model Evaluation
- FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability
- Multiplies matrices and computes a derivative with the chain rule
- Has trained and evaluated a model with a train, validation and test split

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Tensor library with automatic differentiation:** PyTorch, JAX
- **Pretrained model loading:** Transformers, timm, Sentence Transformers
- **Local runtime for small open-weight language models:** llama.cpp, Ollama, a free hosted open-weight endpoint
- **Vector similarity search:** FAISS, brute-force search in NumPy
- **Experiment tracking:** MLflow, TensorBoard
- **Environment and notebook:** uv with JupyterLab, marimo, a free hosted notebook service
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA Generative AI Guidelines: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
