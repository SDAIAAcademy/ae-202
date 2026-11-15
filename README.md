<!-- academy:title -->
# Deep Learning and Generative AI Fundamentals

أساسيات التعلم العميق والذكاء الاصطناعي التوليدي

AI Engineering Track · SDAIA Academy · AE-202 · Level 200

Train, debug, adapt and measure small deep learning and generative models, and choose a model family with evidence

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
| Capstone | Capstone: a small model for the cooperative, defended live |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Al-Nawl Crafts Cooperative** (جمعية النَّول التعاونية للحِرف) is a fictional cooperative of weavers that sells textiles and baskets through an online shop. Its three small models carry the course: Naqsh names the pattern of a textile photograph, Khayt searches the catalogue in Arabic and English, and Wasf drafts one line of text for a product.

The cooperative, its people and all of its data are invented for this course. The 8,000 textile images are drawn by `labs/prepare_data.py`, the 4,000 product descriptions and the 50 labelled queries are generated by the same script, and nothing describes a real organisation, product or person.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | A gradient by hand and by autograd | [`D1_1_gradient_by_hand.ipynb`](labs/D1_1_gradient_by_hand.ipynb) |
| 1.2 | Day 1 · Session 2 | The training loop | [`D1_2_training_loop.ipynb`](labs/D1_2_training_loop.ipynb) |
| 1.3 | Day 1 · Session 3 | Three learning rates and dropout | [`D1_3_learning_rates_dropout.ipynb`](labs/D1_3_learning_rates_dropout.ipynb) |
| 1.4 | Day 1 · Session 4 | Three faults in a training script | [`D1_4_three_faults.ipynb`](labs/D1_4_three_faults.ipynb) |
| 2.1 | Day 2 · Session 1 | A small convolutional network | [`D2_1_small_cnn.ipynb`](labs/D2_1_small_cnn.ipynb) |
| 2.2 | Day 2 · Session 2 | Frozen features | [`D2_2_frozen_features.ipynb`](labs/D2_2_frozen_features.ipynb) |
| 2.3 | Day 2 · Session 3 | Fine-tune the last block | [`D2_3_fine_tune.ipynb`](labs/D2_3_fine_tune.ipynb) |
| 2.4 | Day 2 · Session 4 | Errors and near-duplicates | [`D2_4_errors_duplicates.ipynb`](labs/D2_4_errors_duplicates.ipynb) |
| 3.1 | Day 3 · Session 1 | Three tokenisers | [`D3_1_three_tokenisers.ipynb`](labs/D3_1_three_tokenisers.ipynb) |
| 3.2 | Day 3 · Session 2 | Attention by hand | [`D3_2_attention.ipynb`](labs/D3_2_attention.ipynb) |
| 3.3 | Day 3 · Session 3 | Catalogue search | [`D3_3_catalogue_search.ipynb`](labs/D3_3_catalogue_search.ipynb) |
| 3.4 | Day 3 · Session 4 | Failures by language | [`D3_4_failures_by_language.ipynb`](labs/D3_4_failures_by_language.ipynb) |
| 4.1 | Day 4 · Session 1 | Four temperatures | [`D4_1_four_temperatures.ipynb`](labs/D4_1_four_temperatures.ipynb) |
| 4.2 | Day 4 · Session 2 | Base against instruction-tuned | [`D4_2_base_and_instruct.ipynb`](labs/D4_2_base_and_instruct.ipynb) |
| 4.3 | Day 4 · Session 3 | Probe the limits | [`D4_3_probe_limits.ipynb`](labs/D4_3_probe_limits.ipynb) |
| 4.4 | Day 4 · Session 4 | The model-choice table | [`D4_4_model_choice.ipynb`](labs/D4_4_model_choice.ipynb) |
<!-- /academy:labs -->

## Capstone

A small model for one cooperative use case (image classifier, bilingual search or constrained text generation), submitted as a public repository whose README reports training curves, test results, a comparison with a simpler alternative, latency on a laptop and the licence of every model and dataset used, followed by an eight-minute defence and four minutes of questions

**Submission:** A public GitHub repository link, individual work: the code, the training logs and the commit history, and a README that reports the training curves, the test results, the comparison with a simpler alternative, the latency on a laptop and the licence of every model and dataset used. Then a live defence: an eight-minute talk and four minutes of questions, in which the participant changes one hyperparameter live and explains the new curve.

The brief, the README template and the rubric are in [`labs/capstone/`](labs/capstone/). Pre-work, a two-hour array-programming refresher checked by an entry quiz, is [`labs/00_prework_arrays.ipynb`](labs/00_prework_arrays.ipynb). The tracked follow-up task, a second dataset to adapt the capstone model to within two weeks, is in [`labs/followup/`](labs/followup/).

## How the course is organised

Four teaching days of four sessions, then a capstone day. A session is about 25 minutes of concepts, one checkpoint and a 25-minute lab on the case study. Every lab starts from supplied files, has a worked solution, and runs on a laptop processor or in Google Colab; Day 3 reads supplied vectors of the embedding model and Day 4 reads supplied logged runs of two small open-weight language models, so no language model is downloaded to a laptop. Practice questions for every outcome are in `assessments/bank.json`.

**Assessment.** A supervised entry quiz and exit check without AI assistance (shapes and gradients by hand, reading a learning curve, the transformer diagram, the effect of sampling settings): 20%. The capstone repository, scored on the five-criterion rubric: 60%. The live defence, in which the participant explains one curve or failure the assessor selects: 20%. Passing takes 70% overall, with at least 60% on the capstone rubric and at least 60% on the supervised check.

**AI policy.** No AI assistance in the supervised checks and in the first attempt at the training-loop lab (1.2) and the attention lab (3.2). In the other labs an assistant may explain and hint, one step at a time. In the capstone any assistant may be used, and the participant defends the work live.

**Responsible use.** Every model and dataset goes through a licence and provenance check that is recorded in the capstone README. Lab 4.3 probes the language model for fabricated answers and for different quality in Arabic and in English, and participants write the limits of a generated output before it may be shown to a customer. The national references are listed under Sources.

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

### Libraries and their documentation

- [PyTorch documentation: automatic differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch documentation: torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch documentation: the scaled dot-product attention function](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [torchvision: models and pretrained weights](https://docs.pytorch.org/vision/stable/models.html)
- [Sentence Transformers documentation](https://sbert.net/)
- [Hugging Face Transformers: text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/index)

### Models used in the labs, with their licences

- [ResNet-18 trained on ImageNet-1k, torchvision weights (model card)](https://huggingface.co/timm/resnet18.tv_in1k)
- [ImageNet: terms of access](https://www.image-net.org/download.php)
- [intfloat/multilingual-e5-small (model card)](https://huggingface.co/intfloat/multilingual-e5-small)
- [Qwen/Qwen2.5-0.5B (model card)](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Qwen/Qwen2.5-0.5B-Instruct (model card)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2.5-3B-Instruct (model card and licence)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [openai-community/gpt2 (model card)](https://huggingface.co/openai-community/gpt2)
- [Qwen/Qwen2.5-7B-Instruct (model card)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

### Papers

- [Vaswani et al., Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [He et al., Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
- [Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Kingma and Ba, Adam: A Method for Stochastic Optimization (2014)](https://arxiv.org/abs/1412.6980)
- [Yosinski et al., How transferable are features in deep neural networks? (2014)](https://arxiv.org/abs/1411.1792)
- [Holtzman et al., The Curious Case of Neural Text Degeneration (2019)](https://arxiv.org/abs/1904.09751)
- [Wang et al., Multilingual E5 Text Embeddings: A Technical Report (2024)](https://arxiv.org/abs/2402.05672)
- [Karpathy, A Recipe for Training Neural Networks (2019)](https://karpathy.github.io/2019/04/25/recipe/)
- [Ouyang et al., Training language models to follow instructions with human feedback (2022)](https://arxiv.org/abs/2203.02155)
- [Ho et al., Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)](https://arxiv.org/abs/2103.00020)

### National references

- [SDAIA: Generative AI Guideline for Government](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
- [SDAIA: Generative AI Guideline for Public](https://sdaia.gov.sa/en/SDAIA/about/Files/GenerativeAIPublicEN.pdf)
- [SDAIA: Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
<!-- /academy:sources -->

