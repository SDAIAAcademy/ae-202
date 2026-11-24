---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.2.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/index.html"
---

# Deep Learning and Generative AI Fundamentals

أساسيات التعلّم العميق والذكاء الاصطناعي التوليدي

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 200

Train, debug, adapt and measure small deep learning and generative models, and choose a model family with evidence

Participants go from tensors to a working generative model and learn what each layer of the stack does. For a fictional crafts cooperative they write a training loop by hand, debug a network that will not learn, adapt a pretrained image model to synthetic textile images, implement attention for a toy sequence, build bilingual semantic search with sentence embeddings, and generate text with a small open-weight model while varying sampling settings. They finish by comparing model families on quality, size, latency, licence and cost for three of the cooperative's use cases.

**Who it is for:** Practitioners. Graduates and early-career developers who have completed the FN courses or passed the placement test, and software engineers moving into AI work.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Al-Nawl Crafts Cooperative.** A fictional cooperative of weavers and basket makers that sells through one online shop. Participants build three small models for it: Naqsh, which names the pattern of a textile from a photograph; Khayt, the search over the bilingual catalogue; and Wasf, a small open-weight language model that drafts one product line. All data is synthetic and drawn from fixed seeds: 8,000 textile-pattern images in eight classes with planted near-duplicates and wrong labels, 4,000 Arabic and English product descriptions with 50 labelled queries, and logged runs of two small open-weight language models.

## The course

### Day 1: Tensors and training

Slides: https://sdaiaacademy.github.io/ae-202/slides/day1.md

- **Session 1: Tensors, shapes and gradients**
  - Predict the shape of a result before running the code
  - Compute the gradients of a two-layer network with the chain rule
  - Check a hand-computed gradient against automatic differentiation
  - [Lab 1.1: A gradient by hand and by autograd (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D1_1_gradient_by_hand.ipynb)
- **Session 2: Writing the training loop**
  - Write a training step: forward, loss, backward, update
  - Draw mini-batches from a shuffled training set
  - Measure validation loss and accuracy without gradients
  - [Lab 1.2: The training loop (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D1_2_training_loop.ipynb)
- **Session 3: Learning rates, batches and regularisation**
  - Recognise a learning rate that is too low or too high from its curve
  - Explain what dropout and weight decay change
  - Choose a stopping epoch from the validation curve
  - [Lab 1.3: Three learning rates and dropout (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D1_3_learning_rates_dropout.ipynb)
- **Session 4: When training goes wrong**
  - Run an overfit-one-batch test and say what its result rules out
  - Find a fault in the loss input, the learning rate and the data order
  - Keep a debugging log that another engineer can follow
  - [Lab 1.4: Three faults in a training script (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D1_4_three_faults.ipynb)

### Day 2: Vision and transfer

Slides: https://sdaiaacademy.github.io/ae-202/slides/day2.md

- **Session 1: Convolutions and what they detect**
  - Compute a convolution on a small image by hand
  - Count the weights of a convolutional and a fully connected layer
  - Read first-layer filters and feature maps
  - [Lab 2.1: A small convolutional network (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D2_1_small_cnn.ipynb)
- **Session 2: Pretrained backbones**
  - Remove the head of a pretrained network and extract features
  - Prepare images the way the backbone was trained
  - Train a linear classifier on frozen features
  - [Lab 2.2: Frozen features (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D2_2_frozen_features.ipynb)
- **Session 3: Fine-tuning with little data**
  - Freeze and unfreeze chosen blocks of a network
  - Apply augmentation that keeps the label true
  - Compare frozen features, partial fine-tuning and training from scratch on one test set
  - [Lab 2.3: Fine-tune the last block (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D2_3_fine_tune.ipynb)
- **Session 4: Evaluating a deep model**
  - Read a confusion matrix by class
  - Inspect the most confident errors and sort them by cause
  - Find near-duplicates between training and test images and report accuracy without them
  - [Lab 2.4: Errors and near-duplicates (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D2_4_errors_duplicates.ipynb)

### Day 3: Transformers and embeddings

Slides: https://sdaiaacademy.github.io/ae-202/slides/day3.md

- **Session 1: Tokens, including Arabic**
  - Tokenise a sentence and read its pieces
  - Compare token counts of Arabic and English text across tokenisers
  - Explain what the token count changes: the window, the time and the price
  - [Lab 3.1: Three tokenisers (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D3_1_three_tokenisers.ipynb)
- **Session 2: Attention step by step**
  - Compute attention scores, weights and outputs for a toy sequence
  - Apply a causal mask and say what it prevents
  - Check an implementation against the library function
  - [Lab 3.2: Attention by hand (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D3_2_attention.ipynb)
- **Session 3: Sentence embeddings**
  - Embed texts and compare them with cosine similarity
  - Build a brute-force search over 4,000 vectors
  - Measure recall at 5 on labelled queries against a keyword baseline
  - [Lab 3.3: Catalogue search (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D3_3_catalogue_search.ipynb)
- **Session 4: Multilingual search and its failures**
  - Compare recall by query language
  - Separate a failure of language from a failure of wording
  - Write up three failure cases with a likely cause and a fix to try
  - [Lab 3.4: Failures by language (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D3_4_failures_by_language.ipynb)

### Day 4: Generation and model choice

Slides: https://sdaiaacademy.github.io/ae-202/slides/day4.md

- **Session 1: Next-token prediction and sampling**
  - Turn logits into probabilities at a given temperature
  - Apply nucleus sampling to a distribution by hand
  - Measure variation, length and rule violations across samples
  - [Lab 4.1: Four temperatures (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D4_1_four_temperatures.ipynb)
- **Session 2: From base models to assistants**
  - Show what a chat template adds to a prompt
  - Compare a base and an instruction-tuned model on the same tasks
  - Make a base model do a task with a few-shot prompt
  - [Lab 4.2: Base against instruction-tuned (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D4_2_base_and_instruct.ipynb)
- **Session 3: Limits: fabrication, cut-off and context**
  - Separate an abstention from a fabricated answer
  - Show what a runtime does with a prompt longer than its window
  - Write the limits of a generated output before a customer sees it
  - [Lab 4.3: Probe the limits (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D4_3_probe_limits.ipynb)
- **Session 4: Choosing a model family**
  - Match a model family to a task
  - Apply licence and data constraints before comparing quality
  - Justify a model choice with measured quality, size, latency and cost
  - [Lab 4.4: The model-choice table (25 min)](https://sdaiaacademy.github.io/ae-202/labs/D4_4_model_choice.ipynb)

### Capstone: Capstone: a small model for the cooperative, defended live

Slides: https://sdaiaacademy.github.io/ae-202/slides/capstone.md

## Capstone

A small model for one cooperative use case (image classifier, bilingual search or constrained text generation), submitted as a public repository whose README reports training curves, test results, a comparison with a simpler alternative, latency on a laptop and the licence of every model and dataset used, followed by an eight-minute defence and four minutes of questions

**What you submit:** A public GitHub repository link, individual work: the code, the training logs and the commit history, and a README that reports the training curves, the test results, the comparison with a simpler alternative, the latency on a laptop and the licence of every model and dataset used. Then a live defence: an eight-minute talk and four minutes of questions, in which the participant changes one hyperparameter live and explains the new curve.

**Rubric:**

- Working model and training: 25%
- Evaluation against a simpler alternative: 25%
- Measurement of latency and size: 15%
- Licence and limits: 15%
- Reproducibility: 20%

## Outcomes

- **LO1** Explain how a transformer turns input tokens into next-token probabilities, naming the role of embeddings, attention, the context window and sampling, given an annotated architecture diagram with five blanks and two generated outputs (Awareness) Evidence: a completed diagram and a short written explanation scored against a rubric in the supervised check
- **LO2** Implement a training loop with a loss, an optimiser, mini-batches and validation checks, and use it to train a small network, given a tensor library, synthetic image tensors and a starter script with no training code (Applied) Evidence: a script whose loss and validation curves match the reference behaviour within a stated tolerance
- **LO3** Apply a debugging checklist to a network whose loss does not fall and correct each fault found, given a training script with three seeded faults in the loss input, the learning rate and data shuffling (Applied) Evidence: a log showing the overfit-one-batch test, each fault and the corrected learning curve
- **LO4** Use transfer learning to adapt a pretrained image model to a new label set with little data, and measure the improvement over training from scratch, given 800 labelled textile images in eight classes and a laptop CPU time budget (Applied) Evidence: a results table comparing frozen features, partial fine-tuning and training from scratch on the same test set
- **LO5** Use sentence embeddings to build and test a bilingual semantic search over product descriptions, given 4,000 Arabic and English descriptions and 50 labelled queries (Applied) Evidence: recall at 5 for Arabic, English and mixed queries, with three failure cases and a likely cause for each
- **LO6** Measure how temperature, nucleus sampling and the instruction change the outputs of a small open-weight language model, given a locally run model and 20 fixed prompts sampled five times each (Applied) Evidence: a table of variation, length and rule violations per setting, with an explanation of one surprising result
- **LO7** Evaluate model families for three stated use cases on quality, size, latency, licence terms and cost, and recommend one per use case, given a supplied shortlist of open-weight and hosted models and results measured in the labs (Performance) Evidence: a model-choice table with a two-line justification per use case, defended in the capstone

## Tools

- **Tensor library with automatic differentiation:** PyTorch, JAX
- **Pretrained model loading:** Transformers, timm, Sentence Transformers
- **Local runtime for small open-weight language models:** llama.cpp, Ollama, a free hosted open-weight endpoint
- **Vector similarity search:** FAISS, brute-force search in NumPy
- **Experiment tracking:** MLflow, TensorBoard
- **Environment and notebook:** uv with JupyterLab, marimo, a free hosted notebook service

## Prerequisites

- AE-201 Machine Learning Foundations and Model Evaluation
- FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability
- Multiplies matrices and computes a derivative with the chain rule
- Has trained and evaluated a model with a train, validation and test split

## Sources

### Libraries and their documentation

- [PyTorch documentation: automatic differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch documentation: torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch documentation: torch.nn.BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
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

---

SDAIA Academy · AI Engineering Track · Deep Learning and Generative AI Fundamentals

Course version 2027.2.0 · engine 1.0.0 · built 2026-11-24 · commit c6f9bd5
