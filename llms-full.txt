---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.2.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/"
---

# Deep Learning and Generative AI Fundamentals

Train, debug, adapt and measure small deep learning and generative models, and choose a model family with evidence

Participants go from tensors to a working generative model and learn what each layer of the stack does. For a fictional crafts cooperative they write a training loop by hand, debug a network that will not learn, adapt a pretrained image model to synthetic textile images, implement attention for a toy sequence, build bilingual semantic search with sentence embeddings, and generate text with a small open-weight model while varying sampling settings. They finish by comparing model families on quality, size, latency, licence and cost for three of the cooperative's use cases.

- **Audience:** Practitioners. Graduates and early-career developers who have completed the FN courses or passed the placement test, and software engineers moving into AI work.
- **Prerequisites:** AE-201 Machine Learning Foundations and Model Evaluation; FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability; Multiplies matrices and computes a derivative with the chain rule; Has trained and evaluated a model with a train, validation and test split
- **Case study:** Al-Nawl Crafts Cooperative: A fictional cooperative of weavers and basket makers that sells through one online shop. Participants build three small models for it: Naqsh, which names the pattern of a textile from a photograph; Khayt, the search over the bilingual catalogue; and Wasf, a small open-weight language model that drafts one product line. All data is synthetic and drawn from fixed seeds: 8,000 textile-pattern images in eight classes with planted near-duplicates and wrong labels, 4,000 Arabic and English product descriptions with 50 labelled queries, and logged runs of two small open-weight language models.

## Outcomes

- LO1 (Awareness): Explain how a transformer turns input tokens into next-token probabilities, naming the role of embeddings, attention, the context window and sampling, given an annotated architecture diagram with five blanks and two generated outputs
- LO2 (Applied): Implement a training loop with a loss, an optimiser, mini-batches and validation checks, and use it to train a small network, given a tensor library, synthetic image tensors and a starter script with no training code
- LO3 (Applied): Apply a debugging checklist to a network whose loss does not fall and correct each fault found, given a training script with three seeded faults in the loss input, the learning rate and data shuffling
- LO4 (Applied): Use transfer learning to adapt a pretrained image model to a new label set with little data, and measure the improvement over training from scratch, given 800 labelled textile images in eight classes and a laptop CPU time budget
- LO5 (Applied): Use sentence embeddings to build and test a bilingual semantic search over product descriptions, given 4,000 Arabic and English descriptions and 50 labelled queries
- LO6 (Applied): Measure how temperature, nucleus sampling and the instruction change the outputs of a small open-weight language model, given a locally run model and 20 fixed prompts sampled five times each
- LO7 (Performance): Evaluate model families for three stated use cases on quality, size, latency, licence terms and cost, and recommend one per use case, given a supplied shortlist of open-weight and hosted models and results measured in the labs

## Day 1: Tensors and training

From a tensor to a network that learns, and what to do when it does not

#### [day1#title] Tensors and training

*SDAIA Academy · Deep Learning and Generative AI Fundamentals*

Day 1: From a tensor to a network that learns, and what to do when it does not

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-202

#### [day1#agenda] Tensors and training

*Today*

- **Session 1 · Tensors, shapes and gradients**: What a tensor is, how shapes combine, and how a gradient flows back through a computation
- **Session 2 · Writing the training loop**: The five lines every training step has, and the validation pass that tells you whether they worked
- **Session 3 · Learning rates, batches and regularisation**: How the learning rate, the batch size, dropout and weight decay change a learning curve, and when to stop
- **Session 4 · When training goes wrong**: A checklist for a loss that does not fall, starting with the cheapest test

#### [day1#case-study] One cooperative, three small models

*The case study*

- **Naqsh · the pattern classifier.** Names the pattern of a textile from a photograph: eight classes, 8,000 images. Days 1 and 2.
- **Khayt · the catalogue search.** Finds products among 4,000 Arabic and English descriptions. Day 3.
- **Wasf · the description drafter.** A small open-weight language model that writes one product line. Day 4.

> **Al-Nawl Crafts Cooperative is fictional.** Every image, description and log is synthetic and drawn from a fixed seed. You are Zaid Al-Luhaidan, its junior machine learning engineer.

#### [day1#what-you-keep] Each day removes one layer of magic

*The week*

1. **Day 1**: Tensors and training (the loop, by hand)
2. **Day 2**: Vision and transfer (800 labels are enough)
3. **Day 3**: Transformers and embeddings (tokens, attention, search)
4. **Day 4**: Generation and choice (sampling, limits, licences)

Day 5 is yours: a small model for one use case, defended live.

### Session 1: Tensors, shapes and gradients

#### [day1#s1] Tensors, shapes and gradients

*Session 1*

- Predict the shape of a result before running the code
- Compute the gradients of a two-layer network with the chain rule
- Check a hand-computed gradient against automatic differentiation

#### [day1#tensor-is-array] A tensor is an array with a shape, a number format and a device

*What a tensor is*

- **Shape**: how many axes, how long each one is
- **Number format**: 32-bit floats by default; 64-bit when you check arithmetic
- **Device**: the processor or a graphics processor; two tensors must share one
- One more thing than an array: it can **remember how it was computed**

```python
X.shape    # torch.Size([4, 576])
X.dtype    # torch.float32
X.device   # cpu
W1.requires_grad   # True
```

> **At Al-Nawl.** A batch of four grey swatches of 24 by 24 pixels is a tensor of shape `(4, 576)`.

#### [day1#swatches] Eight patterns, laid straight on the sorting table

*Today's data*

Figure: 4,000 synthetic swatches: 3,000 to train, 500 to validate, 500 to test. Grey, 24 by 24 pixels, 576 numbers each.

#### [day1#shapes-first] Predict the shape before you run the line

*Shapes*

| Operation | Shapes in | Shape out |
| --- | --- | --- |
| `X @ W1` | (4, 576) and (576, 5) | (4, 5) |
| `X @ W1 + b1` | (4, 5) and (5,) | (4, 5) |
| `X.mean(axis=0)` | (4, 576) | (576,) |
| `X.mean(axis=1, keepdims=True)` | (4, 576) | (4, 1) |
| `X - X.mean(axis=1)` | (4, 576) and (4,) | an error |

> **What goes wrong.** Most deep learning bugs are shape bugs that do not raise an error: a shape of `(4,)` against `(4, 1)` broadcasts to `(4, 4)` and trains on nonsense.

#### [day1#broadcasting-rule] Shapes align from the right; a size of 1 stretches

*Broadcasting*

- **Works.** `(4, 576)` with `(576,)`: the last axes match, the vector is repeated over the 4 rows.
  `(4, 576)` with `(4, 1)`: the 1 stretches over the 576 columns.
- **Fails, or worse.** `(4, 576)` with `(4,)`: 576 against 4, an error.
  `(4,)` with `(4, 1)`: no error, a `(4, 4)` table nobody asked for.

> **Habit.** Print `.shape` after every new line until the shapes are boring.

#### [day1#two-layer-network] Two layers: multiply, bend, multiply, compare

*The network of today*

1. **Input** X: (N, 576)
2. **Layer 1** z1 = X W1 + b1, then relu
3. **Layer 2** z2 = h W2 + b2: eight scores
4. **Loss** softmax, then −log p of the true class

- The eight raw scores are called **logits**; softmax turns them into probabilities
- **Cross-entropy** is the mean of −log p(true class): 0 when certain and right, ln 8 = 2.08 when guessing among eight

#### [day1#chain-rule-backwards] The backward pass is the chain rule, one local step at a time

*Gradients*

1. Softmax with cross-entropy: `dz2 = (p − onehot) / N`
2. Linear layer: `dW = a.T @ dz`, `db = dz.sum(0)`
3. Pass back: `da = dz @ W.T`
4. Relu: the gradient passes where `z1 > 0`

- **One rule to check everything.** A gradient has the shape of the thing it is the gradient of. `dW1` is `(576, 5)` because `W1` is.

> **You know this.** It is the chain rule of FN-103, written for matrices. The lab does it for four swatches.

#### [day1#autograd-graph] The library records the forward pass and replays it backwards

*Automatic differentiation*

```python
W1 = torch.tensor(w1, requires_grad=True)
z2 = torch.relu(X @ W1 + b1) @ W2 + b2
loss = F.cross_entropy(z2, y)
loss.backward()
W1.grad.shape     # (576, 5)
```

- Each operation adds a node to a **computational graph**
- `backward()` applies the local rules from the loss to every leaf
- Gradients **add up** in `.grad` until you clear them

**1e-16** largest gap between hand and autograd in Lab 1.1, in 64-bit numbers; 9e-8 between a 32-bit and a 64-bit run

#### [day1#check-shapes] Which gradient is right?

*Checkpoint*

**Question:** A layer computes `z = a @ W + b` with `a` of shape (32, 64) and `W` of shape (64, 8). You have `dz`, the gradient of the loss with respect to `z`. Which line gives the gradient for `W`?

- A. `dz @ W.T`
- B. `a @ dz`
- C. `a.T @ dz`
- D. `dz.sum(axis=0)`

#### [day1#lab-1-1] A gradient by hand and by autograd

Lab 1.1 · 25 min

Compute the gradients of a two-layer network by hand in NumPy and with automatic differentiation, and compare the numbers.

[D1_1_gradient_by_hand.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D1_1_gradient_by_hand.ipynb)

- [ ] Shapes and broadcasting on a batch of swatches (5 min)
- [ ] The forward pass by hand (5 min)
- [ ] The backward pass with the chain rule (8 min)
- [ ] Automatic differentiation and the comparison (4 min)
- [ ] Number formats: 32 and 64 bits (3 min)
- [ ] ✱ Stretch: Check one weight with a finite difference

### Session 2: Writing the training loop

#### [day1#s2] Writing the training loop

*Session 2*

- Write a training step: forward, loss, backward, update
- Draw mini-batches from a shuffled training set
- Measure validation loss and accuracy without gradients

#### [day1#five-lines] Every training step is the same five lines

*The training step*

```python
logits = model(xb)                  # 1 forward
loss = F.cross_entropy(logits, yb)  # 2 loss
optimiser.zero_grad()               # 3 clear old gradients
loss.backward()                     # 4 backward
optimiser.step()                    # 5 update the weights
```

> **From a two-layer network to a language model**, these five lines do not change. What changes is the model, the data and the loss.

#### [day1#what-loss-receives] Cross-entropy takes logits, not probabilities

*The loss*

- `F.cross_entropy` applies the softmax itself, in a numerically safe way
- Its inputs: raw scores of shape (N, classes) and class numbers of shape (N,)
- Give it probabilities and it applies a second softmax: **no error, a loss that cannot reach zero**

**1.27** the loss of a perfect prediction when softmax is applied twice (Lab 1.4)

#### [day1#optimiser-rule] The optimiser is the rule that turns a gradient into a step

*The optimiser*

- **Gradient descent (SGD).** `w = w − lr × grad`. One setting: the learning rate is the step size and nothing else.
- **Adam.** Keeps a running average of each weight's gradient and of its square, and scales every step by them. Forgiving: a rate of 0.001 works on most small problems.

> **In the labs.** Adam when the point is the loop (1.2), plain gradient descent when the point is the learning rate (1.3, 1.4).

#### [day1#mini-batches] A mini-batch is a noisy, cheap estimate of the full gradient

*Batches*

- All 3,000 swatches per step: an exact gradient, one update per pass
- 32 swatches per step: a noisy gradient, **94 updates per pass**
- An **epoch** is one pass over the training set in a new random order
- Shuffle every epoch, and keep each input with its own label

> **What goes wrong.** Inputs shuffled without their labels give a network nothing to learn.

#### [day1#validation-pass] The validation pass measures and must not learn

*Validation*

```python
model.eval()
with torch.no_grad():
    logits = model(Xva)
    loss = F.cross_entropy(logits, yva)
    acc = (logits.argmax(1) == yva)
          .float().mean()
```

- `eval()`: layers that behave differently in training switch to their test behaviour
- `no_grad()`: no graph is recorded, so no memory and no time are spent on it
- Once per epoch, on data the training step never sees

#### [day1#reference-curves] Both losses fall together, from chance towards zero

*What a healthy run looks like*

Figure: Training and validation loss of the reference run over 12 epochs. Both start at chance, 2.08, and fall together to about 0.7; the validation loss stays just under the training loss.

**0.79** validation accuracy after 12 epochs; chance is 0.125

37,448 weights, Adam at 0.001, batches of 32. Validation loss from 1.99 to 0.70 (Lab 1.2, reference run).

#### [day1#tolerance-not-equality] Two correct runs differ; say by how much they may

*Reproducibility*

- A different batch order or seed gives a different path to a similar place
- Fix the seed to repeat **your** run; state a tolerance to compare with **someone else's**
- Keep the curve as a file, with the settings that produced it

> **Experiment tracking** (MLflow, TensorBoard) is this habit with a user interface: one record per run with its settings, its curves and its code version.

#### [day1#check-zero-grad] The curve that jumps

*Checkpoint*

**Question:** Zaid's loop runs. The training loss falls for a few steps, then swings wildly and grows. His step has four lines: forward, loss, `loss.backward()`, `optimiser.step()`. What is the most likely cause?

- A. Gradients from earlier steps are still added to each new one
- B. The model was never put in evaluation mode
- C. The batches were not shuffled between epochs
- D. The batch size of 32 is too small for this network

#### [day1#lab-1-2] The training loop

Lab 1.2 · 25 min

Write a training loop with batching, a loss and an optimiser for the textile swatches and plot the loss curves against the reference.

[D1_2_training_loop.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D1_2_training_loop.ipynb)

- [ ] Mini-batches from a shuffled order (4 min)
- [ ] Model, loss and optimiser (4 min)
- [ ] The training step (7 min)
- [ ] The validation pass (5 min)
- [ ] Curves against the reference (5 min)
- [ ] ✱ Stretch: Replace your batching with a library data loader

### Session 3: Learning rates, batches and regularisation

#### [day1#s3] Learning rates, batches and regularisation

*Session 3*

- Recognise a learning rate that is too low or too high from its curve
- Explain what dropout and weight decay change
- Choose a stopping epoch from the validation curve

#### [day1#learning-rate-first] Tune the learning rate first: it decides whether anything else matters

*The learning rate*

Figure: Validation loss over eight epochs at three learning rates. At 0.001 the loss stays flat at chance, 2.08. At 0.1 it falls steadily to 1.23. At 3.0 it jumps between 2.4 and above 4.

- **0.001**: flat at 2.08, accuracy 0.11. Too low: the weights barely move
- **0.1**: falls to 1.23, accuracy 0.62. Usable
- **3.0**: between 2.4 and 12. Too high: every step overshoots

#### [day1#sweep-powers-of-ten] Sweep powers of ten, read curves, then refine

*How to search*

1. **1 · Coarse** 0.001, 0.01, 0.1, 1: a few epochs each
2. **2 · Read** too low, usable or too high, from the validation curve
3. **3 · Refine** two or three values around the best one

- Change **one setting at a time**, from the same starting weights
- A larger batch gives a less noisy gradient and usually tolerates a larger rate
- Schedules (warm-up, decay) shape the rate over time; a constant rate is the baseline

#### [day1#overfitting-gap] The gap between the two losses is what the network memorised

*Overfitting*

- Training loss keeps falling as long as the network has capacity to spare
- Validation loss stops falling when the rest is **particular to the training examples**
- The gap grows with a larger network, a smaller training set, more epochs

> **At Al-Nawl.** With 400 swatches and 256 hidden units: training loss 0.52, validation loss 0.95 after 60 epochs. A gap of 0.43 (Lab 1.3).

#### [day1#dropout-weight-decay] Dropout and weight decay make memorising harder

*Regularisation*

- **Dropout.** In training, each hidden unit is switched off with probability p at every step. No unit can rely on one other unit. Off in evaluation mode.
- **Weight decay.** Every step also pulls each weight a little towards zero. Large weights must earn their size.
- **Augmentation.** Show changed copies that keep the label: a flipped or shifted image. More data without new labels. Day 2.

> **What goes wrong.** Validating with dropout still on: the model is measured with half its units missing. This is what `model.eval()` prevents.

#### [day1#regularisation-evidence] The gap shrank; the validation loss hardly moved

*What the lab measured*

| 400 swatches, 60 epochs | Training loss | Validation loss | Gap | Validation accuracy |
| --- | --- | --- | --- | --- |
| Plain | 0.52 | 0.95 | 0.43 | 0.64 |
| Dropout 0.5 | 0.86 | 1.05 | 0.19 | 0.68 |
| Weight decay 0.001 | 0.60 | 0.93 | 0.33 | 0.64 |
| All 3,000 swatches, 12 epochs (Lab 1.2) | 0.73 | 0.70 | 0 | 0.79 |

> **Read it honestly.** Regularisation stops a network from flattering itself. It does not replace the data that is missing.

#### [day1#early-stopping] Keep the weights of the best validation epoch

*When to stop*

- Watch the validation loss; save the weights whenever it improves
- **Patience**: stop after this many epochs without an improvement
- A noisy curve needs more patience: with 500 validation swatches, a patience of 3 stops on a bump
- The stopping epoch is chosen on **validation** data; the test set is used once, at the end

> **In Lab 1.3.** On the plain run, a patience of 3 keeps epoch 43; a patience of 10 keeps epoch 58, with a lower loss (0.93 against 0.98).

#### [day1#curve-reading-table] Four shapes cover most of what a curve can tell you

*Reading curves*

| What you see | Likely meaning | First thing to try |
| --- | --- | --- |
| Both losses flat at chance | Rate far too low, or no signal reaches the network | Overfit one batch (session 4) |
| Loss jumps, grows or becomes "nan" | Rate too high | Divide the rate by 10 |
| Training falls, validation rises | Overfitting | Stop earlier, more data, regularise |
| Both fall slowly and are still falling | Not finished, or rate a little low | More epochs, or a rate 3 times larger |

#### [day1#check-curve] What would you change first?

*Checkpoint*

**Question:** A network with 256 hidden units trains on 400 swatches. After 60 epochs the training loss is 0.20 and still falling; the validation loss reached 0.93 at epoch 35 and is now 1.10. Which change addresses what the curves show?

- A. Raise the learning rate so that training converges faster
- B. Add a second hidden layer of 256 units
- C. Train for 60 more epochs until the training loss is near zero
- D. Keep the weights of epoch 35 and get more labelled swatches

#### [day1#lab-1-3] Three learning rates and dropout

Lab 1.3 · 25 min

Run the same model at three learning rates, then with and without dropout on a small training set, and read the curves.

[D1_3_learning_rates_dropout.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D1_3_learning_rates_dropout.ipynb)

- [ ] Three learning rates, one model (8 min)
- [ ] Read the three curves (4 min)
- [ ] Dropout and weight decay on 400 swatches (8 min)
- [ ] The early-stopping epoch (5 min)
- [ ] ✱ Stretch: Repeat one run with batches of 8 and of 256

### Session 4: When training goes wrong

#### [day1#s4] When training goes wrong

*Session 4*

- Run an overfit-one-batch test and say what its result rules out
- Find a fault in the loss input, the learning rate and the data order
- Keep a debugging log that another engineer can follow

#### [day1#silent-failure] A broken training script usually runs without an error

*Why this is hard*

- Ordinary code fails loudly: an exception, a line number
- Training code fails quietly: **a number that is worse than it should be**
- Several faults stack, and each hides the others
- So you need tests with a **known right answer**

- log: epoch 1 · train loss 2.130 · val loss 65.4 · val acc 0.124
- log: epoch 2 · train loss 2.148 · val loss 67.3 · val acc 0.126
- log: epoch 3 · train loss 2.149 · val loss 31.9 · val acc 0.128

#### [day1#overfit-one-batch] First, make the network memorise one batch

*The cheapest test*

- Take 16 examples. Train on them alone for a few hundred steps
- A healthy model, loss and optimiser drive the loss **towards zero**
- It takes seconds and has a known right answer

- **It fails.** The fault is in the model, the loss or the update. More data will not help.
- **It passes, and full training still fails.** Those three work. Look at what reaches them: the data pipeline.

#### [day1#checklist] Five checks, from cheap to expensive

*The checklist*

1. **Run as found** and write down the numbers: is the loss at chance, above it, "nan"?
2. **Overfit one batch**
3. **Check the loss on a known case**: a perfect prediction must cost about 0
4. **Sweep the learning rate** on the one-batch test, in powers of ten
5. **Test the pipeline**: does every input still travel with its own label?

> **One change at a time, and a log line for each.** What you tested, what it showed, what you changed.

#### [day1#stacked-faults] A correct fix can leave the test failing

*Stacked faults*

| State of the script | One-batch loss after 300 steps | Verdict |
| --- | --- | --- |
| As found | 134.2 | Fails |
| Loss input fixed, rate still 30 | 4.4 | Fails: another fault |
| Loss input fixed, rate 0.1 | 0.08 | Passes |

> **What goes wrong.** Undoing a correct fix because the curve did not improve. Keep the fix when its own check passes (a perfect prediction now costs 0), and look for the next fault.

#### [day1#pipeline-test] Tag every row and see whether the pairs survive

*The data pipeline*

```python
tag = torch.arange(40)
X = tag.float().unsqueeze(1)
for xb, yb in make_batches(X, tag, 8, g):
    assert torch.equal(
        xb.squeeze(1).long(), yb)
```

- Give the pipeline inputs that **are** their labels
- Any batch where they differ is a pipeline bug
- Passing the one-batch test and sitting at exactly ln 8 on full data points here

**2.08 → 1.23** validation loss with two fixes, and with the third (the data order), after 8 epochs (Lab 1.4)

#### [day1#batches-by-class] A loss that falls within each class and jumps at the next

*Another data-order fault*

- batch 1: 32 stripes · loss 1.98
- batch 11: 32 stripes · loss 0.51
- batch 13: 32 chevrons · loss 4.35
- batch 23: 32 chevrons · loss 1.37
- epoch 3: val acc 0.14 · 90% of validation swatches called "dot"

- The swatches stored class by class and never shuffled: **87 of 94 batches hold one class**
- From the second class on, each class starts at 2.6 to 4.3 and ends its run at 1.4 to 2.3: the network learns "say chevron", and the next class punishes it
- Pairs broken (Lab 1.4) look different: every batch loss stays between 2.06 and 2.12, near ln 8

#### [day1#batches-by-class-fix] Shuffle the examples, not the batches

*The fix*

| Same swatches, same network, 3 epochs | Classes per batch | Validation accuracy |
| --- | --- | --- |
| Stored order, class by class | 1.07 | 0.14 |
| One-class batches, taken in a new order each epoch | 1.07 | 0.22 |
| Swatches in a new random order each epoch | 7.84 | 0.52 |

> **The test.** Print the labels of the first three batches. A representative batch looks like the whole training set: most classes, in about their usual shares.

#### [day1#other-usual-suspects] The usual suspects, by where they sit

*Beyond today's three*

- **Data.** Labels shifted or shuffled away; inputs not normalised as the model expects; the same preprocessing missing at validation.
- **Model and loss.** Wrong loss input; an output layer with the wrong number of classes; a silent broadcast between (N,) and (N, 1).
- **Update.** Rate too high or too low; gradients never cleared; weights frozen by mistake; evaluation mode left on.

#### [day1#check-one-batch] Where is the fault?

*Checkpoint*

**Question:** Rayan trains a classifier on ten product categories. On 16 examples the loss reaches 0.02 in 200 steps. On the full training set it stays at 2.30 for 20 epochs, at every learning rate he tries. Where should he look first?

- A. At the size of the network: it cannot hold ten categories
- B. At the data pipeline: inputs and labels no longer match
- C. At the loss function: it cannot reach zero
- D. At the learning rate: it needs a finer sweep

#### [day1#lab-1-4] Three faults in a training script

Lab 1.4 · 25 min

Find three seeded faults in a broken training script, starting with an overfit-one-batch test, and log each fix.

[D1_4_three_faults.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D1_4_three_faults.ipynb)

- [ ] Run the broken script and read its log (3 min)
- [ ] The overfit-one-batch test (6 min)
- [ ] Fault in the loss input (5 min)
- [ ] Fault in the learning rate (4 min)
- [ ] Fault in the data order, and the corrected curve (7 min)
- [ ] ✱ Stretch: Plant a fourth fault for your neighbour

#### [day1#wrap-up] Three things to keep

*Day 1 · wrap-up*

- **Shapes and gradients.** Predict the shape first. A gradient has the shape of its weight, and autograd computes what you computed by hand.
- **Five lines and two curves.** Forward, loss, clear, backward, update. Then read the training and validation losses together, against chance.
- **Debug with known answers.** Overfit one batch first. Keep a correct fix even when the curve stays flat, and log every step.

#### [day1#next] Day 2: Vision and transfer

*Next*

The network that reached 0.79 on swatches meets the catalogue photographs. Unfinished labs can be finished at home; each one starts from supplied files.

## Day 2: Vision and transfer

Why convolutions suit images, and how a pretrained model turns 800 labels into a working classifier

#### [day2#title] Vision and transfer

*SDAIA Academy · Deep Learning and Generative AI Fundamentals*

Day 2: Why convolutions suit images, and how a pretrained model turns 800 labels into a working classifier

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-202

#### [day2#agenda] Vision and transfer

*Today*

- **Session 1 · Convolutions and what they detect**: A convolution as one small detector slid over the image, pooling, and what the first layer learns
- **Session 2 · Pretrained backbones**: What a network trained on a large image set already knows, and how to use it as a fixed feature extractor
- **Session 3 · Fine-tuning with little data**: Which layers to unfreeze, at which learning rate, and how augmentation stands in for data you do not have
- **Session 4 · Evaluating a deep model**: Beyond one accuracy number: the confusion matrix, the confident errors and a test set that leaks

#### [day2#review-day1] A flat curve, one more time

*Review · Day 1*

**Question:** A new script trains an eight-class classifier. The loss stays at 2.08 for ten epochs. On 16 examples alone, with the same loss and learning rate, the loss also stays near 2.0 after 300 steps. What do you do next?

- A. Tag the rows and test the data pipeline
- B. Check the loss on a perfect prediction, then sweep the learning rate on the one batch
- C. Collect more labelled examples and train again
- D. Add dropout and weight decay to stabilise the training

### Session 1: Convolutions and what they detect

#### [day2#s1] Convolutions and what they detect

*Session 1*

- Compute a convolution on a small image by hand
- Count the weights of a convolutional and a fully connected layer
- Read first-layer filters and feature maps

#### [day2#photographs] The catalogue photographs are turned, shifted and in colour

*Today's data*

Figure: 8,000 synthetic photographs, 48 by 48 pixels: 6,000 to train, 1,000 to validate, 1,000 to test. Colour says nothing about the class.

#### [day2#mlp-fails] One weight per pixel position cannot learn a pattern that moves

*Yesterday's network on today's data*

**0.15** validation accuracy of a two-layer network with 885,896 weights on the photographs (Lab 2.1). Chance is 0.125; on straight swatches it reached 0.79

- A fully connected layer learns "bright at row 3, column 7"
- Shift a stripe by three pixels and every one of those weights is wrong
- It would need to see each pattern at **every** position and angle

#### [day2#convolution-idea] A convolution slides one small detector over the whole image

*Convolution*

- A **kernel** is a small grid of weights: 3 by 3, 5 by 5
- At each position: multiply kernel and pixels under it, add up
- The result is a **feature map**: where in the image the kernel's pattern occurs
- The same weights at every position: **a stripe is a stripe wherever it lies**

```python
edge = torch.tensor([[-1., 0., 1.],
                     [-1., 0., 1.],
                     [-1., 0., 1.]])
out[i, j] = (img[i:i+3, j:j+3] * edge).sum()
# 48 x 48 in, 46 x 46 out
```

#### [day2#weights-shared] Sharing weights across positions cuts the first layer 700 times

*Why it is small*

| Item | Value |
| --- | --- |
| Fully connected, 6,912 inputs to 128 units | 884,864 |
| Convolution, 16 kernels of 5 by 5 by 3 | 1,216 |

- Whole network of Lab 2.1: three convolution blocks and a head, **25,096 weights**
- Fewer weights to fit, and each one sees every position of every image

#### [day2#pooling-and-depth] Pooling shrinks the map; depth widens what a unit sees

*The rest of the network*

1. **Conv 5×5, 16** edges and bars, 48 × 48
2. **Pool, conv 3×3, 32** corners and crossings, 24 × 24
3. **Pool, conv 3×3, 64** motifs, 12 × 12
4. **Average, linear** 64 numbers, then 8 scores

- **Max pooling** keeps the largest value of each 2 by 2 block: half the size, less sensitive to a small shift
- After two poolings, one unit of the third layer looks at a 20-pixel patch of the photograph
- The final **average over all positions** asks "does this motif occur?", not "where?"

#### [day2#head-average-or-flatten] Average or flatten: count the layer that follows

*The last map and the head*

| Map 32 × 6 × 6, 8 classes | Inputs | Head weights |
| --- | --- | --- |
| Average, then linear | 32 | 264 |
| Flatten, then linear | 1,152 | 9,224 |

> **Count it.** Flatten: inputs = channels × height × width. Average: inputs = channels. Flattened, the head holds 66% of the network's weights (5% averaged) and accepts one image size only: on a 32 × 32 swatch it fails.

#### [day2#cnn-result] 35 times fewer weights, and it learns the photographs

*What the lab measured*

| Network, 6,000 training photographs | Weights | Validation accuracy |
| --- | --- | --- |
| Two layers, fully connected, 5 epochs | 885,896 | 0.15 |
| Convolutional, supplied checkpoint (8 epochs) | 25,096 | 0.71 |
| Convolutional, 2 more epochs in the lab | 25,096 | 0.86 |

#### [day2#first-layer-filters] Nobody designed these filters; they are what lowered the loss

*What the first layer learnt*

Figure: The 16 first-layer filters of the supplied checkpoint, each rescaled to its own range.

- Bars and edges at different angles
- Colour contrasts: one yarn against another
- Deeper layers combine them; those are too abstract to draw as pictures

#### [day2#check-conv] What does the convolution buy?

*Checkpoint*

**Question:** Shaima photographs the same striped rug twice; in the second photograph the rug sits four pixels to the right. Which statement about the first layer of each network is right?

- A. Both layers give the same output for the two photographs
- B. The convolution gives the same feature map for the two photographs
- C. The convolution needs both photographs in training to recognise the second
- D. The convolution's feature map shifts by four pixels; its values stay the same

#### [day2#lab-2-1] A small convolutional network

Lab 2.1 · 25 min

Train a small convolutional network on the catalogue photographs from a supplied checkpoint and visualise its first-layer filters.

[D2_1_small_cnn.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D2_1_small_cnn.ipynb)

- [ ] A convolution by hand (5 min)
- [ ] Build the network and count its weights (6 min)
- [ ] Train from the supplied checkpoint (8 min)
- [ ] First-layer filters and feature maps (6 min)
- [ ] ✱ Stretch: Shift every test image by four pixels and compare the two networks; then count the last layer with the map flattened instead of averaged

### Session 2: Pretrained backbones

#### [day2#s2] Pretrained backbones

*Session 2*

- Remove the head of a pretrained network and extract features
- Prepare images the way the backbone was trained
- Train a linear classifier on frozen features

#### [day2#eight-hundred-labels] The cooperative has 800 labelled photographs, not 6,000

*The real constraint*

- Labels cost an expert's time: Shaima named 100 photographs per pattern
- The network of this morning, trained from scratch on those 800 for 20 epochs: **0.48** on the test set

**0.48** test accuracy, small network from scratch on 800 photographs (supplied run, 60 seconds)

> **This is the baseline of the day.** Every result this afternoon is read against it, on the same 1,000 test photographs.

#### [day2#pretrained-backbone] Someone has already trained the filters on a million photographs

*Pretrained backbones*

- A **backbone** is a network trained on a large, general image set, published as a file of weights
- Early layers: edges, colours. Middle: textures, repeated shapes. Last layer: that set's own classes
- The early and middle layers **transfer**: every image task needs them

- **In the lab: ResNet-18.** 11.7 million weights, 11.2 million without its 1,000-class head, trained on ImageNet-1k (1.28 million photographs, 1,000 everyday classes). A 45 MB download, loaded with one line (torchvision; the same weights are in timm).

#### [day2#remove-head] Remove the last layer and the network returns 512 features

*Feature extraction*

```python
m = torchvision.models.resnet18(
        weights="IMAGENET1K_V1")
m.fc = nn.Identity()   # was 512 -> 1000
m.eval()
with torch.no_grad():
    features = m(prepare(X))   # (N, 512)
```

- The 1,000-class layer is the only part about ImageNet's classes
- Everything before it becomes a fixed function: photograph in, 512 numbers out
- **Frozen**: evaluation mode, no gradients, nothing trained

#### [day2#prepare-as-trained] Feed a backbone what it was trained on

*Preprocessing*

- **Normalise** each colour channel with the mean and standard deviation of the backbone's training set
- **Resize** towards the size it was trained at: its filters expect motifs of a certain size in pixels
- These numbers belong to the weights; they are published with them

> **What goes wrong.** Raw pixels in [0, 1] fed to a backbone raise no error. The features are just worse, by a little or by a lot. It is the quiet kind of fault from yesterday's list.

#### [day2#linear-probe] One linear layer on frozen features is the cheapest transfer

*The classifier*

- Extract features once; they are a table of numbers now
- Standardise with statistics of the **training** features
- Train `Linear(512, 8)`: 4,104 weights, seconds
- Nothing in the backbone can overfit, because nothing in it moves

| Item | Value |
| --- | --- |
| From scratch, 800 photographs | 0.48 |
| Frozen features + linear layer | 0.77 |

#### [day2#several-labels] A rug with stripes and stars needs a yes or no for each motif

*One label or several*

| Motif | Score, set by hand | Softmax | Sigmoid |
| --- | --- | --- | --- |
| stripe | 2.4 | 0.55 | 0.92 |
| star | 2.0 | 0.37 | 0.88 |
| wave | −0.4 | 0.03 | 0.40 |
| check | −0.8 | 0.02 | 0.31 |
| four others | −2.6 to −1.5 | 0.01 or less | 0.07 to 0.18 |
| Sum |  | 1.00 | 3.01 |

- Mariam wants every piece listed under **each** motif it carries
- Softmax makes the motifs compete for one unit of probability: only stripe passes 0.5
- A sigmoid per motif answers eight separate questions: stripe and star both pass

#### [day2#sigmoid-per-label] Independent sigmoids with binary cross-entropy

*The head for several labels*

```python
head = nn.Linear(512, 8)       # the same layer
Y = torch.tensor(tags).float() # (N, 8) of 0 and 1
loss = F.binary_cross_entropy_with_logits(
    head(f), Y)
chosen = torch.sigmoid(head(f)) > 0.5
```

| Each example has | Output | Loss |
| --- | --- | --- |
| Exactly one class | softmax | cross-entropy on class numbers |
| Any number of labels, zero included | a sigmoid per label | binary cross-entropy on a 0/1 table |

> **What goes wrong.** A softmax head with a 0.5 cut runs without an error, and two motifs can never both pass it.

#### [day2#where-features-come-from] A backbone carries the terms of the data it was trained on

*Provenance*

- **The code and the weights file.** torchvision is under a BSD 3-clause licence, and the model card of these ResNet-18 weights states the same licence.
- **The training data.** ImageNet's terms of access allow non-commercial research and educational use. Whether weights trained on it may be used in a shop is a question for Bushra Al-Aqeel, not for the engineer alone.

> **The habit for the capstone.** For every model and dataset: its licence, where it came from, and who checked. Day 4 turns this into a column of the model-choice table.

#### [day2#check-frozen] Which statistics?

*Checkpoint*

**Question:** Zaid extracts features for 800 training and 1,000 test photographs. Before the linear layer he standardises them. Which mean and standard deviation does he use for the test features?

- A. Those of the 800 training features
- B. Those of the 1,000 test features
- C. The ImageNet mean and standard deviation of the backbone
- D. Those of all 1,800 features together

#### [day2#lab-2-2] Frozen features

Lab 2.2 · 25 min

Extract features from the 800 labelled images with a pretrained image model and train a linear classifier on them.

[D2_2_frozen_features.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D2_2_frozen_features.ipynb)

- [ ] Load the backbone and remove its head (4 min)
- [ ] Resize and normalise as the backbone expects (4 min)
- [ ] Extract features without gradients (7 min)
- [ ] A linear classifier on the features (6 min)
- [ ] Against training from scratch on the same 800 (4 min)
- [ ] ✱ Stretch: Take the features one block earlier and compare

### Session 3: Fine-tuning with little data

#### [day2#s3] Fine-tuning with little data

*Session 3*

- Freeze and unfreeze chosen blocks of a network
- Apply augmentation that keeps the label true
- Compare frozen features, partial fine-tuning and training from scratch on one test set

#### [day2#three-options] Three ways to use 800 labels, from least to most trained

*The choice*

- **Frozen features.** Train only a new head. Seconds. Cannot damage the backbone, cannot adapt it either.
- **Partial fine-tuning.** Also train the last block or two. The most task-specific layers adapt; the general ones stay.
- **Full fine-tuning.** Train everything at a small rate. The most capacity to adapt, and to overfit little data.

> **Rule of thumb.** The less data you have and the closer your images are to the backbone's, the more you freeze. Then measure: the rule is a starting point.

#### [day2#freezing-in-code] A frozen weight has no gradient and is never updated

*Freezing*

```python
m.fc = nn.Linear(512, 8)
for p in m.parameters():
    p.requires_grad = False
for p in m.layer4.parameters():
    p.requires_grad = True
for p in m.fc.parameters():
    p.requires_grad = True
```

**75%** of ResNet-18's weights, without the head, sit in its last block: 8.4 of 11.2 million (Lab 2.3)

"Only the last block" is still 8 million numbers fitted to 800 photographs.

#### [day2#two-learning-rates] Small steps for what is already good, larger steps for what is new

*Learning rates*

```python
optimiser = torch.optim.Adam([
    {"params": m.layer4.parameters(),
     "lr": 3e-4},
    {"params": m.fc.parameters(),
     "lr": 3e-3},
])
```

- The pretrained block starts from useful weights: it is nudged
- The new head starts from random numbers: it must travel
- One large rate for both destroys the block in the first steps

#### [day2#augmentation] Augmentation is extra data that costs no labels, if it keeps the label true

*Augmentation*

- **Safe for these patterns.** Flips, small shifts, a rotation, a change of colours: a flipped chevron is a chevron, and the class does not depend on colour.
- **Not safe everywhere.** Flip a photograph of text or of a left-hand glove and the label is wrong. A heavy blur can turn a wave into a stripe.

- Applied to each batch in **training** only, with new random choices every time
- The validation and test photographs are never augmented

#### [day2#results-table] Transfer bought 30 points; fine-tuning added one more

*What the lab measured*

| 800 labelled photographs, same 1,000 test photographs | Test accuracy | Weights trained | Seconds |
| --- | --- | --- | --- |
| Small network from scratch | 0.476 | 25,096 | 60 |
| Frozen features + linear layer | 0.774 | 4,104 | 24 |
| Last block fine-tuned, with augmentation | 0.788 | 8,397,832 | 102 |

> **Read all three columns.** The last row is the best by 1.4 points, for four times the training time and 2,000 times the trained weights. On 1,000 test photographs, 1.4 points is 14 photographs.

#### [day2#when-finetuning-pays] Fine-tuning pays more when the new images differ more

*Beyond this table*

- Close to the backbone's data, little data: **frozen features** are hard to beat
- Far from it (medical scans, satellite images, microscope slides), more data: **unfreeze more**
- Always keep the frozen-feature result as the row to beat: it costs seconds
- A time budget is part of the task: this outcome says "a laptop CPU time budget"

> **Going further.** Detection, segmentation, export and field tests of vision models are the subject of AE-309 Computer Vision Engineering.

#### [day2#check-finetune] What went wrong in the first epoch?

*Checkpoint*

**Question:** Rayan unfreezes the last block of a pretrained backbone and trains it together with a new head, one Adam optimiser at a learning rate of 0.01 for everything. After one epoch the validation accuracy is below what frozen features gave. What is the most likely cause?

- A. The augmentation made the training images too different
- B. Too few of the backbone's layers were left frozen
- C. The large steps overwrote the pretrained weights of the block
- D. The new head had too few weights to use the features

#### [day2#lab-2-3] Fine-tune the last block

Lab 2.3 · 25 min

Fine-tune the last block with augmentation on 800 images and compare with the supplied frozen-feature and from-scratch results.

[D2_3_fine_tune.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D2_3_fine_tune.ipynb)

- [ ] Freeze all but the last block and the head (5 min)
- [ ] Augmentation that keeps the label true (5 min)
- [ ] Fine-tune with two learning rates (9 min)
- [ ] The results table on one test set (6 min)
- [ ] ✱ Stretch: Unfreeze one more block and measure the cost in time

### Session 4: Evaluating a deep model

#### [day2#s4] Evaluating a deep model

*Session 4*

- Read a confusion matrix by class
- Inspect the most confident errors and sort them by cause
- Find near-duplicates between training and test images and report accuracy without them

#### [day2#one-number-hides] "0.79" answers none of Mariam's next three questions

*Evaluation*

- **Which patterns?** Is it 0.79 on every class, or perfect on six and guessing on two? The confusion matrix.
- **What kind of errors?** Hard photographs, a systematic confusion, or labels that are wrong? The confident errors.
- **Can I trust the test?** Has the model seen some test photographs before? The duplicate check.

#### [day2#confusion-matrix] Two pairs of classes hold most of the errors

*The confusion matrix*

| Labelled as | Photographs | Right | Most often mistaken for |
| --- | --- | --- | --- |
| chevron | 127 | 0.50 | wave (62 times) |
| dot | 143 | 0.69 | diamond (30 times) |
| diamond | 138 | 0.78 | dot (18 times) |
| wave | 123 | 0.86 | chevron (15 times) |
| check | 118 | 0.97 | star (2 times) |

> **Read the direction.** Chevrons are called waves four times as often as waves are called chevrons: at 48 pixels a zigzag's corners blur into curves, and the model's default between the two is "wave".

#### [day2#confident-errors] Five of the ten most confident errors were wrong labels

*Confident errors*

Figure: The ten test photographs on which the model was most confidently wrong, each with its label and the model's prediction. In several the picture plainly shows the pattern the model named, not the one on the label.

- Sort the wrong predictions by the model's confidence
- Look at the top ten, one by one
- Either the model learnt something wrong, or **the label is wrong**
- The test set held 12 wrong labels; the model disagreed with 11 of them

#### [day2#near-duplicates] The same photograph in training and in test is not a test

*Near-duplicates*

Figure: Top: test. Bottom: its nearest training photograph. Four re-uploads, then two honest look-alikes.

- Makers upload a photograph twice; a random split puts one copy on each side
- Fingerprint every image, compare test against training
- Sorted similarities: 0.999, 0.998, 0.997, then a drop to 0.972
- **40** test photographs above 0.985

#### [day2#report-without] Report the accuracy without the photographs the model had seen

*What to report*

- **0.788** all 1,000 test photographs
- **1.000** the 40 near-duplicates
- **0.779** the other 960: the number to report

> **Here it moved the number by one point.** In a set where a quarter of the test images have a twin in training, the same check changes the decision. You do not know which case you are in until you look.

#### [day2#evaluation-habits] What stands behind a number in your capstone README

*The habits*

1. The test set, its size, and how it was kept apart
2. The result of a **simpler alternative** on the same test set
3. The confusion matrix, and the two worst pairs in words
4. The confident errors you looked at, and what they were
5. The duplicate check, and the number without duplicates

#### [day2#check-evaluation] What do you do with this error?

*Checkpoint*

**Question:** On a test photograph labelled "diamond" the model says "check" with a confidence of 0.999. You look at the photograph: it shows a checked cloth. What is the right next step?

- A. Add the photograph to the training set with its label and fine-tune again
- B. Send it to the owner of the labels and count such cases
- C. Delete the photograph from the test set and report the new accuracy
- D. Lower the model's confidence with a threshold so that it abstains here

#### [day2#lab-2-4] Errors and near-duplicates

Lab 2.4 · 25 min

Build a confusion matrix for the fine-tuned model, inspect its ten most confident errors and check the splits for near-duplicates.

[D2_4_errors_duplicates.ipynb](https://sdaiaacademy.github.io/ae-202/labs/D2_4_errors_duplicates.ipynb)

- [ ] The confusion matrix (6 min)
- [ ] The ten most confident errors (7 min)
- [ ] Near-duplicates between training and test (8 min)
- [ ] Accuracy with and without them (4 min)
- [ ] ✱ Stretch: Precision and recall per class

#### [day2#wrap-up] Three things to keep

*Day 2 · wrap-up*

- **Convolution shares weights.** One small detector at every position: 25 thousand weights learnt what 886 thousand could not.
- **Start from someone's weights.** With 800 labels: 0.48 from scratch, 0.77 on frozen features, 0.79 fine-tuned. Keep the cheap row as the one to beat.
- **Look behind the number.** Confusion pairs, confident errors, duplicates. Then report the number you can defend.

#### [day2#next] Day 3: Transformers and embeddings

*Next*

From pixels to text: tokens, attention, and a search over the cooperative's bilingual catalogue. Unfinished labs can be finished at home.

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
