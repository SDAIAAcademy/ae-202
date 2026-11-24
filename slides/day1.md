---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.2.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/slides/day1.html"
---

# Deep Learning and Generative AI Fundamentals

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
