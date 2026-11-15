---
course: "Deep Learning and Generative AI Fundamentals"
code: "AE-202"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-202/slides/day2.html"
---

# Deep Learning and Generative AI Fundamentals

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

#### [day2#cnn-result] 35 times fewer weights, and it learns the photographs

*What the lab measured*

| Network, 6,000 training photographs | Weights | Validation accuracy |
| --- | --- | --- |
| Two layers, fully connected, 5 epochs | 885,896 | 0.15 |
| Convolutional, supplied checkpoint (8 epochs) | 25,096 | 0.71 |
| Convolutional, 2 more epochs in the lab | 25,096 | 0.86 |

> **Starting from a checkpoint** is how every lab today begins: someone else's training, saved as a file of weights, continued by you.

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
- [ ] ✱ Stretch: Shift every test image by four pixels and compare the two networks

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
- Its validation accuracy jumps between 0.27 and 0.56 from epoch to epoch

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
