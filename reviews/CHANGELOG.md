# Changes

## 2027.2.0 (2026-11-24)

- Added: `batches-by-class` and `batches-by-class-fix` (day 1, session 4): reading the batch log of training data stored class by class, and shuffling the examples rather than the batches.
- Added: `head-average-or-flatten` (day 2, session 1): the weights of the layer after an average over positions and after flattening.
- Added: `several-labels` and `sigmoid-per-label` (day 2, session 2): independent sigmoids with binary cross-entropy for several labels per example, against a softmax head.
- Added: a stretch exercise in Lab 2.1 that counts the last layer of the network with the map flattened instead of averaged, with its solution and check.
- Added: three practice questions, `q-order-01`, `q-head-01` and `q-labels-01`.
- Added: `labs/_tools/make_heads_and_order.py`, which computes the numbers of the new slides into `labs/data/reference/heads_and_order.json`.
- Tightened: the warning on `mini-batches` (day 1) now covers only inputs shuffled without their labels; a bullet of `eight-hundred-labels` and the callout of `cnn-result` (day 2) moved into the notes.
