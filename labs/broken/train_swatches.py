"""Shaima's first training script for the swatch classifier. It runs without an error and
its loss does not fall. Three things are wrong: one in what the loss receives, one in the
learning rate and one in the order of the data. Lab 1.4 finds them.

Do not fix this file: the lab passes corrected pieces to train() so that the log shows each
fault and its fix.
"""
import torch
import torch.nn.functional as F

LEARNING_RATE = 30.0


def compute_loss(logits, y):
    probabilities = torch.softmax(logits, dim=1)
    return F.cross_entropy(probabilities, y)


def make_batches(X, y, batch_size, generator):
    order = torch.randperm(len(X), generator=generator)
    for i in range(0, len(X), batch_size):
        yield X[order[i:i + batch_size]], y[i:i + batch_size]


def train(model, Xtr, ytr, Xva, yva, epochs=6, batch_size=32, seed=0,
          compute_loss=compute_loss, lr=LEARNING_RATE, make_batches=make_batches, verbose=True):
    """Train with plain gradient descent. Returns the validation loss (true cross-entropy)
    and accuracy per epoch."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in make_batches(Xtr, ytr, batch_size, g):
            loss = compute_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(xb), n + len(xb)
        model.eval()
        with torch.no_grad():
            out = model(Xva)
            val_loss = F.cross_entropy(out, yva).item()
            val_acc = (out.argmax(1) == yva).float().mean().item()
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch + 1}  train loss {total / n:.3f}  val loss {val_loss:.3f}  val acc {val_acc:.3f}")
    return history
