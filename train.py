import torch
from torch import nn
from sklearn.metrics import f1_score

# ImageFolder sorts classes alphabetically: {'Fractured': 0, 'Non_fractured': 1}
FRACTURED = 0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)              # shape: (batch, num_classes)
        loss = criterion(logits, labels)    # no reshaping needed
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    return (total_loss / total, *compute_metrics(all_preds, all_labels))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    return (total_loss / total, *compute_metrics(all_preds, all_labels))


def compute_metrics(all_preds, all_labels):
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    # F1 for the fractured class, since that's the minority class we care about
    f1 = f1_score(labels, preds, pos_label=FRACTURED, zero_division=0)
    return acc, f1


def train(device, model, train_loader, val_loader, num_epochs=10, lr=1e-3, class_weights=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        for k, v in zip(history, (train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)):
            history[k].append(v)
        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")

    return history
