"""
Shared runner for the fracture project (FracAtlas), modeled on the Assignment 2 runner.py.

Put this file in the SAME FOLDER as the homework files it imports:
    VGG16.py               for vgg16_pretrained
    transfer_learning.py   for resnet18_pretrained, mobilenetv2_pretrained
    ResNet.py              for resnet18_scratch
    NiN.py                 for nin_scratch
You only need the files for the models you run.

No data files are needed: on the first run the script downloads FracAtlas from figshare (same
source as runner.py) and unpacks it into FracAtlas_data/ (the zip itself is not kept). It reads
dataset.csv from inside the data and finds each image by the name in the CSV (any folder layout).

Everyone edits ONLY the two lines marked EDIT, runs the file, and sends back the CSVs in OUT_DIR.
Start with SMOKE_TEST = True (tiny and fast), then set it to False for the real runs.
[Dataset] / [Method] / [Experiments] in the comments = the writeup section the code implements.
"HW" = what the Assignment 2 runner.py did where this file differs.
"""
import os
import random
import shutil
import time
import zipfile
import hashlib
import tempfile
import requests
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)
from tqdm import tqdm

# A few FracAtlas JPEGs are slightly truncated (a few bytes missing at the end). Without this
# line PIL raises "image file is truncated" and training crashes; with it, PIL loads what is there.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------- settings -----------------------------
SMOKE_TEST = False                       # True = tiny pipeline check; False = real runs
RUNNER_TAG = "kent"                 # EDIT: your name (used in output file names)
MODELS_TO_RUN = ["resnet18_scratch","resnet18_pretrained","vgg16_pretrained","mobilenetv2_pretrained","resnet18_scratch","nin_scratch"]   
 # EDIT: resnet18_pretrained, vgg16_pretrained,
                                        #       mobilenetv2_pretrained, resnet18_scratch, nin_scratch

# Everything below must stay identical across the team.
DATA_ROOT, OUT_DIR = "FracAtlas_data", "project_outputs"
LABELS_CSV = "dataset.csv"        # FracAtlas metadata: image_id, fractured, region, hardware, ...
FIGSHARE_ARTICLE_ID, FIGSHARE_FILE_ID = 22363012, 65518038   # FracAtlas.zip; downloaded into DATA_ROOT on the first run
IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS = 224, 32, 2     # [Method] 224x224 inputs
SPLIT_SEED = 42                                      # [Dataset] one fixed stratified 64/16/20 split
SEEDS = [1, 2, 3]                                    # [Experiments] training seeds
AUG_SETTINGS = ["none", "flip_rotate"]               # [Method] no augmentation vs. flip + rotation
ROTATION_DEG = 10                                    # [Method] rotation up to +/-10 degrees
LR_HEAD, LR_FULL, LR_SCRATCH = 0.001, 0.0001, 0.001  # [Method] learning rates
E1, E2, E_SCRATCH = 5, 10, 20                        # [Method] epochs: head only, full, from scratch
STEP_SIZE, GAMMA = 5, 0.1                            # [Method] StepLR (HW defaults)
RUN_MAJORITY = True                                  # majority-class baseline row (one person only)
RUN_WEIGHTED = False                                 # class-weighted retrain of the team's best config
WEIGHTED_MODEL, WEIGHTED_AUG = None, None            # e.g. "resnet18_pretrained", "flip_rotate"

PRETRAINED = {"resnet18_pretrained": "resnet18",     # pretrained models (two-stage training);
              "vgg16_pretrained": "vgg16",           # the name is the one transfer_learning.py uses,
              "mobilenetv2_pretrained": "mobilenet_v2"}   # except VGG16, which has its own VGG16.py
SCRATCH = ["resnet18_scratch", "nin_scratch"]

if SMOKE_TEST:
    E1, E2, E_SCRATCH, SEEDS, NUM_WORKERS = 1, 1, 1, [1], 0
    OUT_DIR = "smoke_outputs"            # separate folder, so smoke results never mix with real runs


# ------------------------------ data ------------------------------
class TransformedSubset(Dataset):
    """Subset of the base dataset with its own transform (HW: copied)."""
    def __init__(self, base, indices, transform=None):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, target = self.base.samples[self.indices[i]]
        img = self.base.loader(path)                      # loader converts to 3-channel RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_transforms(image_size, train_aug="none"):
    """[Method] 'none' = resize only. 'flip_rotate' = flip + small rotation, no crops or zoom
    (they could remove the fracture). Training only: val/test are never augmented.
    (HW: always flipped, and offered random_resized_crop.)"""
    size = (image_size, image_size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet statistics
                                     std=[0.229, 0.224, 0.225])
    val_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize])
    if train_aug == "flip_rotate":
        train_tf = transforms.Compose([transforms.Resize(size),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation(ROTATION_DEG),
                                       transforms.ToTensor(), normalize])
    elif train_aug == "none":
        train_tf = val_tf
    else:
        raise ValueError(f"unknown augmentation: {train_aug}")
    return train_tf, val_tf


def download_dataset():
    """If DATA_ROOT does not exist yet, download FracAtlas from figshare (same source as runner.py)
    and unpack it into DATA_ROOT. The zip goes to a temporary file that is deleted afterwards, so
    only the unpacked data is kept. It unpacks into DATA_ROOT_tmp first, so a failed run never
    leaves a half-empty DATA_ROOT. If DATA_ROOT already exists, this does nothing."""
    if os.path.isdir(DATA_ROOT):
        return
    meta = requests.get(f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}", timeout=30).json()
    target = next(f for f in meta["files"] if f["id"] == FIGSHARE_FILE_ID)
    tmp = DATA_ROOT + "_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)                              # leftover from an earlier failed run
    with tempfile.TemporaryFile() as fh:                # deleted automatically when closed
        print(f"Downloading {target['name']} ({target['size'] / 1e6:.0f} MB) from figshare ...")
        md5 = hashlib.md5()
        with requests.get(target["download_url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                md5.update(chunk)
        if md5.hexdigest() != target["computed_md5"]:
            raise RuntimeError("FracAtlas download failed its checksum; run again to retry.")
        print(f"Unzipping into {DATA_ROOT} ...")
        fh.seek(0)
        with zipfile.ZipFile(fh) as z:
            z.extractall(tmp)
    for dirpath, _, files in os.walk(tmp):
        for f in files:
            if f.lower().endswith(".zip"):              # unpack zips nested inside the download
                nested = os.path.join(dirpath, f)
                with zipfile.ZipFile(nested) as z:
                    z.extractall(dirpath)
                os.remove(nested)
    os.rename(tmp, DATA_ROOT)


def find_labels_csv(root):
    """dataset.csv from this folder if it is there, otherwise one found inside the unzipped data."""
    if os.path.exists(LABELS_CSV):
        return LABELS_CSV
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() == os.path.basename(LABELS_CSV).lower():
                return os.path.join(dirpath, f)
    return LABELS_CSV                                   # FracDataset will raise a clear error


class FracDataset:
    """[Dataset] Built from dataset.csv (image_id + fractured label + region/hardware flags) plus
    the image files, which can sit ANYWHERE under image_root (any folder layout). Replaces HW's
    ImageFolder; exposes the attributes TransformedSubset uses (samples, loader, target_transform).
    Labels: non-fractured = 0, fractured = 1."""
    def __init__(self, csv_path, image_root):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"'{csv_path}' not found. Put the FracAtlas dataset.csv in this folder (or inside the unzipped data).")
        self.meta = pd.read_csv(csv_path)
        found = {}
        for dirpath, _, files in os.walk(image_root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    found.setdefault(f, os.path.join(dirpath, f))      # first match wins
        missing = [i for i in self.meta.image_id if i not in found]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.meta)} images listed in {csv_path} were not found under "
                f"'{image_root}' (for example {missing[0]}). Download the FracAtlas images from "
                f"https://figshare.com/articles/dataset/The_dataset/22363012 and unzip it into '{image_root}'.")
        self.samples = [(found[i], int(y)) for i, y in zip(self.meta.image_id, self.meta.fractured)]
        self.targets = [t for _, t in self.samples]
        self.classes = ["non_fractured", "fractured"]
        self.class_to_idx = {"non_fractured": 0, "fractured": 1}
        self.target_transform = None

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert("RGB")           # grayscale X-rays -> 3 channels [Method]


def stratified_split(targets, seed):
    """[Dataset] stratified 64/16/20 (Milestone I feedback): 20% test first, then 20% of the
    remainder for validation. Keeps the ~17.6% fractured share in every subset.
    (HW: random 80/20, no test set.)"""
    idx, y = np.arange(len(targets)), np.array(targets)
    trainval_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.2, stratify=y[trainval_idx],
                                          random_state=seed)
    return train_idx, val_idx, test_idx


def set_up(seed, data_dir):
    """Device, dataset, fixed split. HW difference: `seed` is the SPLIT seed, not a training seed."""
    set_seeds_to(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator {device} will be used.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    download_dataset()                                      # first run only
    base_ds = FracDataset(find_labels_csv(data_dir), data_dir)
    pos = 1                                              # fractured = 1
    split = stratified_split(base_ds.targets, seed)
    y = np.array(base_ds.targets)
    print("classes:", base_ds.class_to_idx, "| images:", len(base_ds))
    for name, idx in zip(["train", "val", "test"], split):
        print(f"  {name}: {len(idx)} images, {(y[idx] == pos).sum()} fractured")
    return device, base_ds, split, pos


def make_loaders(base_ds, split, seed, train_aug):
    """HW make_loaders, plus a test loader. The training seed only changes batch order."""
    train_idx, val_idx, test_idx = split
    train_tf, val_tf = make_transforms(IMAGE_SIZE, train_aug)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return (DataLoader(TransformedSubset(base_ds, train_idx, train_tf), shuffle=True, generator=g, **kw),
            DataLoader(TransformedSubset(base_ds, val_idx, val_tf), shuffle=False, **kw),
            DataLoader(TransformedSubset(base_ds, test_idx, val_tf), shuffle=False, **kw))


def make_class_weights(base_ds, train_idx, pos):
    """[Method] fractured weight = non-fractured / fractured TRAINING images (about 4.7)."""
    counts = np.bincount(np.array(base_ds.targets)[train_idx], minlength=2)
    w = torch.ones(2)
    w[pos] = float(counts[1 - pos]) / float(counts[pos])
    return w


# ---------------------------- training (HW) ----------------------------
def set_seeds_to(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract=False):
    """While the backbone is frozen, batch norm stays in eval mode (ResNet18, MobileNetV2;
    VGG16 has no batch norm)."""
    model.train()
    if feature_extract:
        model.apply(_set_bn_eval)
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def train_model(device, model, train_loader, val_loader, num_epochs=10, lr=0.001,
                feature_extract=False, step_size=5, gamma=0.1, class_weights=None):
    """HW train_model (Adam + StepLR), plus optional class_weights for the weighting comparison."""
    model = model.to(device)
    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, feature_extract)
        vl, va = evaluate(model, val_loader, criterion, device)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"  epoch {epoch + 1}/{num_epochs}: train loss {tl:.4f} acc {ta:.1f}% | "
              f"val loss {vl:.4f} acc {va:.1f}%", flush=True)
        scheduler.step()
    return history


def plot_training_history(history, title):
    """[Experiments] training/validation curves (HW figure; saved to OUT_DIR)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Acc'); ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title(f'{title} - Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"training_history_for_{title.replace(' ', '_')}.png"))
    plt.close(fig)


# ------------------------------ models ------------------------------
def load_homework_module(name):
    """Import a homework file from this folder (HW main() used import_module)."""
    try:
        return import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            raise ModuleNotFoundError(f"Upload {name}.py into the same folder as this runner.") from e
        raise


def build_model(model_name, num_classes=2, freeze=True):
    """[Method] Two output scores per image. freeze=True freezes a pretrained backbone."""
    if model_name == "vgg16_pretrained":                 # VGG16.py (its own file)
        return load_homework_module("VGG16").get_vgg16(num_classes=num_classes, freeze=freeze)
    if model_name in PRETRAINED:                         # transfer_learning.py (ResNet18, MobileNetV2)
        tl = load_homework_module("transfer_learning")
        return tl.get_pretrained_model(PRETRAINED[model_name], num_classes=num_classes,
                                       feature_extract=freeze)
    if model_name == "resnet18_scratch":
        return load_homework_module("ResNet18Scratch").ResNet18Scratch([2,2,2,2], num_classes=num_classes)
    if model_name == "nin_scratch":
        model = load_homework_module("NiN").NiN(num_classes=num_classes)
        out_block = model.sequential[6]                  # output NiNBlock; its last layer is a ReLU
        k = len(out_block.sequential) - 1                # that would clamp scores at 0, so remove it
        assert isinstance(out_block.sequential[k], nn.ReLU), "unexpected NiN layout"
        out_block.sequential[k] = nn.Identity()
        return model
    raise ValueError(f"unknown model: {model_name}")


# ----------------------------- metrics -----------------------------
@torch.no_grad()
def predict_probs(model, loader, device, pos):
    """P(fractured) from softmax (applied here only, since CrossEntropyLoss takes raw scores)."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1)[:, pos].cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def best_f1_threshold(y, p, pos):
    """[Experiments] threshold = the cutoff that maximizes F1 on the VALIDATION set."""
    prec, rec, thr = precision_recall_curve(y == pos, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def fracture_metrics(y, p, thr, pos):
    """Recall (primary), precision, F1, specificity, AUROC, accuracy, confusion counts."""
    yt, yp = (y == pos).astype(int), (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(yt, p)                     # ranking quality, independent of threshold
        pr_auc = average_precision_score(yt, p)          # PR-AUC; a random model scores about 0.18 here
    except ValueError:                                   # only one class present (smoke test)
        auroc = pr_auc = float("nan")
    return dict(threshold=thr, recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                f1=f1_score(yt, yp, zero_division=0), specificity=tn / max(tn + fp, 1),
                auroc=auroc, pr_auc=pr_auc, accuracy=(tp + tn) / len(yt),
                tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))


# --------------------------- efficiency (HW) ---------------------------
def count_parameters(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_model_size_mb(model):
    """In-memory size assuming float32 weights (not the file size)."""
    n = sum(p.nelement() * p.element_size() for p in list(model.parameters()) + list(model.buffers()))
    return n / (1024 ** 2)


def measure_inference_time(model, input_shape=(1, 3, 224, 224), num_runs=100):
    """[Experiments] average ms per forward pass of one image, after a warm-up."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / num_runs * 1000.0


def estimate_flops(model, input_shape, device):
    """FLOPs per forward pass: conv and linear layers only, multiply-add = 2 FLOPs (HW rule)."""
    total = [0]

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        n, ho, wo = out.shape[0], out.shape[-2], out.shape[-1]
        total[0] += n * m.out_channels * ho * wo * (2 * (m.in_channels // m.groups) * kh * kw
                                                    + (1 if m.bias is not None else 0))

    def linear_hook(m, inp, out):
        batch = inp[0].numel() // m.in_features
        total[0] += batch * m.out_features * (2 * m.in_features + (1 if m.bias is not None else 0))

    handles = [m.register_forward_hook(conv_hook if isinstance(m, nn.Conv2d) else linear_hook)
               for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))
    for h in handles:
        h.remove()
    return total[0]


def efficiency_row(model_name, device):
    shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    m = build_model(model_name, freeze=False).to(device)
    return {"model": model_name, "params_millions": count_parameters(m)[0] / 1e6,
            "size_mb_in_memory": get_model_size_mb(m),
            "gflops": estimate_flops(m, shape, device) / 1e9,
            "inference_ms": measure_inference_time(m, shape), "runner": RUNNER_TAG}


# ------------------------------- runs -------------------------------
def run_one(model_name, aug, seed, ctx, class_weights=None):
    """Train one model with one augmentation setting and seed; evaluate the final-epoch model."""
    device, base_ds, split, pos = ctx
    tag = "weighted" if class_weights is not None else "unweighted"
    print(f"\n{'=' * 60}\n{model_name} | aug={aug} | {tag} | seed={seed}\n{'=' * 60}", flush=True)
    set_seeds_to(seed)
    train_loader, val_loader, test_loader = make_loaders(base_ds, split, seed, aug)
    model = build_model(model_name)                      # pretrained models start frozen

    # [Method] pretrained: head only, then full network at a lower rate. Scratch: one stage.
    stages = ([(E1, LR_HEAD, True), (E2, LR_FULL, False)] if model_name in PRETRAINED
              else [(E_SCRATCH, LR_SCRATCH, False)])
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epochs, lr, frozen in stages:
        print(f"-- stage: {'head only (backbone frozen)' if frozen else 'full network'}, "
              f"{epochs} epochs, lr {lr}")
        if not frozen:                                   # unfreeze, keep the same model object
            for p in model.parameters():
                p.requires_grad = True
        h = train_model(device, model, train_loader, val_loader, epochs, lr, frozen,
                        STEP_SIZE, GAMMA, class_weights)
        for k in history:
            history[k] += h[k]
    plot_training_history(history, f"{model_name} {aug} {tag} seed{seed}")

    pv, yv = predict_probs(model, val_loader, device, pos)
    thr = best_f1_threshold(yv, pv, pos)                 # tuned on validation only
    pt, yt = predict_probs(model, test_loader, device, pos)

    # Per-image test predictions, for the error analysis (which fractures were missed)
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    meta = base_ds.meta.iloc[split[2]].reset_index(drop=True)      # region, hardware, ... per test image
    pd.concat([meta[["image_id"]],
               pd.DataFrame({"label": (yt == pos).astype(int), "prob_fractured": pt,
                             "pred": (pt >= thr).astype(int)}),     # label and pred: 1 = fractured
               meta.drop(columns=["image_id", "fractured"])], axis=1
              ).to_csv(os.path.join(OUT_DIR, "predictions", f"{model_name}_{aug}_{tag}_seed{seed}.csv"),
                          index=False)
    return {**fracture_metrics(yt, pt, thr, pos),
            "val_recall": fracture_metrics(yv, pv, thr, pos)["recall"],   # to pick the best config
            "model": model_name, "aug": aug, "seed": seed,
            "weighted": class_weights is not None, "runner": RUNNER_TAG}


def majority_row(base_ds, test_idx, pos):
    """[Experiments] baseline that always predicts 'non-fractured': recall 0, AUROC 0.5."""
    y = np.array(base_ds.targets)[test_idx]
    return {**fracture_metrics(y, np.zeros(len(y)), 0.5, pos), "val_recall": 0.0,
            "model": "majority_class", "aug": "-", "seed": -1, "weighted": False, "runner": RUNNER_TAG}


def append_row(path, row):
    """Saved after every run, so a Colab disconnect loses nothing. Rewrites the small CSV with the
    new row added, so a newly added column (like pr_auc) never misaligns an older results file."""
    new = pd.DataFrame([row])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def load_done_runs(path):
    """Finished (model, aug, seed, weighted) runs, so a restart skips them."""
    if not os.path.exists(path):
        return set()
    return {(str(r.model), str(r.aug), int(r.seed), bool(r.weighted))
            for r in pd.read_csv(path).itertuples()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for m in MODELS_TO_RUN:
        assert m in PRETRAINED or m in SCRATCH, f"unknown model '{m}'"
    device, base_ds, split, pos = set_up(SPLIT_SEED, DATA_ROOT)
    if SMOKE_TEST:
        split = tuple(idx[:128] for idx in split)        # tiny subsets for a fast check
    ctx = (device, base_ds, split, pos)
    path = os.path.join(OUT_DIR, f"results_{RUNNER_TAG}.csv")
    done = load_done_runs(path)

    if RUN_MAJORITY and ("majority_class", "-", -1, False) not in done:
        append_row(path, majority_row(base_ds, split[2], pos))

    for model_name in MODELS_TO_RUN:                     # core experiment: models x aug x seeds
        for aug in AUG_SETTINGS:
            for seed in SEEDS:
                if (model_name, aug, seed, False) in done:
                    print(f"skip (already done): {model_name} {aug} seed={seed}")
                    continue
                append_row(path, run_one(model_name, aug, seed, ctx))

    eff = pd.DataFrame([efficiency_row(m, device) for m in MODELS_TO_RUN])   # once per model
    eff.to_csv(os.path.join(OUT_DIR, f"efficiency_{RUNNER_TAG}.csv"), index=False)

    if RUN_WEIGHTED:                                     # class weighting on the best config only
        assert WEIGHTED_MODEL and WEIGHTED_AUG in AUG_SETTINGS, "set WEIGHTED_MODEL / WEIGHTED_AUG"
        w = make_class_weights(base_ds, split[0], pos)
        for seed in SEEDS:                               # same seeds as the unweighted runs
            if (WEIGHTED_MODEL, WEIGHTED_AUG, seed, True) not in done:
                append_row(path, run_one(WEIGHTED_MODEL, WEIGHTED_AUG, seed, ctx, class_weights=w))

    df = pd.read_csv(path)
    df = df[~df.weighted.astype(bool)]
    print("\nMean / SD across seeds (test set):")       # [Experiments] mean and SD across seeds
    print(df.groupby(["model", "aug"])[[c for c in ["recall", "precision", "f1", "specificity", "auroc", "pr_auc"] if c in df.columns]]
            .agg(["mean", "std"]).round(3).to_string())
    print("\nEfficiency:\n", eff.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
