'''
This is a base file to run and test the project from. 
Be careful making edits that may cause conflicts.
'''
from importlib import import_module
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import requests, zipfile
from pathlib import Path
import random
import numpy as np
from train import train

ARTICLE_ID = 22363012
RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "FracAtlas" / "FracAtlas" / "images"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def fracatlas_download():
    # 1. List the files in the article
    meta = requests.get(f"https://api.figshare.com/v2/articles/{ARTICLE_ID}", timeout=30).json()
    for f in meta["files"]:
        print(f["id"], f["name"], f["size"], f["download_url"])

    # 2. Download the file you want (65518038 is the id from your link)
    target = next(f for f in meta["files"] if f["id"] == 65518038)
    out = RAW_DIR / target["name"]

    if not out.exists():
        with requests.get(target["download_url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

    # 3. Verify integrity (figshare provides an MD5)
    import hashlib
    md5 = hashlib.md5(out.read_bytes()).hexdigest()  # fine for moderate sizes
    assert md5 == target["computed_md5"], "checksum mismatch"

    # 4. Extract if it's an archive
    if out.suffix == ".zip":
        with zipfile.ZipFile(out) as z:
            z.extractall(RAW_DIR / out.stem)

def setup(seed: int):
    set_seeds_to(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not IMAGE_DIR.exists() or not any(IMAGE_DIR.iterdir()):
        fracatlas_download()
    base_ds = datasets.ImageFolder(IMAGE_DIR, transform=transforms.ToTensor())
    print("Setup complete. You can now run the project.")
    g = torch.Generator().manual_seed(seed)
    n = len(base_ds)
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        base_ds, [train_size, val_size], generator = g
    )
    train_idx = train_subset.indices
    val_idx = val_subset.indices
    return device, base_ds, train_idx, val_idx

class TransformedSubset(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices, transform = None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        path, target = self.base.samples[base_idx]
        img = self.base.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.base.target_transform is not None:
            target = self.base.target_transform(target)
        
        return img, target
    
def make_transform(image_size: int = 224):
    if isinstance(image_size, int):
            size_hw = (image_size, image_size)
    else:
        size_hw = image_size
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.Resize(size_hw),
        transforms.ToTensor(),
        normalize
    ])
    val_tf = transforms.Compose([
        transforms.Resize(size_hw),
        transforms.ToTensor(),
        normalize
    ])
    return train_tf, val_tf

def make_loaders(base_ds, train_idx, val_idx, seed: int, batch_size: int, num_workers: int, image_size: int = 224):
    train_tf, val_tf = make_transform(image_size)
    train_ds = TransformedSubset(base_ds, train_idx, transform = train_tf)
    val_ds = TransformedSubset(base_ds, val_idx, transform = val_tf)
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, generator = g, pin_memory = pin
    )
    val_loader = DataLoader(
        val_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = pin
    )
    return train_loader, val_loader

def test_ResNet18Scratch(ResNet18Scratch, accelerator, train_loader, val_loader, num_classes) -> dict:
    # Part 4: Test ResNet18Scratch
    print("\n" + "="*60)
    print("Testing ResNet18Scratch")
    print("="*60)
    results = {}
    try:
        resnet_model = ResNet18Scratch(layers=[2, 2, 2, 2], num_classes=num_classes)
        print(f"ResNet18 Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
        test_epochs = 1
        resnet_history = train(accelerator, resnet_model, train_loader, val_loader, num_epochs=test_epochs)
        results['ResNet18'] = resnet_history['val_acc'][-1]
        # plot_training_history(resnet_history, "ResNet18")
        return resnet_history
    except Exception as e:
        print(f"Error in ResNet18Scratch: {e}")
        results['ResNet18Scratch'] = 0
        return {}
    
def main() -> None:
    seed = 42
    batch_size = 32
    num_workers = 0
    device, base_ds, train_idx, val_idx = setup(seed)
    train_loader_150, val_loader_150 = make_loaders(
        base_ds, train_idx, val_idx, seed = seed, batch_size = batch_size,
        num_workers = num_workers
    )

    # Load and test models here
    # ResNet18Scratch = import_module("ResNet18Scratch")
    # test_ResNet18Scratch(ResNet18Scratch.ResNet18Scratch, device, train_loader_150, val_loader_150, num_classes=len(base_ds.classes))
    print("The run has completed.")

if __name__ == "__main__":
    main()