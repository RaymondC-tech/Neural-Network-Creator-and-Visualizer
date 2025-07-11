import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import TabularDataset
from src.model_factory import build_model


def make_dataloader(csv_path: Path | str, target: str, batch: int = 32):
    ds = TabularDataset(csv_path, target)
    return DataLoader(ds, batch_size=batch, shuffle=True)   #stacks samples into mini-batch tensors and does shuffling and multi-process prefetching for you

def get_device() -> torch.device:
    """CUDA if available, else use CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model(input_dim: int, spec_path: str, device: torch.device):
    """Instantiate the model and move it to the device"""
    model = build_model(input_dim, spec_path)
    return model.to(device)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """Loss + optimizer & one training epoch"""
    model.train()
    total = 0.0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()  #clear old gradients stroed in params otherwise being accumulated
        preds = model(xb) # forward pass to get prediction
        loss = loss_fn(preds, yb) #compare preds to ground truth
        loss.backward() #backward pass - autograd fills param.grad to compute delta loss / delta param
        optimizer.step() #update the weights using stored grads
        total += loss.item() * xb.size(0) # add batch loss (scaled by batch size) to epoch sum
    
    return total / len(dataloader.dataset)

def train(csv_path: str, target: str, spec_path: str, 
          epochs: int = 5, batch: int = 32, lr: float = 1e-3):
    device = get_device()
    dl = make_dataloader(csv_path, target, batch)        # DataLoader() wraps dataset so Pytorch can hand out mini-batches of size batch, shuffled each epoch d1 is now an iterator, each iteration returns (xb, yb) - a batch of featuers and labels
    model = make_model(dl.dataset.X.shape[1], spec_path, device)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, dl, loss_fn, opt, device)
        print(f"epoch {ep:2d}  loss {loss:,.2f}")