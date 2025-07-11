import pytorch_lightning as L
from src.lit_model import LitRegressor 
from src.train_loop import make_dataloader, get_device

def quick_fit():
    device = get_device()
    dl = make_dataloader("data/housing.csv", "price", batch=16)
    model = LitRegressor(dl.dataset.X.shape[1], "src/model_spec.json")
    trainer = L.Trainer(max_epochs=3, accelerator=str(device))
    trainer.fit(model, dl)

if __name__ == "__main__":
    quick_fit()