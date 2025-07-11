import torch
import torch.nn as nn
import pytorch_lightning as L

from src.model_factory import build_model

class LitRegressor(L.LightningModule):
    def __init__(self, input_dim: int, spec_path: str, lr: float = 1e-3):
        super().__init__()
        self.model = build_model(input_dim, spec_path)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        xb, yb = batch
        preds = self(xb)
        loss = self.loss_fn(preds, yb)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)