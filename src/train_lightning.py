import pytorch_lightning as L
from src.lit_model import LitRegressor 
from src.train_loop import make_dataloader, get_device

def quick_fit():
    """
    Quick training demo using PyTorch Lightning for tabular regression.
    
    Demonstrates the complete Lightning training pipeline with mixed numeric and categorical features.
    Uses the housing dataset to train a TabularNet model with embeddings for categorical variables.
    
    Side Effects:
        - Loads and preprocesses the housing.csv dataset
        - Creates TabularNet model with embeddings based on categorical feature cardinality
        - Trains for 3 epochs using Lightning's automatic training loop
        - Prints training progress and loss metrics to console
        - No return value - training results are logged by Lightning
    """
    device = get_device()  # Auto-select best device (GPU or CPU)
    dl = make_dataloader("data/mix.csv", "price", batch=16)  # Create data loader
    
    # Extract feature dimensions from preprocessed dataset
    num_dim = dl.dataset.X_num.shape[1]  # Number of numeric features
    cat_dims = [len(dl.dataset.cat_maps[col]) for col in dl.dataset.cat_cols]  # Categories per categorical feature
    
    # Create Lightning module and trainer
    model = LitRegressor(num_dim, cat_dims, "src/model_spec.json")  # Lightning wrapper for TabularNet
    trainer = L.Trainer(max_epochs=3, accelerator=str(device))  # Lightning trainer configuration
    trainer.fit(model, dl)  # Execute training loop

if __name__ == "__main__":
    quick_fit()