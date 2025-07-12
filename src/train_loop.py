import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import TabularDataset
from src.model_factory import build_model


def make_dataloader(csv_path: Path | str, target: str, batch: int = 32):
    """
    Create a PyTorch DataLoader for tabular data with batching and shuffling.
    
    Combines TabularDataset preprocessing with DataLoader functionality to create
    an iterator that yields mini-batches of preprocessed tabular data ready for training.
    
    Args:
        csv_path: Path to the CSV file containing the tabular dataset
        target: Name of the target column to predict (will be separated from features)
        batch: Number of samples per mini-batch for training
        
    Returns:
        DataLoader instance that yields batches of dictionaries containing:
            - "num": Batch of numeric features, shape (batch_size, n_numeric_features)
            - "cat": Batch of categorical features, shape (batch_size, n_categorical_features)
            - "y": Batch of target values, shape (batch_size, 1)
    """
    ds = TabularDataset(csv_path, target)  # Create preprocessed dataset
    return DataLoader(ds, batch_size=batch, shuffle=True)   # Wrap with batching and shuffling

def get_device() -> torch.device:
    """
    Automatically select the best available device for PyTorch computations.
    
    Checks if CUDA GPU acceleration is available on the system and returns the appropriate
    device. This enables automatic GPU usage when available for faster training.
    
    Returns:
        torch.device object set to:
            - "cuda" if NVIDIA GPU with CUDA support is available
            - "cpu" if no GPU acceleration is available (fallback)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available, else CPU

def make_model(input_dim: int, spec_path: str, device: torch.device):
    """
    Create and initialize a neural network model from JSON specification.
    
    Builds a TabularNet model using the architecture defined in the JSON config file,
    then moves all model parameters to the specified device (CPU or GPU) for computation.
    
    Args:
        input_dim: Number of input features (numeric + categorical embeddings)
        spec_path: Path to JSON file containing model architecture specification
        device: PyTorch device where the model should be placed (CPU or CUDA)
        
    Returns:
        TabularNet model instance with all parameters moved to the specified device,
        ready for training or inference
    """
    model = build_model(input_dim, spec_path)  # Build model from JSON spec
    return model.to(device)  # Move model parameters to target device

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Execute one complete training epoch through the entire dataset.
    
    Performs forward pass, loss calculation, backpropagation, and parameter updates
    for all batches in the dataset. Accumulates and returns the average loss across all samples.
    
    Args:
        model: Neural network model to train (TabularNet instance)
        dataloader: DataLoader yielding batches of training data
        loss_fn: Loss function for calculating training loss (e.g., MSELoss)
        optimizer: Optimization algorithm for updating model parameters (e.g., Adam)
        device: Device where computations should be performed (CPU or CUDA)
        
    Returns:
        Average loss value across all samples in the epoch (total loss / num samples)
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    total = 0.0  # Accumulator for total loss across all batches
    
    for xb, yb in dataloader:  # Iterate through all batches in the dataset
        xb, yb = xb.to(device), yb.to(device)  # Move batch data to target device
        optimizer.zero_grad()  # Clear gradients from previous batch
        preds = model(xb)  # Forward pass to get predictions
        loss = loss_fn(preds, yb)  # Calculate loss between predictions and targets
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters using computed gradients
        total += loss.item() * xb.size(0)  # Add batch loss (scaled by batch size) to total
    
    return total / len(dataloader.dataset)  # Return average loss per sample

def train(csv_path: str, target: str, spec_path: str, 
          epochs: int = 5, batch: int = 32, lr: float = 1e-3):
    """
    Complete training pipeline for tabular regression models.
    
    Handles the entire training workflow from data loading to model optimization.
    Creates dataset, builds model, sets up optimizer, and runs training loop with progress logging.
    
    Args:
        csv_path: Path to CSV file containing the training dataset
        target: Name of the target column to predict
        spec_path: Path to JSON file defining the neural network architecture
        epochs: Number of complete passes through the training dataset
        batch: Number of samples per mini-batch during training
        lr: Learning rate for the Adam optimizer
        
    Side Effects:
        - Prints training loss for each epoch to console
        - Modifies model parameters through training process
        - No return value - training results are printed
    """
    device = get_device()  # Auto-select best available device (GPU or CPU)
    dl = make_dataloader(csv_path, target, batch)  # Create batched data iterator
    model = make_model(dl.dataset.X.shape[1], spec_path, device)  # Build and place model
    loss_fn = torch.nn.MSELoss()  # Mean squared error for regression
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # Training loop - run for specified number of epochs
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, dl, loss_fn, opt, device)  # Train one epoch
        print(f"epoch {ep:2d}  loss {loss:,.2f}")  # Log progress