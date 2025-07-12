import torch
import torch.nn as nn
import pytorch_lightning as L

from src.model_factory import build_model

class LitRegressor(L.LightningModule):
    """
    PyTorch Lightning module for tabular regression with mixed numeric and categorical features.
    
    Wraps a TabularNet model in the Lightning framework to handle training loops, optimization,
    and logging automatically. Designed for regression tasks on structured/tabular data.
    """
    def __init__(self, input_dim: int, cat_dims: list[int], spec_path: str, lr: float = 1e-3):
        """
        Initialize the Lightning regression module.
        
        Args:
            input_dim: Number of numeric/continuous input features
            cat_dims: List containing number of unique categories for each categorical feature
            spec_path: Path to JSON file defining the neural network architecture
            lr: Learning rate for the Adam optimizer
            
        Instance Variables Created:
            self.model: TabularNet instance that processes numeric + categorical features
            self.loss_fn: MSE loss function for regression targets
            self.lr: Learning rate stored for optimizer configuration
        """
        super().__init__()
        self.model = build_model(input_dim, cat_dims, spec_path)  # Main neural network model
        self.loss_fn = nn.MSELoss()  # Mean squared error for regression
        self.lr = lr  # Learning rate for optimization
    
    def forward(self, batch):
        """
        Forward pass through the neural network model.
        
        Extracts numeric and categorical features from the batch dictionary and passes them
        through the TabularNet model to generate predictions.
        
        Args:
            batch: Dictionary containing preprocessed features with keys:
                - "num": Tensor of numeric features, shape (batch_size, n_numeric_features)
                - "cat": Tensor of categorical features, shape (batch_size, n_categorical_features)
                - "y": Target tensor (not used in forward pass)
                
        Returns:
            Tensor of model predictions with shape (batch_size, output_dim)
        """
        # Extract feature tensors from batch dictionary
        x_num = batch["num"]  # Numeric features tensor
        x_cat = batch["cat"]  # Categorical features tensor
        return self.model(x_num, x_cat)  # Pass through TabularNet model
    
    def training_step(self, batch, batch_idx):
        """
        Execute one training step (forward pass + loss calculation).
        
        Automatically called by Lightning during training. Computes predictions, calculates
        loss against targets, logs the loss for monitoring, and returns loss for backpropagation.
        
        Args:
            batch: Dictionary containing one batch of data with keys:
                - "num": Numeric features tensor
                - "cat": Categorical features tensor  
                - "y": Target values tensor
            batch_idx: Index of current batch (provided by Lightning, not used here)
            
        Returns:
            Loss tensor that Lightning will use for backpropagation and optimization
        """
        preds = self(batch)  # Get model predictions using forward method
        target = batch["y"]  # Extract target values from batch
        loss = self.loss_fn(preds, target)  # Calculate MSE loss
        self.log("train_loss", loss, prog_bar=True)  # Log loss for monitoring
        return loss  # Return loss for backpropagation
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Called automatically by Lightning to set up the optimization strategy. Creates an Adam
        optimizer with the specified learning rate to update all trainable parameters in the model.
        
        Returns:
            Adam optimizer instance configured with:
                - All model parameters (including embeddings and MLP weights)
                - Learning rate specified during initialization
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # Adam optimizer with stored learning rate