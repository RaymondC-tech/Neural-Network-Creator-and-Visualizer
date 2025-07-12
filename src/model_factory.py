import torch
import torch.nn as nn
import json
from pathlib import Path

_ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        None: nn.Identity,    # maps JSON null -> no activation
    }
"""
Dictionary mapping activation function names to their PyTorch implementations.
Maps string names from JSON config to actual PyTorch activation classes.
None key handles cases where no activation is specified in JSON.
"""
class TabularNet(nn.Module):
    """
    Neural network for tabular data that combines numeric and categorical features.
    
    Creates embedding layers for categorical variables and combines them with numeric features
    before passing through an MLP (Multi-Layer Perceptron).
    """
    def __init__(self, num_dim: int, cat_dims: list[int], mlp: nn.Sequential):
        """
        Initialize the TabularNet model.
        
        Args:
            num_dim: Number of numeric/continuous features in the input data
            cat_dims: List containing the number of unique categories for each categorical feature
            mlp: Pre-built Sequential model that processes the combined numeric + embedding features
        """
        super().__init__()
        #1. building embedding layer list here (we'll write it next step)
        self.embs = nn.ModuleList()
        emb_out_total = 0
        for n_cat in cat_dims:  #cat dums contain how many unique things you have for each categorical varaible
            emb_dim = infer_emb_dim(n_cat)
            self.embs.append(nn.Embedding(n_cat, emb_dim)) # nn.Embedding object whose internal weight is a matrix that is (num_embedding x embedding_dim, each row is the embedding vector for ecah category_id) this is the matrix for that categorical variable
            emb_out_total += emb_dim
            #self.embs is list of all the embedding layuers, storing all the matrices

        #2. store the MLP (multi layer perception which is the plain feed-forward network you already had for numeric inputs only)
        self.mlp = mlp
        self.num_dim = num_dim
        self.emb_out_total = emb_out_total
    
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Embeded categorical inputs, concatenate with numeric inputs, and run through the MLP

        Args:
            x_num: Float tenwsor of shape (batch_size, num_dim) with continuous featuers
            x_cat (torch.Tensor): Long tensor of shape (batch_size, n_cat_cols) with integer-encoded categories before the embedding, just there id

        Returns:
            torch.Tensor: Float tensor of shape (batch_size, output_dim) containing the model’s raw outputs.
        
        """
        #1. lookup categorical column's embeddings
        if self.embs:
            emb_outs = [ 
                emb(x_cat[:,i])    #x_cat[:,i] means for all rows, gets the ith column
                for i, emb in enumerate(self.embs)  # example embs = (tensor([1, 2])) means [emb.weight[1], emb.weight[2]]
            ]
            cat_emb = torch.cat(emb_outs, dim=1) #concatenate these tensor into one tensor since emb_out is a list of tensor of shape (batch_size, emd_dim_i). now it is [(batch_size x emb_dim0), (batch_size x emb_dim1)]
            #dim=1 means extend it as dim=0 mean stack top to bottom instead of extending the feature vector
            x = torch.cat([x_num, cat_emb], dim=1)

        else:
            x = x_num

        #2. pass the concatenated tensor through the mlp
        return self.mlp(x)

def build_model(num_dim: int, cat_dims: list[int], spec_path: Path | str) -> nn.Module:
    """
    Build a TabularNet model from a JSON specification file.
    
    Reads a JSON configuration file that specifies the architecture of the neural network
    and creates a TabularNet that combines numeric features with categorical embeddings.
    
    Args:
        num_dim: Number of numeric/continuous features in the input data
        cat_dims: List containing the number of unique categories for each categorical feature
        spec_path: Path to the JSON file containing the model architecture specification
        
    Returns:
        TabularNet model instance configured according to the JSON specification
    """
    #compute embedding output width
    emb_out_total = sum(infer_emb_dim(n) for n in cat_dims)

    #read and parse the json folder
    spec = json.loads(Path(spec_path).read_text())

    #pulling fields out of the spec
    layers_cfg = spec["layers"] 
    out_units = spec.get("output_units", 1)  # numbner of neurons on the final layer
    out_act_name = spec.get("output_activation", None)  #final layer activatoin function

    modules = []
    prev_dim = num_dim + emb_out_total

    #2 hidden layer loop
    for cfg in layers_cfg:
        units = cfg["units"]  # number of neurons in that unit
        act = cfg["activation"].lower()  #activtion function for that layer
        modules.append(nn.Linear(prev_dim, units))  #incoming matrix and neurons u want in this layer / shape coming out
        modules.append(_ACTIVATIONS[act]())
        prev_dim = units

    #3. ouput layers
    modules.append(nn.Linear(prev_dim, out_units))
    modules.append(_ACTIVATIONS[out_act_name]())

    mlp =  nn.Sequential(*modules) # the asterik tells Python to take each element of the list and feed it to the function one-by-one
    return TabularNet(num_dim, cat_dims, mlp)

def infer_emb_dim(n_categories: int) -> int:
    """
    Calculate the optimal embedding dimension for a categorical feature.
    
    Uses a rule of thumb formula to determine the appropriate embedding size
    based on the number of unique categories. The formula balances between
    having enough capacity to represent the categories and avoiding overfitting.
    
    Args:
        n_categories: Number of unique categories in the categorical feature
        
    Returns:
        Optimal embedding dimension (capped at 50) for the categorical feature
    """
    return int(min(50, round(1.6 * n_categories ** 0.56)))

