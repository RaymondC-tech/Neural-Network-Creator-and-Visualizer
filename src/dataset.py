import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data with both numeric and categorical features.
    
    Handles preprocessing of tabular data including:
    - Separation of numeric and categorical features
    - Z-score normalization for numeric features  
    - Encoding categorical features as integer indices
    - Creation of embedding mappings for categorical variables
    """
    def __init__(self, csv_path: str, target: str):
        """
        Initialize the TabularDataset from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing the tabular data
            target: Name of the target/label column to predict
        """
        df = pd.read_csv(csv_path)

        #1. Separate labels     one big single 2d tensor whose first dimention indexes the row and second dimenion has length 1 (each same with 1 lavel)
        #turning the collum "target" into a 2d tensor of (n_rows, 1)
        self.y = torch.tensor(
           df.pop(target).values,  #removes label column from df
           dtype=torch.float32    
        ).unsqueeze(1)      #shape -> (n_rows, 1)


        #2 separate numerical vs categorical
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols and c != target]

        self.num_cols = num_cols
        self.cat_cols = cat_cols


        # 2. numeric tensor with z-score
        X_num = df[num_cols].values.astype("float32")
        X_num = ( X_num - X_num.mean(0)) / X_num.std(0).clip(10e-6)
        self.X_num = torch.tensor(X_num)

        # 3. categorical to integer ids
        if cat_cols:
            cat_arrays = []
            self.cat_maps = {}
            for c in cat_cols:
                uniques = df[c].astype("category").cat.categories  #after that uniques is jsut sroted list of unique strings in that column
                self.cat_maps[c] = {k: i for i, k in enumerate(uniques)}  #dictoiary of things to its number id
                cat_arrays.append(df[c].map(self.cat_maps[c]).values)   #FIXED: removed .to_numpy() which doesn't exist on dict
            
            X_cat = np.stack(cat_arrays, axis=1).astype("int64")  #FIXED: moved outside the for loop
            self.X_cat = torch.tensor(X_cat)
        else:
            self.X_cat = torch.empty(len(df), 0, dtype=torch.int64)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            Total number of rows/samples in the dataset
        """
        return len(self.y)   #number of rows total
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - "num": Numeric features tensor for the sample
                - "cat": Categorical features tensor for the sample  
                - "y": Target/label tensor for the sample
        """
        return {
           "num": self.X_num[idx], #self.X_num is 2d torch tensor of all the numiercal columns with z-score normalizatoin adn each array is a row
           "cat": self.X_cat[idx], # self.X_cat is a 2d torch tensor that strore all categorical columns (BOTH NUM AND CAT IS (n, n_numeric/n_categorical))
           "y": self.y[idx]   #2d torch tensor /the target column giving back the actual thing of the prediction   (N, 1)
        }

       # self.X_num:
       # e.x: tensor([[-1.03, -0.71],     # row-0
       #              [ 1.35,  1.41],     # row-1
       #              [-0.32, -0.71]])    # row-2

       # self.X_cat
       # tensor([[0],   row 0
       #  [1],          row 1
       #  [0]])         row 2


        #return self.X[idx], self.y[idx]     #one (featuer, label) pair
    
    #example, self.X = [-0.806, -0.392, -0.539] [ 0.453, 0.588, 0.312] (first two rows) self.Y = 245000. 312000. (2 rows which is price collumn)