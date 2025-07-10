import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, csv_path: str, target: str):
      df = pd.read_csv(csv_path)

      #1. Separate labels     one big single 2d tensor whose first dimention indexes the row and second dimenion has length 1 (each same with 1 lavel)

      #turning the collum "target" into a 2d tensor of (n_rows, 1)
      self.y = torch.tensor(
         df.pop(target).values,  #removes label column from df
         dtype=torch.float32    
      ).unsqueeze(1)      #shape -> (n_rows, 1)

      #2. keep only numeric features
      X = df.select_dtypes(include="number").values.astype("float32")  # X is (num_rows, num_numeric_collumns)

      #3. z-score normalisation
      X = (X - X.mean(0)) / X.std(0).clip(1e-6)

      self.X = torch.tensor(X)

    def __len__(self):
       return len(self.y)   #number of rows total
    
    def __getitem__(self, idx):
       return self.X[idx], self.y[idx]     #one (featuer, label) pair
    
    #example, self.X = [-0.806, -0.392, -0.539] [ 0.453, 0.588, 0.312] (first two rows) self.Y = 245000. 312000. (2 rows which is price collumn)