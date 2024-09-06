import torch
import pandas as pd
import numpy as np
from bert_tokenizer import load_list
from torch.utils.data import Dataset, DataLoader
import random

"""
dataset with labels
"""
class ImpressionsDataset(Dataset):
        """The dataset to contain report impressions and their labels."""
        
        def __init__(self, csv_path, list_path):
                self.df = pd.read_csv(csv_path)
                self.df =  self.df[['pneumatosis', 'pvg','freeair']]
                self.df.fillna(0, inplace=True) #blank label is 0
                self.encoded_imp = load_list(path=list_path)

        def __len__(self):
                return self.df.shape[0]

        def __getitem__(self, idx):
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                label = self.df.iloc[idx].to_numpy()
                label = torch.LongTensor(label)
                imp = self.encoded_imp[idx]
                imp = torch.LongTensor(imp)
                return {"imp": imp, "label": label, "len": imp.shape[0]}
        
        
class BootstrappedImpressionsDataset(ImpressionsDataset):
        """A dataset class that creates a bootstrap sample from the original dataset."""
        def __init__(self, csv_path, list_path,seed=42):
                super().__init__(csv_path, list_path)
                self.seed = seed
                self.bootstrap_indices = self._create_bootstrap_indices()
        def _create_bootstrap_indices(self):
                """Create bootstrap indices for the dataset."""
                n = len(self.df)
                random.seed(self.seed)
                return [random.randint(0, n - 1) for _ in range(n)]
        def __getitem__(self, idx):
                """Functionality to index into the bootstrapped dataset."""
                bootstrap_idx = self.bootstrap_indices[idx]
                return super().__getitem__(bootstrap_idx)
        