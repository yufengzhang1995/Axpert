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
                """ Initialize the dataset object
                @param csv_path (string): path to the csv file containing labels
                @param list_path (string): path to the list of encoded impressions
                """
                self.df = pd.read_csv(csv_path)
                self.df =  self.df[['nec_features']]
                self.df.fillna(0, inplace=True) #blank label is 0
                self.encoded_imp = load_list(path=list_path)

                
        def __len__(self):
                """Compute the length of the dataset

                @return (int): size of the dataframe
                """
                return self.df.shape[0]
        
        def __getitem__(self, idx):
                """ Functionality to index into the dataset
                @param idx (int): Integer index into the dataset

                @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                                a LongTensor of an encoded impression. The value of 'label'
                                is a LongTensor containing the labels and 'the value of
                                'len' is an integer representing the length of imp's value
                """
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

