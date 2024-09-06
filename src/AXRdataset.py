import torch
from torch.utils.data import Dataset
from process_data import *

class AXRDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input']
        output_text = self.data.iloc[idx]['output']
        instruction_text = self.data.iloc[idx]['instruction']
        prompt_text = self.data.iloc[idx]['prompt']
        
        return {'input': input_text, 'output': output_text, 'instruction': instruction_text, 'prompt': prompt_text}
