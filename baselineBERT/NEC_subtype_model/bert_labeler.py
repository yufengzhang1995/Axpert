import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class bert_labeler(nn.Module):
    def __init__(self, 
                 p=0.1, 
                 clinical=False, 
                 freeze_embeddings=False, 
                 pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with 
            transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. 
                Ignored if pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 3, bias=True) for _ in range(3)])

    def forward(self, 
                source_padded, 
                attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 3 containing tensors.  
        """
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(3):
            out.append(self.linear_heads[i](cls_hidden))
        return out
    
if __name__ == "__main__":
### test purpose ###
    
    import torch
    import random


    random.seed(42)
    torch.manual_seed(42)


    batch_size = 2
    max_len = 10
    vocab_size = 30522  

    source_padded = torch.randint(0, vocab_size, (batch_size, max_len))
    attention_mask = torch.ones(batch_size, max_len)
    for i in range(batch_size):
        pad_idx = (source_padded[i] == 0).nonzero(as_tuple=True)[0]
        if len(pad_idx) > 0:
            pad_idx = pad_idx[0]
            attention_mask[i, pad_idx:] = 0

    print("source_padded:", source_padded)
    print("attention_mask:", attention_mask)

    model  = bert_labeler(clinical = True)

    model(source_padded, 
                    attention_mask)