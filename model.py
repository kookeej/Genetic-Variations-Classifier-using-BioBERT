import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel

import gc

from config import DefaultConfig

cfg = DefaultConfig()


#id embedding model
class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.MODEL_NAME, config=config)
        self.id_embedding = nn.Embedding(num_embeddings=cfg.ID_DICT_LEN, 
                                           embedding_dim=768,
                                           padding_idx=1)
        self.sequential = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 9),
            nn.Softmax(1)
        )

    def forward(self, input_ids=None, attention_mask=None, ids=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )
        pooler = outputs[1]
        id_vec = self.id_embedding(ids)
        cat_vec = torch.cat([pooler, id_vec], dim=-1)
        logits = self.sequential(cat_vec)
        
        del pooler, outputs, id_vec, cat_vec

        return logits
    
# # No Embedding
# class CustomModel(nn.Module):
#     def __init__(self, config):
#         super(CustomModel, self).__init__()
#         self.model = AutoModel.from_pretrained(MODEL_NAME, config=config)
#         self.fc1 = nn.Linear(768, 128)
#         self.fc2 = nn.Linear(128, 9)
#         self.LN = nn.LayerNorm(128)
#         self.softmax = nn.Softmax(1)

#     def forward(self, input_ids=None, attention_mask=None, labels=None):
#         outputs = self.model(
#             input_ids, attention_mask=attention_mask
#         )
#         pooler = outputs[1]
#         pooler = self.LN(self.fc1(pooler))
#         pooler =self.fc2(pooler)
#         logits = self.softmax(pooler)
        
#         del pooler, outputs

#         return logits
    

