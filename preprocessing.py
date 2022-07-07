import argparse
import pickle
from tqdm import tqdm
import gc

import pandas as pd
import numpy as np
import warnings
from ast import literal_eval
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DefaultConfig

cfg = DefaultConfig()
warnings.filterwarnings(action='ignore')



""" Train dataset Preprocessing """
def make_one_hot(dataset):
    one_hot = torch.zeros([len(dataset), 9])
    for i in range(len(dataset)):
        one_hot[i][dataset.iloc[i]['Class']-1] = 1
    return one_hot

def make_dataset():
    text_df = pd.read_csv("data/training_text", sep="\|\|", engine='python', names=["ID", "TEXT"], skiprows=1)
    var_df = pd.read_csv("data/training_variants", sep=",")
    dataset = pd.merge(var_df, text_df, on='ID', how='left')
    dataset['texts'] = " </s> " + dataset['Gene'] + " </s> " + dataset['Variation'] + ' </s> ' + dataset['TEXT']
    
    # 결측치 제거
    dataset = dataset.dropna()
    # 클래스 원핫 인코딩
    dataset['labels'] = make_one_hot(dataset).tolist()
    
    # 2000 글자씩 split
    n = 2000
    new_df = pd.DataFrame(columns={'ID', 'TEXT', 'NEW_TEXT', 'LABELS', 'CLASS'})
    for i in range(len(dataset)):
        result = [dataset.iloc[i]['TEXT'][k * n:(k + 1) * n] for k in range((len(dataset.iloc[i]['TEXT']) + n - 1) // n )] 
        for j in range(len(result)):
            item = {'ID': dataset.iloc[i]['ID'], 'TEXT': result[j], 
                     'NEW_TEXT': "</s> " + dataset.iloc[i]['Gene'] + " </s> " + dataset.iloc[i]['Variation'] + " </s> " + result[j],
                     'LABELS': dataset.iloc[i]['labels'],
                     'CLASS': dataset.iloc[i]['Class']}
            new_df = new_df.append(item, ignore_index=True)

    new_df = new_df.sample(frac=1).reset_index(drop=True)
    new_df.to_csv("data/train_split_aug.csv", index=False)

#     for i in range(len(dataset)):
#         new_df['LABELS'][i] = literal_eval(new_df.iloc[i]['LABELS'])
        
    return new_df



# 벡터화 및 인덱스 재정렬
def numerize(dataset):
    id_lst = list(set(dataset['ID']))
    id_num = np.arange(len(set(dataset['ID'])))
    id_dict = {}
    for i in range(len(id_lst)):
        id_dict[id_lst[i]] = id_num[i]
    dataset['ID'] = dataset['ID'].map(id_dict)
    return dataset




# Tokenizer
def tokenizing(dataset, mode):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, add_special_tokens=True)
    texts = dataset['NEW_TEXT'].tolist()
    ids = dataset['ID'].tolist()
    labels = None
    length = len(texts)

    if mode == "train":
        labels = torch.tensor(dataset['LABELS'].tolist())

    tokenized = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=False
    )

    return tokenized, labels, ids, length


# Dataset 구성.
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset, labels, ids, length, mode):
        self.tokenized_dataset = tokenized_dataset
        self.mode = mode
        self.length = length
        self.ids = ids
        if self.mode == "train":
            self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item['ids'] = self.ids[idx]
        if self.mode == "train":
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return self.length
    
    
def pro_dataset(dataset, batch_size, mode="train"):
    tokenized, labels, ids, length = tokenizing(dataset, mode=mode)
    custom_dataset = CustomDataset(tokenized, labels, ids, length, mode=mode)
    if mode == "train":
        OPT = True
    else:
        OPT = False
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    return dataloader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.1, help="train/test split size")

    args = parser.parse_args()
    
    print("Make Dataframe...")
    dataset = make_dataset()
    dataset = numerize(dataset)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=args.test_size, random_state=cfg.SEED)
    
    print("train dataset size: {}    |    valid dataset size: {}".format(len(train_dataset), len(valid_dataset)))
    
    print("Preprocessing dataset...")
    train_dataloader = pro_dataset(train_dataset, cfg.TRAIN_BATCH)
    valid_dataloader = pro_dataset(valid_dataset, cfg.VALID_BATCH)
    print("complete!")
    
    # Save DataLoader with pickle file.
    print("Save DataLoader...")
    gc.collect()
    pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
    pickle.dump(valid_dataloader, open('data/valid_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)   
    print("Data Preprocessing Complete!")     
