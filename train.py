import argparse
import pickle
from tqdm import tqdm
import gc
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset
from utils import get_criterion, get_optimizer, get_scheduler, seed_everything

from colorama import Fore, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
r_ = Fore.RED
sr_ = Style.RESET_ALL

# Settings
cfg = DefaultConfig()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(cfg.SEED)



def train(train_dataloader, valid_dataloader, args):
    # 모델 로딩
    model = CustomModel(config=cfg.MODEL_CONFIG)
    model.parameters
    model.to(device)

    # cost function
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_dataloader, args)

    gc.collect()
    train_total_loss = []
    valid_total_loss = []

    best_val_loss = np.inf

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")

        # 학습 단계 loss
        train_loss_value = 0
        train_epoch_loss = []

        # 검증 단계 loss
        valid_loss_value = 0
        valid_epoch_loss = []


        gc.collect()
        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            item = {key: val.to(device) for key, val in items.items()}
            optimizer.zero_grad()
            outputs = model(**item)
            loss = criterion(outputs, item['labels'])

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_value += loss.item()
            if (idx + 1) % cfg.TRAIN_LOG_INTERVAL == 0:
                train_bar.set_description("Loss: {:3f}".\
                    format(train_loss_value/cfg.TRAIN_LOG_INTERVAL))
                train_epoch_loss.append(train_loss_value/cfg.TRAIN_LOG_INTERVAL)
                train_loss_value = 0

                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        del loss, item, outputs
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            print(f"{b_}---- Validation.... ----{sr_}")
            model.eval()
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                item = {key: val.to(device) for key,val in items.items()}
                outputs = model(**item)
                loss = criterion(outputs, item['labels'])

                valid_loss_value += loss.item()
                if (idx + 1) % cfg.VALID_LOG_INTERVAL == 0:
                    valid_bar.set_description("Loss: {:3f}".\
                        format(valid_loss_value/cfg.VALID_LOG_INTERVAL))
                    valid_epoch_loss.append(valid_loss_value/cfg.VALID_LOG_INTERVAL)
                    valid_loss_value = 0
            del loss, item, outputs
            print("{}Best Loss: {:3f}    |    This epoch Loss: {:3f}".format(g_, best_val_loss, (sum(valid_epoch_loss)/len(valid_epoch_loss))))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), cfg.MODEL_PATH)
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))

            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
        print()
    del train_total_loss, valid_total_loss, train_loss_value, train_epoch_loss, valid_loss_value, valid_epoch_loss, train_bar, valid_bar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    train_dataloader = pickle.load(open('data/train_dataloader.pkl', 'rb'))
    valid_dataloader = pickle.load(open('data/valid_dataloader.pkl', 'rb'))

    train(train_dataloader, valid_dataloader, args)