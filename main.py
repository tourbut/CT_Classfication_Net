import pandas as pd
import os
import torch
import torch.nn as nn

import dataloader
import utils

from model import generate_model
from optimizer import Adam,SGD
from train_wrapper import train_epoch
import datetime

def load_config():
    config = utils.load_config()
    return config

if __name__ == "__main__":
    config = load_config()

    df_dataset = pd.read_csv(config['DATASET_PATH'])
    df_dataset = df_dataset.dropna().reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    X = df_dataset.drop(labels='label',axis=1)
    Y = df_dataset['label']

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=None,random_state=1004)

    traindata=dataloader.CTDataset(X_train,y_train)
    valdata=dataloader.CTDataset(X_test,y_test)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(traindata , batch_size=4, shuffle=True, sampler = None
                                ,num_workers=1,pin_memory = True)
    val_dataloader = DataLoader(valdata , batch_size=4, shuffle=True, sampler = None
                                ,num_workers=1,pin_memory = True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_name = config['model']['model_name']
    model_depth = config['model']['model_depth']

    model, _ = generate_model(model_name=model_name,model_depth = model_depth,n_classes=2,resnet_shortcut='B',add_last_fc_num = 0)
    model.to(device)

    optimizer = Adam(model, learning_rate = 0.001)
    criterion_clf = nn.CrossEntropyLoss().to(device)
    
    train_epoch(device,train_dataloader,val_dataloader,val_dataloader,model,criterion_clf,optimizer,config,epoch = 1,num_classes=2)