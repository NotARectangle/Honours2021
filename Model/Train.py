# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, AutoModel, Trainer, TrainingArguments


def train(input_dict, model):
    print("start training method")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   # device = torch.device("cpu")
    print(device)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    train_dataset = input_dict

    #convert to dataset.
    train_loader = DataLoader(train_dataset, batch_size=10)
    train_loader = DataLoader(train_dataset, batch_size=10)
    print(train_dataset[0][0])

    for epoch in range(3):
      print("Epoch :" + str(epoch))
        #  load input in batches
      for batch in train_loader:
        optim.zero_grad()
        input_ids, token_type_ids, labels, attention_mask = batch
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids)
        loss = outputs.loss
        loss.backward()
        optim.step()
        print("step")

    print("Finished training")
    model.eval()

    model.save_pretrained("./TNG/MakeItSo4")

    print("model_changed?")




def config_dir():
    return "/MakeItSo"