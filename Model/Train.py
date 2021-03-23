import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, AutoModel, Trainer, TrainingArguments


"""
class tng_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])
"""

def train(input_dict, model):
    print("start training method")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   # device = torch.device("cpu")
    print(device)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    train_dataset = TensorDataset(*input_dict["train"])
    test_dataset = TensorDataset(*input_dict["test"])
   # train_dataset = tng_dataset(input_dict["train"])
  #  test_dataset = tng_dataset(input_dict["test"])
    #convert to dataset.
    train_loader = DataLoader(train_dataset, batch_size=10)
    print(train_dataset[0][0])

    for epoch in range(3):
      print("Epoch :" + str(epoch))
        #  load input in batches
      for batch in train_loader:
        optim.zero_grad()
        input_ids, token_type_ids, labels, attention_mask = batch
        input_ids = input_ids.to(device)
        """
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        """
        outputs = model(input_ids=input_ids)
        loss = outputs.loss
        loss.backward()
        optim.step()
        print("step")
       # print(outputs.loss)


        #lm_logits = outputs.logits
        #mc_logits = outputs.mc_logits
        #optim.step()

    print("Finished training")
    model.eval()

    model.save_pretrained("./TNG/MakeItSo4")

    print("model_changed?")




def config_dir():
    return "/MakeItSo"