import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, AutoModel
import transformers

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
    device = torch.device("cpu")
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    # get tensors
   # print(input_ids)
   # model(input_ids=input_ids)
    """
    input_ids = torch.tensor(input_dict["input_ids"])
    lm_targets = torch.tensor(input_dict["lm_targets"])
    token_type_ids = torch.tensor(input_dict["token_type_ids"])
    #att_mask = torch.tensor(input_dict["att_mask"])
    """
    train_dataset = TensorDataset(*input_dict["train"])
    #convert to dataset.
    train_loader = DataLoader(train_dataset, batch_size=10)
    for epoch in range(3):
      print("Epoch :" + str(epoch))
        #  load input in batches
      for batch in train_loader:
        optim.zero_grad()
        input_ids, token_type_ids, labels = batch
       # labels = input_ids
        print(str(len(input_ids)) +" " + str(len(token_type_ids)) + " " + str(len(labels)))
        print(input_ids)
        outputs = model(input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
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