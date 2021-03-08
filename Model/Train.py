import torch
from torch.utils.data import DataLoader
from transformers import AdamW

class tng_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def train(input_dict, model):
    print("start training method")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    # get tensors
    train_dataset = tng_dataset(input_dict)

    """
    input_ids = torch.tensor(input_dict["word_inputs"])
    lm_targets = torch.tensor(input_dict["lm_targets"])
    token_type_ids = torch.tensor(input_dict["token_type_ids"])
    #att_mask = torch.tensor(input_dict["att_mask"])
    """
    #convert to dataset.
    train_loader = DataLoader(train_dataset, True)
    for epoch in range(3):
      #  load input in batches
      for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        lm_targets = batch["lm_targets"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        #last token
        mc_token_ids = batch["mc_token_ids"].to(device)
        outputs = model(input_ids= input_ids, lm_targets=lm_targets, mc_token_ids=mc_token_ids, token_type_ids=token_type_ids)
        loss = outputs[0]
        loss.backward()
        optim.step()

    print("Finished training")

def config_dir():
    return "/MakeItSo"