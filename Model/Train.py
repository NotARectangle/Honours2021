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
    device = torch.device("cpu")
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    train_dataset = TensorDataset(*input_dict["train"])
    test_dataset = TensorDataset(*input_dict["test"])
    #convert to dataset.
    train_loader = DataLoader(train_dataset, batch_size=10)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset  # evaluation dataset
    )

    trainer.train()
    """
    for epoch in range(3):
      print("Epoch :" + str(epoch))
        #  load input in batches
      for batch in train_loader:
        optim.zero_grad()
        input_ids, token_type_ids, labels = batch
        outputs = model(input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
        # lm_coef = 2.0 #language modeling weight higher than clasification head weight
        # mc_coef = 1.0
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
    """



def config_dir():
    return "/MakeItSo"