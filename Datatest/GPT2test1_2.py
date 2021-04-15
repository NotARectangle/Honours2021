from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

from transformers import TextDataset, DataCollatorForLanguageModeling


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

from transformers import Trainer, TrainingArguments, GPT2LMHeadModel

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model.to('cuda')

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,
    data_collator=data_collator,            # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

trainer.train()

trainer.save_model()
