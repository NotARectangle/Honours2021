import json
import re

from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, pipeline, AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch
from Model.Train import train
from Model.dataImport import load_dataset, prepare_inputs_from_data, seperate_train_test, convert_to_tensors
from personaID import setSpecTokens

"""
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

setSpecTokens(model, tokenizer)
filepath = '../Dataset/picardData2.json'
data = load_dataset(filepath)
input_dict = prepare_inputs_from_data(data, model, tokenizer)
#print(input_dict["input_ids"][0])
tensor_dataset = {"train": []}
length = []
for item in input_dict["input_ids"]:
    length.append(len(item))
print(length[0])

input = tokenizer.decode(input_dict["input_ids"][0])
print("Input = " + input )
print("Input = " + str(input_dict["input_ids"][0]))

for key in input_dict:
    tensor = torch.tensor(input_dict[key])
    tensor_dataset["train"].append(tensor)

train(tensor_dataset, model)

tokenizer.save_pretrained("./TNG/MakeItSo3")
"""
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

filepath = '../Dataset/tngPersonaData.json'
data = load_dataset(filepath)

newDict = seperate_train_test(data)

test = newDict["Train"]["PICARD:"]
input_dict = prepare_inputs_from_data(test, model, tokenizer)

tensor_dataset = convert_to_tensors(input_dict)

train(tensor_dataset, model)