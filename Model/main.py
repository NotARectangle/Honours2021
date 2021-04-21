# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021

import json
import re

from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, pipeline, AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch
from Model.Train import train
from Model.dataImport import load_dataset, prepare_inputs_from_data, seperate_train_test, convert_to_tensors
from personaID import setSpecTokens


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

setSpecTokens(model, tokenizer)
filepath = '../Dataset/tngPersonaData.json'
data = load_dataset(filepath)

newDict = seperate_train_test(data)

train_dict = newDict["Train"]
test_dict = newDict["Test"]
#store train and test dicionary
with open('../Dataset/Train_test_set.json', 'w', encoding='utf-8') as json_file:
  json.dump(newDict, json_file)

#prepare inputs
input_dict_train = prepare_inputs_from_data(train_dict, model, tokenizer)
input_dict_test = prepare_inputs_from_data(test_dict, model, tokenizer)

pad_value = tokenizer.pad_token_id



train_tensor_dataset = convert_to_tensors(input_dict_train, pad_value)
test_tensor_dataset = convert_to_tensors(input_dict_test, pad_value)

tensor_dataset = {"train" : train_tensor_dataset, "test" : test_tensor_dataset}


train(train_tensor_dataset, model)
