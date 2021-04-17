import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from Interact import generate_output
from dataImport import load_dataset
from evaluation import prepare_model_input

mainModelPath = "./TNG/TNGv5"
altModelPath = "./TNG/TNGv5"

mainModel = GPT2LMHeadModel.from_pretrained(mainModelPath)
mainTokenizer = GPT2Tokenizer.from_pretrained(mainModelPath)

altModel = GPT2LMHeadModel.from_pretrained(mainModelPath)
altTokenizer = GPT2Tokenizer.from_pretrained(mainModelPath)

# For Human evaluation part 1
#load training data

filepath = '../Dataset/Train_eval_main.json'
data = load_dataset(filepath)
evaluation_dataset = data["Test"]

#create samples.


model_inputs, references, persIds = prepare_model_input(evaluation_dataset)

print(model_inputs[0])