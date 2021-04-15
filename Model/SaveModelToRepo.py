from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
modelPath = "./TNG/MakeItSo2"

model = GPT2LMHeadModel.from_pretrained(modelPath)
tokenizer = GPT2Tokenizer.from_pretrained(modelPath)

model.save_pretrained("..\..\TNGMain")
tokenizer.save_pretrained("..\..\TNGMain")

print("modelUploaded?")