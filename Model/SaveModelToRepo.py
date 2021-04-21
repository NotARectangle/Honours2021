# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021

from transformers import GPT2Tokenizer, GPT2LMHeadModel
modelPath = "./TNG/TNGv5"

# code to upload latest version to Hugging Face git for accessing the model in the demo.
model = GPT2LMHeadModel.from_pretrained(modelPath)
tokenizer = GPT2Tokenizer.from_pretrained(modelPath)

model.save_pretrained("..\..\TNGMain")
tokenizer.save_pretrained("..\..\TNGMain")

print("modelUploaded?")