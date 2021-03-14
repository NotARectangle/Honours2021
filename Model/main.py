from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, pipeline
import torch
from Model.Train import train
from Model.dataImport import load_dataset, prepare_inputs_from_data
from personaID import setSpecTokens

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

setSpecTokens(model, tokenizer)
filepath = '../Dataset/picardData.json'
data = load_dataset(filepath)
input_dict = prepare_inputs_from_data(data, model, tokenizer)
#print(input_dict["input_ids"][0])
tensor_dataset = {"train": []}
length = []
for item in input_dict["input_ids"]:
    length.append(len(item))
print(length)


for key in input_dict:
    tensor = torch.tensor(input_dict[key])
    tensor_dataset["train"].append(tensor)

print(input_dict["labels"][1])

train(tensor_dataset, model)

tokenizer.save_pretrained("./TNG/MakeItSo")
#Model/TNG/MakeItSoTok
makeItSo = pipeline('text-generation', model='./TNG/MakeItSo', tokenizer="./TNG/MakeItSo")
#Model/TNG/MakeItSoTok/config.json
print(makeItSo("PICARD: make it so."))
print(makeItSo("RIKER: Are you alright Captain."))
