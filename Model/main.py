from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, pipeline, AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch
from Model.Train import train
from Model.dataImport import load_dataset, prepare_inputs_from_data
from personaID import setSpecTokens

"""
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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
"""
#Model/TNG/MakeItSoTok
makeItSo = pipeline('text-generation',model='./TNG/MakeItSo', tokenizer='./TNG/MakeItSo', config={'max_length':1200})
#Model/TNG/MakeItSoTok/config.json
print(makeItSo("<bos> PICARD: make it so."))
print(makeItSo("<bos> RIKER: Are you alright Captain."))

print("Start")
tokenizer = GPT2Tokenizer.from_pretrained('./TNG/MakeItSo')

model = GPT2LMHeadModel.from_pretrained('./TNG/MakeItSo')

print("model loaded")

input_ids = tokenizer.encode("<bos> TROI: Or an incredibly powerful forcefield. But if we collide with either it could be", return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
#model.generate(input_ids)
print("done")
