import re

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
#Model/TNG/MakeItSoTok
makeItSo = pipeline('text-generation',model='./TNG/MakeItSo3', tokenizer='./TNG/MakeItSo3', config={'max_length':1600})
#Model/TNG/MakeItSoTok/config.json
print(makeItSo("<bos>PICARD: You will agree, Data, that Starfleet's orders are difficult? DATA: Difficult? Simply solve the mystery of Farpoint Station. " +
               "PICARD:"))


print(makeItSo("<bos> TROI: Or an incredibly powerful forcefield. But if we collide with either it could be dangerous. PICARD:"))

print(makeItSo("<bos> PICARD: I am Jean-Luc Picard, Captain of the Enterprise. RIKER: Are you alright Captain Picard? PICARD:"))

print(makeItSo("<bos> PICARD: I am Jean-Luc Picard, Captain of the Enterprise. DATA: Are you alright Captain Picard? PICARD:"))
print(makeItSo("<bos> PICARD: I am Jean-Luc Picard, Captain of the Enterprise. TROI: Are you alright Captain Picard? PICARD:"))

print("Start")
from transformers import top_k_top_p_filtering
from torch.nn import functional as F

tokenizer = GPT2Tokenizer.from_pretrained('./TNG/MakeItSo3')

model = GPT2LMHeadModel.from_pretrained('./TNG/MakeItSo3')
#model = GPT2DoubleHeadsModel.from_pretrained('./TNG/MakeItSo')
print("model loaded")

sequence = f"TROI: Or an incredibly powerful forcefield. But if we collide with either it could be dangerous."



for i in range(100):


    input_ids = tokenizer.encode(sequence, return_tensors='pt')

    # get logits of last hidden state
    next_token_logits = model(input_ids).logits[:, -1, :]

    #filter
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits,top_k=50, top_p=1.0)

    #sample
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    generated = torch.cat([input_ids, next_token], dim=-1)

    resulting_string = tokenizer.decode(generated.tolist()[0])

    last_token = tokenizer.decode(next_token[0])

    sequence = resulting_string
    last_word = sequence[sequence.rindex(" ")+1:]
   # last_word = " TROI:"
    if re.match("([A-Z]+:)", last_word):
        print("matched")
        resulting_string = resulting_string[: sequence.rindex(last_word)]
        break;

print(resulting_string)

sequence = "PICARD: You will agree, Data, that Starfleet's orders are difficult? DATA: Difficult? Simply solve the mystery of Farpoint Station. "
input_ids = tokenizer.encode(sequence, return_tensors='pt')
# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
#model.generate(input_ids)
print("done")

input_ids = tokenizer.encode(sequence, return_tensors='pt')
sample_outputs = model.generate(input_ids, do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=5)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



print("end")