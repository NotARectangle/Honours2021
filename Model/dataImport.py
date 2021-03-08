import json
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from Model.personaID import prepare_inputs

# load dataset
def load_dataset(filePath):
    data = json.load(open(filePath, 'r'))

    return data

def prepare_inputs_from_data(data, model, tokenizer):
    input_dict = {"word_inputs": [], "Sequence": [], "positions": [], "token_type_ids":[] }
    persona = data["PersonaID"]
    utterances = data["utterances"]
    index = 0
    while index < len(utterances):
        history = utterances[index]["history"]
        reply = utterances[index]["reply"]
        #tokenize and build word sequence sing prepare inputs
        words, sequence, positions, token_type_ids = prepare_inputs(persona, history, reply, model, tokenizer)
        #Add inputs to input_dict
        input_dict["word_inputs"].append(words)
        input_dict["Sequence"].append(sequence)
        input_dict["positions"].append(positions)
        input_dict["token_type_ids"].append(token_type_ids)
        index += 1
        print(index)

    print("finished while prep input")
    return input_dict

# split dataset in train and test dataset
"""
filepath = '../Dataset/picardData.json'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
data = load_dataset(filepath)
input_dict = prepare_inputs_from_data(data)

print(input_dict["word_inputs"][0])
"""
