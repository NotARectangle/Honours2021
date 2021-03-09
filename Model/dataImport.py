import json
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from Model.personaID import prepare_inputs, padding


# load dataset
def load_dataset(filePath):
    data = json.load(open(filePath, 'r', encoding="utf-8"))

    return data

def prepare_inputs_from_data(data, model, tokenizer):
    input_dict = {"input_ids": [], "lm_targets": [], "positions": [], "token_type_ids":[], "mc_token_ids": [], "att_mask": [] }
    persona = data["PersonaID"]
    utterances = data["utterances"]
    index = 0
#    while index < len(utterances):
    while index < 10: #less to make it faster for testing
        history = utterances[index]["history"]
        reply = utterances[index]["reply"]
        #tokenize and build word sequence sing prepare inputs
        words, sequence, positions, token_type_ids = prepare_inputs(persona, history, reply, model, tokenizer)
        if len(words) < 1020:
            #Add inputs to input_dict
            input_dict["input_ids"].append(words)
            last_token = len(words)-1
            input_dict["mc_token_ids"].append(last_token)
            lm_targets = []
            for seq in sequence:
                lm_targets = seq[:-1]

            input_dict["lm_targets"].append(lm_targets)
            input_dict["positions"].append(positions)
            input_dict["token_type_ids"].append(token_type_ids)
        index += 1
        print(index)

    print("finished while prep input")


    #pad and then convert to tensors.
    for key in input_dict:
        if key == "input_ids":
            paddedInput, att_mask = padding(input_dict[key])
            input_dict["att_mask"] = att_mask
         #   input_dict[key] = paddedInput
        elif key != "att_mask" and key != "mc_token_ids":
            paddedInput, _ = padding(input_dict[key])
           # input_dict[key] = paddedInput

    return input_dict

 #build tensor dataset.

# split dataset in train and test dataset

