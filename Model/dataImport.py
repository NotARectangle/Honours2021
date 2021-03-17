import json
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from Model.personaID import prepare_inputs, padding
import numpy as np


# load dataset
def load_dataset(filePath):
    data = json.load(open(filePath, 'r', encoding="utf-8"))

    return data

def prepare_inputs_from_data(data, model, tokenizer):
    input_dict = {"input_ids": [], "lm_targets": [], "positions": [], "token_type_ids":[]
        , "mc_token_ids": [], "labels": []}#, "attention_mask": []}
    persona = data["PersonaID"]
    utterances = data["utterances"]
    index = 0
#    while index < len(utterances):
    trainingLen = len(utterances) - int(len(utterances) * 0.20)
    while index < trainingLen: #less to make it faster for testing
        history = utterances[index]["history"]
        reply = utterances[index]["reply"]
        #tokenize and build word sequence sing prepare inputs
        words, sequence, positions, token_type_ids = prepare_inputs(persona, history, reply, model, tokenizer)
        if len(words) < 1000:
            #Add inputs to input_dict
            input_dict["input_ids"].append(words)
            last_token = len(words)-1
            input_dict["mc_token_ids"].append(last_token)
            lm_targets = []
            labels = []
            #language modeling targets
            lm_targets = sequence[(len(sequence)-1)]
            lm_targets = lm_targets[:-1]

            in_reply = False
            for seq in sequence:
                # make labels pointing to reply
                j = 0
                label = -100
                # check if reply
                if seq == sequence[(len(sequence)-1)]:
                    #label = 10
                    in_reply = True
                while j < len(seq):
                    if in_reply is True:
                        label = seq[j]
                    labels.append(label)
                    j += 1

            input_dict["lm_targets"].append(lm_targets)
            input_dict["positions"].append(positions)
            input_dict["token_type_ids"].append(token_type_ids)
            input_dict["labels"].append(words)
        index += 1
        print(index)

    print("finished while prep input")

    #pad and then convert to tensors.
    for key in input_dict:
        if key != "mc_token_ids" and key != "attention_mask":
            paddedInput = padding(input_dict[key], tokenizer)

    count = 0
    while count < len(input_dict["labels"]):
        newLabels = []
        for i in input_dict["labels"][count]:
            if i == tokenizer.pad_token_id:
                newLabels.append(-100)
            else:
                newLabels.append(i)
        input_dict["labels"][count] = newLabels
        count+=1

    """
    # add attention mask
    att_Mask = []
    for input in input_dict["input_ids"]:
        item = []
        for i in input:
            if i != tokenizer.pad_token_id:
                item.append(1)
            else:
                item.append(0)
        att_Mask.append(item)

    input_dict["attention_mask"] = att_Mask
    print(input_dict["attention_mask"][2])
    print(input_dict["attention_mask"][1])
    """
    return input_dict

 #build tensor dataset.

# split dataset in train and test dataset

