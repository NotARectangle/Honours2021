# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from Model.personaID import prepare_inputs
from sklearn.model_selection import train_test_split


# load dataset
def load_dataset(filePath):
    data = json.load(open(filePath, 'r', encoding="utf-8"))

    return data

def seperate_train_test(data):
    personas = data.keys()
    dict = {"Train" : {}, "Test": {}}
    train_utt = []
    test_utt = []
    for persona in personas:
        utter = data[persona]["utterances"]
        u_train, u_test = train_test_split(utter, test_size=0.20)
        dict["Train"][persona] = {"PersonaID": data[persona]["PersonaID"], "utterances": u_train}
        dict["Test"][persona] = {"PersonaID": data[persona]["PersonaID"], "utterances": u_test}
    return dict

def prepare_inputs_from_data(data, model, tokenizer):
    input_dict = {"input_ids": [], "token_type_ids":[]
        , "labels": [], "attention_mask": []}
    for person in data:
        print(person)
        persona = data[person]["PersonaID"]
        utterances = data[person]["utterances"]
        index = 0
        while index < len(utterances):
            history = utterances[index]["history"]
            reply = utterances[index]["reply"]
            #tokenize and build word sequence sing prepare inputs
            words, sequence, token_type_ids = prepare_inputs(persona, history, reply, model, tokenizer)
            if len(words) < 300:
                #Add inputs to input_dict
                input_dict["input_ids"].append(words)
                last_token = len(words)-1
                labels = []
                current_pers = tokenizer.encode(person)
                current_pers = current_pers[0]
                in_reply = False
                for seq in sequence:
                    # make labels pointing to reply
                    j = 0
                    label = -100
                    # check if reply
                    if current_pers in seq:
                        # label = 10
                        in_reply = True
                    else:
                        in_reply = False
                        label = -100
                    while j < len(seq):
                        if in_reply is True:
                            label = seq[j]
                        labels.append(label)
                        j += 1

                input_dict["token_type_ids"].append(token_type_ids)
                input_dict["labels"].append(labels)
            else:
                print(tokenizer.decode(words))
        index += 1
        print(index)

    print("finished while prep input")

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
   # print(input_dict["attention_mask"])
    return input_dict

 #build tensor dataset.

# split dataset in train and test dataset

def convert_to_tensors(input_dict, pad_value):
    tensor_dataset = []
    length = []
    pad_v = pad_value
    for item in input_dict["input_ids"]:
        length.append(len(item))
    print(max(length))

    for key in input_dict:
        unpad_tensors = []
        for item in input_dict[key]:
            tensor = torch.tensor(item)
            unpad_tensors.append(tensor)
        #pad tensor
        #if key equals labels pad -100 to ignore for calculating loss
        if key == "labels":
            pad_v=-100
        elif key == "attention_mask":
            pad_v=0
        else:
            pad_v=pad_value
        tensors = pad_sequence(unpad_tensors, batch_first=True, padding_value=pad_v)
        tensor_dataset.append(tensors)

    return tensor_dataset

