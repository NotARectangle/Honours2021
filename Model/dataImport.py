import json

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from Model.personaID import prepare_inputs, padding
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
        dict["Test"][persona] = {"PersonaID": data[persona]["PersonaID"], "utterances": u_train}
    return dict

def prepare_inputs_from_data(data, model, tokenizer):
    input_dict = {"input_ids": [], "token_type_ids":[]
        , "labels": []}
    persona = data["PersonaID"]
    utterances = data["utterances"]
    index = 0
#    while index < len(utterances):
    trainingLen = len(utterances) - int(len(utterances) * 0.25)
    #trainingLen = 15000
    while index < trainingLen: #less to make it faster for testing
        history = utterances[index]["history"]
        reply = utterances[index]["reply"]
        #tokenize and build word sequence sing prepare inputs
        words, sequence, positions, token_type_ids = prepare_inputs(persona, history, reply, model, tokenizer)
        if len(words) < 1000:
            #Add inputs to input_dict
            input_dict["input_ids"].append(words)
            last_token = len(words)-1
         #   input_dict["mc_token_ids"].append(last_token)
            labels = []

            in_reply = False
            for seq in sequence:
                # make labels pointing to reply
                j = 0
                label = -1
                # check if reply
                if seq == sequence[(len(sequence)-1)]:
                    #label = 10
                    in_reply = True
                while j < len(seq):
                    if in_reply is True:
                        label = seq[j]
                    labels.append(label)
                    j += 1

           # input_dict["lm_targets"].append(lm_targets)
         #   input_dict["positions"].append(positions)
            input_dict["token_type_ids"].append(token_type_ids)
            input_dict["labels"].append(words)
        index += 1
        print(index)

    print("finished while prep input")
    """
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

def convert_to_tensors(input_dict):
    tensor_dataset = {"train": []}
    length = []
    for item in input_dict["input_ids"]:
        length.append(len(item))
    print(length[0])

    for key in input_dict:
        unpad_tensors = []
        for item in input_dict[key]:
            tensor = torch.tensor(item)
            unpad_tensors.append(tensor)
        #pad tensor
        tensors = pad_sequence(unpad_tensors, padding_value=0)
        tensor_dataset["train"].append(tensors)

    return tensor_dataset

