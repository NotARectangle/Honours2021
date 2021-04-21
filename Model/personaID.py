# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import re
import numpy as np

#set special tokens
def setSpecTokens(model, tokenizer):

    special_tokens = ["PICARD:", "OTHER:", "CRUSHER:", "TROI:", "RIKER:", "DATA:", "LAFORGE:", "WESLEY:", "Q:", "TASHA:", "WORF:"]
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "additional_special_tokens" : special_tokens})
    # adapt model to changes from tokenizer
    model.resize_token_embeddings(len(tokenizer))

# samples of the input segments, structure ref:
# https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
def getSamples():
    # prepare a sample in the same structure as the json file
    persona = ["<bos>", "PICARD", "I'm Jean-Luc Picard, Captain of the Enterprise.",
               "I'm Captain Picard of the Enterprise.",
               "Welcome to the Enterprise. I'm Captain Picard."]
    history = ["PICARD: You will agree, Data, that Starfleet's orders are difficult? ",
               "DATA: Difficult? Simply solve the mystery of Farpoint Station. ",
               "PICARD: As simple as that. ",
               "TROI: Farpoint Station. Even the name sounds mysterious."]
    reply = ["PICARD: It's hardly simple, Data, to negotiate a friendly agreement for Starfleet to use the base "
             "while at the same time snoop around finding how and why the life form there built it. <eos>"]

    return persona, history, reply

# prepares segments into one sequence and returns token type id
def prepare_inputs(persona, history, reply, model, tokenizer):
    max_input = 1024  # GPT2 max input sequence length.
    string_input = persona + history + reply;

    # Get scene speakers
    speaker_token_re = "([A-Z]+ ?[A-Z]+ ?:)|[A-Z]:"
    speakers = []
    count = 0
    for input in string_input:

        if re.match(speaker_token_re, input):
            speaker = re.findall(speaker_token_re, input)
            for s in speaker:
                if s not in tokenizer.additional_special_tokens and s != "":
                    string_input[count] = input.replace(s, "OTHER:")
        count += 1

    spek_tokens = tokenizer.additional_special_tokens
    spek_token_ids = tokenizer.encode(spek_tokens)

    # encode all inputs
    sequence = [tokenizer.encode(s) for s in string_input]

    token_type_ids = []
    words = []

    for seq in sequence:
        t_type = 0
        if seq == sequence[(len(sequence) - 1)]:
            t_type = 1
        for token in seq:
            # cocacennate all tokens in sequence together
            words.append(token)
            token_type_ids.append(t_type)


    return words, sequence, token_type_ids

