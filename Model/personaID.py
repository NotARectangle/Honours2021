from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import re
import numpy as np



#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

#num_add_toks = tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})


def setSpecTokens(model, tokenizer):

    special_tokens = ["PICARD:", "OTHER:", "CRUSHER:", "TROI:", "RIKER:", "DATA:", "LAFORGE:", "WESLEY:", "Q:", "TASHA:", "WORF:"]
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "additional_special_tokens" : special_tokens})
    print(tokenizer.bos_token)
    print(tokenizer.eos_token)
    # adapt model to changes from tokenizer
    model.resize_token_embeddings(len(tokenizer))


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

def prepare_inputs(persona, history, reply, model, tokenizer):
    max_input = 1024  # GPT2 max input sequence length.
    string_input = persona + history + reply;

    # print(string_input)
    problemString = "Q:"

    # Get scene speakers
    speaker_token_re = "([A-Z]+ ?[A-Z]+ ?:)|[A-Z]:"
    speakers = []
    count = 0
    for input in string_input:

        if re.match(speaker_token_re, input):
            speaker = re.findall(speaker_token_re, input)
            if "AMANDA:" in speaker:
                print("attention")
            if len(speaker) > 0:
                for s in speaker:
                    if s not in tokenizer.additional_special_tokens and s != "":
                        string_input[count] = input.replace(s, "OTHER:")
        count += 1

    spek_tokens = tokenizer.additional_special_tokens

        # encode token type ids
    spek_token_ids = tokenizer.encode(spek_tokens)

    # encode all inputs
    try:
        sequence = []
        for s in string_input:
            sequence.append(tokenizer.encode(s))
    except:
        print("Exception occured.")
        print(sequence)
        print(s)

    currentSpeaker = spek_token_ids[0]  # start with selected character
    # currentSpeaker = 0
    token_type_ids = []
    words = []

    for seq in sequence:
        # type = 0
        #   if seq == sequence[(len(sequence)-1)]:
        # type = 1
        for token in seq:
            # cocacennate all tokens in sequence together
            #add to words
            words.append(token)
            if token in spek_token_ids:
                currentSpeaker = spek_token_ids[spek_token_ids.index(token)]
                token_type_ids.append(currentSpeaker)
            elif token is tokenizer.bos_token_id or token is tokenizer.eos_token_id:
                token_type_ids.append(token)
            else:
                token_type_ids.append(currentSpeaker)

    # check if inputs are to long.

    positions = list(range(len(words)))

    return words, sequence, positions, token_type_ids

# pad inputs to same length
def padding(encoded_input, tokenizer):
    input_len = []
    for e_input in encoded_input:
        input_len.append(len(e_input))


    # find the longes input
    maxLen = max(input_len)

    # pad all other inputs to longest lengths
    index = 0
    while index < len(encoded_input):
        if (len(encoded_input[index])) < maxLen:
            # pad difference between longest input and current input
            padding_length = maxLen - len(encoded_input[index])
            count = 0
            while count < padding_length:
                encoded_input[index].append(tokenizer.pad_token_id) #pad token id
                count += 1
        index += 1

    return encoded_input
