from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import re
import torch



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

num_add_toks = tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})


def setSpecTokens(model, tokenizer):
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "additional_special_tokens" : ["PICARD:"]})
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
    #encoded_samples = tokenizer.tokenize(s)
    string_input = persona + history + reply;
    #print(string_input)
    #Get scene speakers
    speaker_token_re = "([A-Z]+:)"
    speakers = []
    repeat = False
    for input in string_input:
        if re.match(speaker_token_re, input):
            speaker = re.findall(speaker_token_re, input)
            for s in speaker:
                if s not in speakers and s not in tokenizer.additional_special_tokens:
                    speakers = speakers + speaker

    spek_tokens = tokenizer.additional_special_tokens
    #Add to special tokens maybe at end
    if len(speakers) != 0:
        spek_tokens = tokenizer.additional_special_tokens + speakers
        tokenizer.add_special_tokens({"additional_special_tokens" : spek_tokens})
        model.resize_token_embeddings(len(tokenizer))

   # print(tokenizer.additional_special_tokens)

    #encode all inputs
    sequence = [tokenizer.encode(s) for s in string_input]
    #encode token type ids
    spek_token_ids = tokenizer.encode(spek_tokens)

    currentSpeaker = spek_token_ids[0] # start with selected character
    token_type_ids = []
    words = []
    for tokens in sequence:
        for token in tokens:
            #cocacennate all tokens in sequence together
            words.append(token)
            if token in spek_token_ids:
                currentSpeaker = spek_token_ids[spek_token_ids.index(token)]
            if token is tokenizer.bos_token_id or token is tokenizer.eos_token_id:
                token_type_ids.append(token)
            else:
                token_type_ids.append(currentSpeaker)
    """
    print(token_type_ids)
    print(spek_token_ids)
    #sequence
    print(sequence)
    print(words)
    """
    positions = list(range(len(words)))

    return words, sequence, positions, token_type_ids

# pad inputs to same length
def padding(encoded_input):
    input_len = []
    for input in encoded_input:
        input_len.append(len(input))

    # find the longes input
    maxLen = max(input_len)

    # pad all other inputs to longest lengths
    index = 0
    while index < len(encoded_input):
        if (len(encoded_input[index])) < maxLen:
            # pad difference between longest input and current input
            padding_length = maxLen - len(encoded_input)
            count = 0
            while count < padding_length:
                encoded_input[index].append(0)
                count += 1
        index += 1

    # add attention mask
    att_Mask = []
    for input in encoded_input:
        item = []
        for i in input:
            if i != 0:
                item.append(1)
            else:
                item.append(0)
        att_Mask.append(item)

    return encoded_input, att_Mask



"""
setSpecTokens(model, tokenizer)
persona, history, reply = getSamples()
words, sequence, positions, token_type_ids = prepare_inputs(persona, history, reply)

print("word length: " + str(len(words)) + " positions len " + str(len(positions)) + " sequence length " + str(len(token_type_ids)))
#pad input

last_token = len(words)-1
lm_targets = []
#build language modeling targets
for seq in sequence:
    lm_targets = (seq[:-1])


#print("Language targets: " + str(lm_targets))

#word tokens
input_ids = torch.tensor(words, dtype=torch.long)
#segement tokens
tok_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
#last token location
mc_token_ids = torch.tensor(last_token, dtype=torch.long)
#language modeling labels
lm_labels = torch.tensor(lm_targets, dtype=torch.long)
# Next-sentence prediction labels
#something mc_labels.... is not working
#mc_labels_s = [1]
#mc_labels = torch.tensor(mc_labels_s, dtype=torch.long)
outputs = model(input_ids, token_type_ids=tok_type_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels, )
"""