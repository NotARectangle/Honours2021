from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

num_add_toks = tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})


def setSpecTokens(model, tokenizer):
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>"})
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

def prepare_inputs(persona, history, reply):
    #encoded_samples = tokenizer.tokenize(s)
    string_input = persona + history + reply;
    print(string_input)

    """Okay...
    We need the add characters in the scene that are not in special tokens into special tokens.
    then encode entire sequence as input_ids, 
    Then keep track of sequence of special tokens in scene , who speaks in what order, and all that they are speaking
    ex. 'DATA:', 'i', 'like', 'playing', 'football', 'PICARD:', 'I', 'don't'
    Data:, Data:, Data:, Data, Data,  Picard, picard, picard.
    Wee need the location of the last token <eos>
    """

# sample of what the data will look like to train the token embeddings.
sample = ["<bos> PICARD: You will agree, Data, that Starfleet's orders are difficult? ",
          "DATA: Difficult? Simply solve the mystery of Farpoint Station. ",
          "PICARD: As simple as that. ",
          "TROI: Farpoint Station. Even the name sounds mysterious. ",
          "PICARD: It's hardly simple, Data, to negotiate a friendly agreement for Starfleet to use the base while at the same time snoop around finding how and why the life form there built it. <eos>"]

#encoded_samples = [tokenizer.encode(s) for s in sample]
#print(encoded_samples)

persona, history, reply = getSamples()
prepare_inputs(persona, history, reply)
"""
add_tokens_loc = []
special_tokens = ["<bos>", "<eos>"]

for token in encoded_samples:
    if (tokenizer.bos_token_id in token):
        add_tokens_loc.append(token.index(tokenizer.bos_token_id))
    if (tokenizer.eos_token_id in token):
        add_tokens_loc.append(token.index(tokenizer.eos_token_id))

print(add_tokens_loc)
print(encoded_samples)
print(len(encoded_samples))
"""