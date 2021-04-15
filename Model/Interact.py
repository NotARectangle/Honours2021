import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, top_k_top_p_filtering
import re
from torch.nn import functional as F
import json

modelPath = "./TNG/MakeItSo2"

model = GPT2LMHeadModel.from_pretrained(modelPath)
tokenizer = GPT2Tokenizer.from_pretrained(modelPath)

"""
input_str = "PICARD: I am Captain Picard, commanding the Enterprise. PICARD: You will agree, Data, that Starfleet's orders are difficult? DATA: Difficult? Simply solve the mystery of Farpoint Station."

personaID = "PICARD: I am Captain Picard, commanding the Enterprise."
persona2 = "DATA: I am commander Data."
"""

def generate_output(history, persona):
    sequence = history + persona

    for i in range(150):

        input_ids = tokenizer.encode(sequence, return_tensors='pt')

        # get logits of last hidden state
        next_token_logits = model(input_ids).logits[:, -1, :]

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=0.95)

        # sample
        probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        n_token = next_token.tolist()[0][0]
        bos_token_id = tokenizer.bos_token_id
        if n_token == tokenizer.pad_token_id or n_token == bos_token_id:
            continue

        generated = torch.cat([input_ids, next_token], dim=-1)

        resulting_string = tokenizer.decode(generated.tolist()[0])

        #last_token = tokenizer.decode(next_token[0])

        sequence = resulting_string
        last_word = sequence[sequence.rindex(" ") + 1:]

        if re.match("([A-Z]+:)", last_word) or last_word in tokenizer.eos_token:
            print("matched")
            resulting_string = resulting_string[: sequence.rindex(last_word)]
            break;

    #just generated outcome
    resulting_string = resulting_string[resulting_string.rindex(persona):]
    return resulting_string


def startChat():
    #select personas
    print("Select Conversation partner")
    persona1, personaID = loadPersona()
    print("Select own persona")
    persona2, personaID2 = loadPersona()
    #start chat.
    history = ["<bos" + " ".join(personaID)] # bos at start of history
  #  history.append(personaID2[0]) # append first itroduction
    for i in range(5):

        userinput = input(persona2 + " ")

        history.append(persona2 + " " + userinput)
        context = " ".join(history)
        output = generate_output(context, persona1)
        history.append(output)
        print(output)
        #labels we can do labels of pad

def loadPersona():
    selected =""

    personaIDs = json.load(open('../Dataset/PersonaIDsCut.json', 'r'))

    personaKeys = personaIDs.keys()
    personas = ""
    for key in personaKeys:
        personas = personas + key[:-1] + " "
    persNum = len(personaKeys)
    PersonaMsg = "Please select a Persona. personas are " + personas+ "\nPlease type persona selected"
    valid = False
    while not valid:
        print(PersonaMsg)
        userinput = input()
        if userinput.lower() in personas.lower() and len(userinput) > 1:
            idx = personas.index(userinput.upper())
            selected = personas[idx: personas.index(" ", idx)] + ":"
            valid = True
        else:
            print("Not valid. Try again, personas are : " + personas)

    return selected, personaIDs[selected]

startChat()
