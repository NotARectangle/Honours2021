import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, top_k_top_p_filtering
import re
from torch.nn import functional as F

modelPath = "./TNG/MakeItSo2"

model = GPT2LMHeadModel.from_pretrained(modelPath)
tokenizer = GPT2Tokenizer.from_pretrained(modelPath)

input_str = "PICARD: I am Captain Picard, commanding the Enterprise. PICARD: You will agree, Data, that Starfleet's orders are difficult? DATA: Difficult? Simply solve the mystery of Farpoint Station."

personaID = "PICARD: I am Captain Picard, commanding the Enterprise."
persona2 = "DATA: I am commander Data."

"""
input_ids = tokenizer.encode(input_str, return_tensors='pt')

sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=100,
    top_k=50,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
"""

def generate_output(history):
    sequence = history + " PICARD:"

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
    resulting_string = resulting_string[resulting_string.rindex("PICARD:"):]
    return resulting_string


#give me for examples with same string
#for i in range(4):
#    generate_output(input_str)

#start chat.
history = [personaID, persona2]

for i in range(5):

    userinput = input("DATA: ")

    history.append("DATA: " + userinput)
    context = " ".join(history)
    output = generate_output(context)
    history.append(output)
    print(output)
    #labels we can do labels of pad