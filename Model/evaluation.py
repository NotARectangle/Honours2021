import json
import re

from datasets import load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering
from torch.nn import functional as F
import torch
from Model.dataImport import load_dataset

metric = load_metric('sacrebleu')

# adding an amended version of the generation method, where more of the string sequence can be set.
# This since blue scores calculate if the ref is matched exactly, part of the sequence will be set from the ref text
def generate_output(model, tokenizer, history, persona, seqStart):
    sequence = history + persona + " " + seqStart
    resulting_string = " "
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

        sequence = resulting_string
        last_word = sequence[sequence.rindex(" ") + 1:]

        if re.match("([A-Z]+:)", last_word) or last_word in tokenizer.eos_token:
            resulting_string = resulting_string[: sequence.rindex(last_word)]
            break;

    #just generated outcome
    if persona in resulting_string:
        resulting_string = resulting_string[resulting_string.rindex(persona):]
    return resulting_string


#returns model input without reply section
def prepare_model_input(evaluation_dataset):
    personas = evaluation_dataset.keys()
    model_inputs = {}
    persIds = []
    references = {}
    for person in personas:
        persIds.append(person)
        model_inputs[person] = []
        references[person] = []
        persona = evaluation_dataset[person]["PersonaID"]
        utterances = evaluation_dataset[person]["utterances"]
        for entry in utterances:
            model_inputs[person].append(persona + entry["history"])
            str_ref = " ".join(persona + entry["history"] + entry["reply"])
            references[person].append([str_ref])
    return model_inputs, references, persIds

def prep_alt_inputs():
    #Get files
    f = open("../Dataset/test_dataset.txt", "r")
    persIds = ["PICARD:", "TROI:", "DATA:", "RIKER:"]

    samplePrep = {persIds[0]: [], persIds[1]: []
        , persIds[2]: [], persIds[3]: []}

    for x in f:
      for person in persIds:
          if person in x:
              #take out eos and \n since thouse won't be predicted by the evaluation
              subX = x[:x.index("<eos>")]
              samplePrep[person].append(subX)

    return samplePrep

def runEval():
    #compare model inputs with references containing the reply.
    mainModelPath = "./TNG/TNGv5"
    altModelPath = "./TNG/TNGALTvs3"

    mainModel = GPT2LMHeadModel.from_pretrained(mainModelPath)
    mainTokenizer = GPT2Tokenizer.from_pretrained(mainModelPath)

    altModel = GPT2LMHeadModel.from_pretrained(altModelPath)
    altTokenizer = GPT2Tokenizer.from_pretrained(altModelPath)

    #score dict recorded.
    score = {"MainModel": {}, "AltModel": {}}

    #Run blue evaluation for Main Model

    filepath = '../Dataset/Train_eval_main.json'
    data = load_dataset(filepath)
    evaluation_dataset = data["Test"]

    model_inputs, references, persIds = prepare_model_input(evaluation_dataset)

    """
    for person in persIds:

        score["MainModel"][person] = []
        # get 100 samples per person to calculate scores
        index = len(model_inputs[person]) -100
        print(len(model_inputs[person]))

        while(index < len(model_inputs[person]) and index < len(references[person])):
            input = " ".join(model_inputs[person][index])
            #don't supply sequence start. Let model predict full reply sentence
            model_predictions = generate_output(mainModel, mainTokenizer, input, person, " ")
            metric.add_batch(predictions=[input + model_predictions], references=[references[person][index]])
            index += 1

        print("Metric for person: " + person + " calculated.")
        # compute metric
        main_final_score = metric.compute()
        score["MainModel"][person] = main_final_score
    """
    # Evaluate second model
    refInputs = prep_alt_inputs()

    for person in persIds:
        score["AltModel"][person] = []
        index = 0
        batch_size = 30
        while (index < len(refInputs[person])):
            #input = refInputs
            count = 0
            refList = []
            if (index + batch_size) < len(refInputs[person]):
                #convert string ref to list.
                while count < batch_size:
                    refList.append(refInputs[person][index + count])
                    count +=1
                # take out last word and take out bos and persona
               # splitRef = refList[2:-1]
                #seqStart = " ".join(splitRef)
                model_predictions = generate_output(altModel, altTokenizer, "<bos> ", person, " ")
                metric.add_batch(predictions=["<bos> " + model_predictions], references=[refList])
                index += batch_size
            else:
                index = len(refInputs[person])

        main_final_score = metric.compute()
        score["AltModel"][person] = main_final_score
        print("Metric for person: " + person + " calculated.")

    print(score)
    """
        #print scores
    with open('../Dataset/EvalResults.json', 'w', encoding='utf-8') as json_file:
        json.dump(score, json_file)
    print(score)
"""

runEval()