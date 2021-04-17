import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from Interact import generate_output
from dataImport import load_dataset
import random
from tabulate import tabulate

#returns model input without reply section
def prepare_main_model_input(evaluation_dataset):
    personas = evaluation_dataset.keys()
    model_inputs = {}
    persIds = []
    references = {}
    for person in personas:
        model_inputs[person] = []
        references[person] = []
        persIds.append(person)
        persona = evaluation_dataset[person]["PersonaID"]
        utterances = evaluation_dataset[person]["utterances"]
        for entry in utterances:

            model_inputs[person].append(" ".join(persona) + " ".join(entry["history"]))
            references[person].append(entry["reply"])
    return model_inputs, references, persIds


def assembleSamples():
    mainModelPath = "./TNG/TNGv5"
    altModelPath = "./TNG/TNGALTvs3"

    mainModel = GPT2LMHeadModel.from_pretrained(mainModelPath)
    mainTokenizer = GPT2Tokenizer.from_pretrained(mainModelPath)

    altModel = GPT2LMHeadModel.from_pretrained(altModelPath)
    altTokenizer = GPT2Tokenizer.from_pretrained(altModelPath)

    filepath = '../Dataset/HumanEvalSamples.json'
    data = load_dataset(filepath)

    MainData = data["TNGMain"]

    # For Human evaluation part 1
    #load training data

    filepath = '../Dataset/Train_eval_main.json'
    data = load_dataset(filepath)
    evaluation_dataset = data["Test"]

    #create 4 samples per persona.
    model_inputs, references, persIds = prepare_main_model_input(evaluation_dataset)
    MainSamples = {}
    """
    for person in persIds:
        indexes = []
        MainSamples[person] = {"context": [], "modelReply": [], "refReply": []}
        persString = " ".join(evaluation_dataset[person]["PersonaID"])
        while len(indexes) < 4:
            #generate random sample idx.
            idx = random.randint(0, len(model_inputs[person])-1)
            #if idx not indexes add sample and idx
            if idx not in indexes:
                indexes.append(idx)
                modelReply = generate_output(mainModel, mainTokenizer, model_inputs[person][idx], person)
                #append conversation context without persona string
                context = model_inputs[person][idx][len(persString):]
                MainSamples[person]["context"].append(context)
                MainSamples[person]["modelReply"].append(modelReply)
                MainSamples[person]["refReply"].append(references[person][idx])
    """

    print("Main Samples assembled.")

    # For Human evaluation part 2
    #Get files
    f = open("../Dataset/test_dataset.txt", "r")
    AltSamples = {persIds[0]: {"modelReply": [], "refReply": []}, persIds[1]: {"modelReply": [], "refReply": []}
        , persIds[2]: {"modelReply": [], "refReply": []}, persIds[3]: {"modelReply": [], "refReply": []}}
    samplePrep = {persIds[0]: [], persIds[1]: []
        , persIds[2]: [], persIds[3]: []}
    for x in f:
      for person in persIds:
          if person in x:
              samplePrep[person].append(x)

    for person in persIds:
        indexes = []
        while len(indexes) < 4:
            #generate random sample idx.
            idx = random.randint(0, len(samplePrep[person])-1)
            #if idx not indexes add sample and idx
            if idx not in indexes:
                indexes.append(idx)
                modelReply = generate_output(altModel, altTokenizer, "<bos> ", person)
                AltSamples[person]["modelReply"].append(modelReply)
                AltSamples[person]["refReply"].append(samplePrep[person][idx])


    print("altModel samples prepared")

    sampleDict = {"TNGMain" : MainData, "TNGAlt": AltSamples}

    with open('../Dataset/HumanEvalSamples.json', 'w', encoding='utf-8') as json_file:
      json.dump(sampleDict, json_file)

def make_tables():
    filepath = '../Dataset/HumanEvalSamples.json'
    data = load_dataset(filepath)

    MainData = data["TNGMain"]
    altData = data["TNGAlt"]

    personas = MainData.keys()
    for person in personas:
        context = ["Context"] + MainData[person]["context"]
        replies = ["Model Reply"] + MainData[person]["modelReply"]
       # ref = ["Ref Reply"] + MainData[person]["refReply"]
        info = [context,replies]
    #info = [['First Name', 'Mary', 'Jennifer'], ['Second Name', 'Smith', 'Jane', 'Doe'], [39, 25, 28]]
        print(tabulate(info))
#assembleSamples()

assembleSamples()