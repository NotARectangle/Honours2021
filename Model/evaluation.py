from datasets import load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from Model.Interact import generate_output
from Model.dataImport import load_dataset

metric = load_metric('sacrebleu')

#returns model input without reply section
def prepare_model_input(evaluation_dataset):
    personas = evaluation_dataset.keys()
    model_inputs = []
    references = []
    persIds = []
    for person in personas:

        persona = evaluation_dataset[person]["PersonaID"]
        utterances = evaluation_dataset[person]["utterances"]
        for entry in utterances:
            persIds.append(person)
            model_inputs.append(persona + entry["history"])
            str_ref = " ".join(persona + entry["history"] + entry["reply"])
            references.append([str_ref])
    return model_inputs, references, persIds

def runEval():
    #compare model inputs with references containing the reply.
    modelPath = "./TNG/MakeItSo2"
    model = GPT2LMHeadModel.from_pretrained(modelPath)
    tokenizer = GPT2Tokenizer.from_pretrained(modelPath)

    filepath = '../Dataset/Train_eval_main.json'
    data = load_dataset(filepath)
    evaluation_dataset = data["Test"]

    model_inputs, references, persIds = prepare_model_input(evaluation_dataset)

    print(model_inputs[0])
    print(references[0])

    index = len(model_inputs) -10

    while(index < len(model_inputs) and index < len(references)):
        input = " ".join(model_inputs[index])
        model_predictions = generate_output(input, persIds[index])
        print(model_predictions)
        metric.add_batch(predictions=[input + model_predictions], references=[references[index]])
        index += 1
        #print(index)

    #compute metric
    final_score = metric.compute()

    print(final_score)