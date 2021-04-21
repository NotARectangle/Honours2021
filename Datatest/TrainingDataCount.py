import json

def count_alt_inputs(datafile):
    #Get files
    f = open(datafile)
    persIds = ["PICARD:", "TROI:", "DATA:", "RIKER:"]

    samplePrep = {persIds[0]: [], persIds[1]: []
        , persIds[2]: [], persIds[3]: []}

    print("file: " + datafile)
    for x in f:
      for person in persIds:
          if person in x:
              samplePrep[person].append(x)

    #print length of training data
    for person in persIds:
        print(person + "data length:" + str(len(samplePrep[person])))
    print()

def count_dataset_entries(datasetPath):
    data = json.load(open(datasetPath, 'r', encoding="utf-8"))
    type = data.keys()
    model_inputs = {}
    for t in type:
        print("MainModel dataset type: " + t)
        personas = data[t].keys()
        for person in personas:
            utterances = data[t][person]["utterances"]
            print(person + "data length:" + str(len(utterances)))


count_alt_inputs("../Dataset/test_dataset.txt")
count_alt_inputs("../Dataset/train_dataset.txt")

count_dataset_entries("../Dataset/train_eval_main.json")