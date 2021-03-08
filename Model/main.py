from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

from Model.dataImport import load_dataset, prepare_inputs_from_data
from personaID import setSpecTokens


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

setSpecTokens(model, tokenizer)
filepath = '../Dataset/picardData.json'
data = load_dataset(filepath)
input_dict = prepare_inputs_from_data(data, model, tokenizer)
print(input_dict["word_inputs"][0])