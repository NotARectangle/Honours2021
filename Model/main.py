from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')