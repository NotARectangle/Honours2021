import torch

from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

# Add special tokens to vocabulary
spTokens = ["PICARD:", "DATA:", "TROI:", "LAFORGE:", "WORF:", "RIKER:", "CRUSHER:", "OTHER:"]

num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": spTokens})
