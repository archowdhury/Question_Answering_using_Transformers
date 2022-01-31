from transformers import AutoModel, AutoTokenizer
import torch


#======================================================
# Initialize the tokenizer and model
#======================================================

model_name = "sentence-transformers/bert-base-nli-mean-tokens"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


#======================================================
# Get the text encoding
#======================================================

# Tokenize the text
text = "hello world what a time to be alive!"

tokens = tokenizer.encode_plus(text, max_length=128,
                               truncation=True, padding='max_length',
                               return_tensors='pt')

# Get the encodings
outputs = model(**tokens)
encodings = outputs.last_hidden_state

# check the shape of the encodings
encodings.shape