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
sentences = ["Three years later the coffin was still full of jello",
             "The fish dreamed of escaping the fishbowl",
             "The box was packed with jelly many months later",
             "Standing on one's head in a job interview",
             "It took him a month to finish the meal",
             "He is the brightest boy in class"]

tokens = {'input_ids':[], 'attention_mask':[]}

for sent in sentences:
    sent_token = tokenizer.encode_plus(sent, max_length=128,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(sent_token['input_ids'][0])
    tokens['attention_mask'].append(sent_token['attention_mask'][0])

import numpy as np


# Reformat the list of tensors into a single tensor
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

tokens['input_ids'].shape


# Get the encodings
outputs = model(**tokens)
outputs.keys()

embeddings = outputs.last_hidden_state

# check the shape of the encodings
embeddings.shape


#========================================================
# Get the attention masked embeddings (only valid words)
#========================================================

mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.shape).float()
mask.shape

masked_embeddings = embeddings * mask


#======================================================
# Get the mean
#======================================================

summed = torch.sum(masked_embeddings, 1)
counts = torch.clamp(mask.sum(1), min=1e-9)
counts.shape

mean_pooled = summed / counts


#======================================================
# Get the cosine similarity
#======================================================

from sklearn.metrics.pairwise import cosine_similarity

mean_pooled = mean_pooled.detach().numpy()

cosine_similarity([mean_pooled[0]], mean_pooled[1:])
