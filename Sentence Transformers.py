from sentence_transformers import SentenceTransformer
import torch


#======================================================
# Initialize the tokenizer and model
#======================================================

model_name = "bert-base-nli-mean-tokens"

model = SentenceTransformer(model_name)


#======================================================
# Get the sentence encoding
#======================================================

sentences = ["Three years later the coffin was still full of jello",
             "The fish dreamed of escaping the fishbowl",
             "The box was packed with jelly many months later",
             "Standing on one's head in a job interview",
             "It took him a month to finish the meal",
             "He is the brightest boy in class"]


embeddings = model.encode(sentences)

# check the shape of the encodings
embeddings.shape

#======================================================
# Get the cosine similarity
#======================================================

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([embeddings[0]], embeddings[1:])
