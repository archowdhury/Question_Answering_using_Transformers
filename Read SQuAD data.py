import os
import requests
import json

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
files = ['train-v2.0.json', 'dev-v2.0.json']
squad_dir = './data/squad'

# Make a directory for the data
#------------------------------
os.makedirs(squad_dir)


# Download the data
#------------------------------
for file in files:
    req = requests.get(url + file)
    with open(os.path.join(squad_dir, file), 'wb') as f:
        for chunk in req.iter_content(chunk_size=40):
            f.write(chunk)


# Load the JSON files into python
#--------------------------------
with open(os.path.join(squad_dir, files[0]), 'rb') as f:
    squad = json.load(f)

# Check the data loaded
#----------------------
squad['data'][0]['paragraphs'][0]
