import os
import requests
import json

squad_dir = './data/squad/'
files = ['train-v2.0.json', 'dev-v2.0.json']

# Load the train data
#--------------------
with open(squad_dir + files[0], 'rb') as f:
    squad = json.load(f)


# Convert the relevant data into a list
#--------------------------------------

new_squad = []

for group in squad['data']:

    for para in group['paragraphs']:
        context = para['context']

        for qa_pair in para['qas']:
            question = qa_pair['question']

            if 'answers' in qa_pair.keys() and len(qa_pair['answers']) > 0:
                answer = qa_pair['answers'][0]['text']
            elif 'probable_answers' in qa_pair.keys() and len(qa_pair['probable_answers']) > 0:
                answer = qa_pair['answers'][0]['text']
            else:
                answer = None

            new_squad.append({'question':question, 'context':context, 'answer':answer})

new_squad[0]

# Save the data as JSON
#-----------------------

with open(os.path.join(squad_dir, 'train.json'), 'w') as f:
    json.dump(new_squad, f)