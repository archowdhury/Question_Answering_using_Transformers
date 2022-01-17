import os
import requests
import json

squad_dir = './data/squad/'
files = ['train-v2.0.json', 'dev-v2.0.json']

# Load the train data
#--------------------
with open(squad_dir + files[0], 'rb') as f:
    train_raw = json.load(f)

with open(squad_dir + files[1], 'rb') as f:
    dev_raw = json.load(f)


# Extract the relevant sections from the training data
#-----------------------------------------------------

train_squad = []

for group in train_raw['data']:

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

            train_squad.append({'question':question, 'context':context, 'answer':answer})


# Extract the relevant sections from the dev/val data
#-----------------------------------------------------

dev_squad = []

for group in dev_raw['data']:

    for para in group['paragraphs']:
        context = para['context']

        for qa_pair in para['qas']:
            question = qa_pair['question']

            if 'answers' in qa_pair.keys() and len(qa_pair['answers']) > 0:
                answer = qa_pair['answers']
            elif 'probable_answers' in qa_pair.keys() and len(qa_pair['probable_answers']) > 0:
                answer = qa_pair['answers']
            else:
                answer = []

            answer = [item['text'] for item in answer]
            answer = list(set(answer))

            for ans in answer:
                dev_squad.append({'question':question, 'context':context, 'answer':ans})

# Save the data as JSON
#-----------------------

with open(os.path.join(squad_dir, 'train.json'), 'w') as f:
    json.dump(train_squad, f)

with open(os.path.join(squad_dir, 'dev.json'), 'w') as f:
    json.dump(dev_squad, f)