import os, json
import torch
import pickle
from sentence_transformers import SentenceTransformer
from alfred_utils.gen import constants
alfred_objects = set(constants.map_all_objects)
alfred_recep = set(constants.map_save_large_objects)
alfred_recep.add('FloorLamp')
len(alfred_recep)

import numpy as np

model = SentenceTransformer('bert-base-nli-mean-tokens')
recep_dict = dict()
for i in alfred_recep:
    recep_dict[i] = None
    
for k in recep_dict.keys():
    text = str(k)
    output = model.encode(text)
    recep_dict[k] = output
    
pickle.dump(recep_dict, open('models/BERT_retriever/recep_withFloorLamp_emb.p', 'wb'))

pickupable_dict = dict()
for i in alfred_objects:
    pickupable_dict[i] = None
    
for k in recep_dict.keys():
    text = str(k)
    output = model.encode(text)
    pickupable_dict[k] = output
    
pickle.dump(pickupable_dict, open('models/BERT_retriever/pickupable_emb.p', 'wb'))