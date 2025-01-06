import os, json
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def retrieve(candidate, objects_in_scene, ispickupable):

    def calculate_cosine_similarity(embedding_vector, dictionary_vectors):
        similarities = np.dot(embedding_vector, dictionary_vectors.T)
        return similarities

    def find_most_similar_keys(embedding_vector, embedding_dictionary, k=5):
        dictionary_vectors = np.array(list(embedding_dictionary.values()))
        similarities = calculate_cosine_similarity(embedding_vector, dictionary_vectors)
        most_similar_indices = np.argsort(similarities)[-k:][::-1]
        most_similar_keys = []
        for idx in most_similar_indices:        
            most_similar_keys.append(list(embedding_dictionary.keys())[idx])
        return most_similar_keys

    model = SentenceTransformer('bert-base-nli-mean-tokens')

    if ispickupable:
        train_dict = pickle.load(open('models/BERT_retriever/pickupable_NoLamp_emb.p', 'rb'))
    else:
        train_dict = pickle.load(open('models/BERT_retriever/recep_emb.p', 'rb'))
    
    emb_pool = dict()
    for k in objects_in_scene:
        if k in train_dict.keys():
            emb_pool[k] = train_dict[k]
        else:
            pass
    text = candidate
    print(emb_pool.keys())
    output = model.encode([text]).squeeze()

    k = 1 
    similar_keys = find_most_similar_keys(output, emb_pool, k)
    
    return similar_keys[0]
