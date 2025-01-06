import os, json
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image

import numpy as np
from tqdm import tqdm

ALFRED_ROOT = os.environ['ALFRED_ROOT']


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors

def calculate_cosine_similarity(embedding_vector, dictionary_vectors):
    similarities = np.dot(embedding_vector, dictionary_vectors.T)
    return similarities

def get_similarites(embedding_vector, embedding_dictionary):
    dictionary_vectors = np.array(list(embedding_dictionary.values()))
    similarities = calculate_cosine_similarity(embedding_vector, dictionary_vectors)
    return similarities

def find_most_similar_keys(similarities, embedding_dictionary, k=9):
    most_similar_indices = np.argsort(similarities)[-k:][::-1]
    most_similar_keys = []
    for idx in most_similar_indices:        
        most_similar_keys.append(list(embedding_dictionary.keys())[idx])
    return most_similar_keys


device = 'cpu'
model = SentenceTransformer('bert-base-nli-mean-tokens')

model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

train_text_dict = pickle.load(open('planner/few_examples_from_song/train_few_instrucitons_emb.p', 'rb'))
train_img_dict = pickle.load(open('planner/few_examples_from_song/train_few_clip_image_panoramic_emb.p', 'rb'))


sps = ['tests_seen', 'tests_unseen', 'valid_seen', 'valid_unseen']

for sp in sps:
    langs_json = json.load(open(f'{sp}_langs.json'))
    sp_retrieved_keys = defaultdict(dict)

    IMG_WEIGHT = 1
    TXT_WEIGHT = 1

    for task, anns in tqdm(langs_json.items()):
        img_path = os.path.join(f'{ALFRED_ROOT}/data/json_2.1.0/{sp}', task, 'init_ego_panoramic.png')
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs_img = model_clip(**inputs)
        image_embeds = outputs_img.image_embeds.detach().cpu().numpy().squeeze()
        for r in anns.keys():
            text = anns[r]
            output = model.encode(text)
            k = 9 
            text_similarities = get_similarites(output, train_text_dict)
            img_similarities = get_similarites(image_embeds, train_img_dict)
            
            text_similarities_normalized = text_similarities / np.linalg.norm(text_similarities)
            img_similarities_normalized = img_similarities / np.linalg.norm(img_similarities)
            combined_similarity = IMG_WEIGHT * img_similarities_normalized + TXT_WEIGHT * text_similarities_normalized

            similar_keys = find_most_similar_keys(combined_similarity, train_text_dict, k=9)
            sp_retrieved_keys[task][r] = similar_keys
            
    with open(f'few_examples_from_song/few-song-{sp}_retrieved_keys_clip_Img{str(IMG_WEIGHT)}_Txt{str(TXT_WEIGHT)}_panoramic.json', 'w') as f:
        json.dump(sp_retrieved_keys, f, indent=4)

