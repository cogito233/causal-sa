# This file is try to detect contradict in the outline, and try to fix it.
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import pickle

model = SentenceTransformer('all-mpnet-base-v2')

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model_contriever = AutoModel.from_pretrained('facebook/contriever')

def cosine_similarity(sentence_a, sentence_b):
    # Encode the two sentences into embeddings
    embeddings_a = model.encode(sentence_a, convert_to_tensor=True)
    embeddings_b = model.encode(sentence_b, convert_to_tensor=True)

    # change the ebmedding to cpu
    embeddings_a = embeddings_a.cpu()
    embeddings_b = embeddings_b.cpu()

    # Calculate the cosine similarity between the two embeddings
    similarity = 1 - cosine(embeddings_a, embeddings_b)

    return similarity

def cosine_similarity_contriever(sentence_a, sentence_b):
    embeddings_a = get_embedding_contriever(sentence_a)
    embeddings_b = get_embedding_contriever(sentence_b)
    similarity = 1 - cosine(embeddings_a, embeddings_b)
    return similarity

def get_embedding(sentence):
    embeddings = model.encode(sentence, convert_to_tensor=True)
    return embeddings.cpu()

def get_embedding_contriever(sentence):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
    outputs = model_contriever(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    #print(embeddings.shape)
    return embeddings[0].detach().numpy()

def similarity_from_embedding(embedding_a, embedding_b):
    similarity = 1 - cosine(embedding_a, embedding_b)
    return similarity

if __name__ == "__main__":
    # load sentence
    path = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data/yelp_sent_senti_pred_0609.csv"
    import pandas as pd
    df = pd.read_csv(path)
    print(df.head())
    from tqdm import trange
    result_list = [] # similarity, similarity_contriever, review_id, sentence_a, sentence_b, sentence_a_index, sentence_b_index
    for review_id in trange(100):
        subdf = df[df["review_id"] == review_id]
        if len(subdf) == 0:
            continue
        #print("review_id: ", review_id)
        length = len(subdf)
        embeddings = [[]]
        embeddings_contriever = [[]]
        for i in range(1, length):
            embeddings.append(get_embedding(subdf.iloc[i]["sentence_text"]))
            embeddings_contriever.append(get_embedding_contriever(subdf.iloc[i]["sentence_text"]))
        for i in range(1, length):
            for j in range(i+1, length):
                simlarity = similarity_from_embedding(embeddings[i], embeddings[j])
                simlarity_contriever = similarity_from_embedding(embeddings_contriever[i], embeddings_contriever[j])
                result_list.append([simlarity, simlarity_contriever, review_id, subdf.iloc[i]["sentence_text"], subdf.iloc[j]["sentence_text"], i, j])

    # Sort the result by similarity and print top 5
    result_list.sort(key=lambda x: x[0], reverse=True)
    print("similarity")
    for i in range(5):
        print(result_list[i])
    # Sort the result by similarity_contriever and print top 5
    result_list.sort(key=lambda x: x[1], reverse=True)
    print("similarity_contriever")
    for i in range(5):
        print(result_list[i])

