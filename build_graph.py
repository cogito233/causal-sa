from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
# TODO: 算一下不同similarity的分布

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

def semantic_similarity(sentence_a, sentence_b):
    # TODO: another way to calculate the similarity, focus on the semantic structure
    pass

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


import nltk
import re
from nltk.tokenize import sent_tokenize

from transformers import GPT2Tokenizer
# 加载GPT-2的tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer_gpt.pad_token = tokenizer.eos_token


def calculate_sentence_length(sentence):
    # Return the length of the sentence
    encoded_input = tokenizer_gpt.encode(sentence)
    return len(encoded_input)


def custom_sentence_tokenize(text):
    # 根据句号分割句子
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    # 进一步根据逗号分割句子
    final_sentences = []
    for sentence in sentences:
        comma_split = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=;)\s', sentence)
        final_sentences.extend(comma_split)

    return final_sentences

def calculate_similarity_matrix(review, label, idx):
    # Return a list of diagonal similarity and a list of similarity matrix
    from yelp_subsample import split_to_sentences
    sentences = split_to_sentences(review)
    if len(sentences) < 5: # filter the review that is too short
        return [], []
    # Calculate the similarity matrix
    sentences_embedding = []
    for sentence in sentences:
        embedding = get_embedding(sentence)
        sentences_embedding.append(embedding)
    similarity_diagonal, similarity_matrix = [], []
    for i in range(len(sentences_embedding)):
        for j in range(i+1, len(sentences_embedding)):
            similarity = similarity_from_embedding(sentences_embedding[i], sentences_embedding[j])
            similarity_dict = {
                'review_id': idx,
                'similarity': similarity,
                'label': label,
                'idx1': i,
                'idx2': j
            }
            if i+1 == j:
                similarity_diagonal.append(similarity_dict)
            similarity_matrix.append(similarity_dict)
    return similarity_diagonal, similarity_matrix

import numpy as np
import pandas as pd

def saveToCSV_overall(outline_list, name):
    from pandas import DataFrame
    keys = outline_list[0].keys()
    result_dict = {key: [] for key in keys}
    for outline in outline_list:
        for key in keys:
            result_dict[key].append(outline[key])
    df = DataFrame(result_dict)
    #df = df.sort_values(by=['similarity'], ascending=True)
    path = "reformated_data/" + name + ".csv"
    print("Saving to ", path)
    df.to_csv(path, index=False)
    return df

if __name__ == '__main__':
    from yelp_split import load_yelp
    dataset = load_yelp()

    #train_dataset = dataset['train']
    test_dataset = dataset['test']
    similarity_diagnoal_list, similarity_matrix_list = [], []
    num = 0
    from tqdm import tqdm
    for review in tqdm(test_dataset):
        #if num==10:
        #    break
        similarity_diagonal, similarity_matrix = calculate_similarity_matrix(review['text'], review['label'], num)
        num += 1
        similarity_diagnoal_list += similarity_diagonal
        similarity_matrix_list += similarity_matrix
    #print(len(similarity_diagnoal_list))
    #print(len(similarity_matrix_list))
    saveToCSV_overall(similarity_diagnoal_list, 'similarity_diagonal_test_0609')
    saveToCSV_overall(similarity_matrix_list, 'similarity_matrix_test_0609')
    # save the similarity matrix
