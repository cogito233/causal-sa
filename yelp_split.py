import os
import re
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import datasets
from datasets import Dataset, load_dataset

# load yelp dataset from huggingface
def load_yelp():
    dataset = load_dataset('yelp_review_full')
    print(dataset)
    return dataset

def split2sentences(str):
    # split reviews into sentences
    sentences = str.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    sentences = [s for s in sentences if len(s.split(' ')) > 3]
    return sentences


from transformers import GPT2Tokenizer
# 加载GPT-2的tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

def calculate_sentence_length(sentence):
    # Return the length of the sentence
    encoded_input = tokenizer_gpt.encode(sentence)
    return len(encoded_input)

def custom_sentence_tokenize(text):
    # 根据句号分割句子
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    # 进一步根据逗号分割句子
    sentences_raw = []
    for sentence in sentences:
        comma_split = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=;)\s', sentence)
        sentences_raw.extend(comma_split)

    sentences = []
    curr_sentence = ''
    curr_length = 0
    # if the length of the sentence is less than 10, concat the sentence
    for i in range(len(sentences_raw)):
        curr_sentence = curr_sentence + ' ' + sentences_raw[i]
        curr_length = calculate_sentence_length(curr_sentence)
        if curr_length > 10:
            sentences.append(curr_sentence[1:])
            curr_sentence = ''
            curr_length = 0
    if curr_length > 0:
        sentences.append(curr_sentence[1:])
    return sentences


# Calculate sentiment score for each sentence and save to file
def save_sentiment_score(save_path):
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", device = 0)# model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer = 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    df_dict = {'sentence':[], 'score':[], 'label':[], 'review_id':[], 'review_stars':[]}
    dataset = load_yelp()['test']
    for i in tqdm(range(len(dataset))):
        review = dataset[i]
        #sentences = split2sentences(review['text'])
        sentences = custom_sentence_tokenize(review['text'])
        for s in sentences:
            try:
                result = sentiment_pipeline(s)
            except:
                continue
            df_dict['sentence'].append(s)
            df_dict['score'].append(result[0]['score'])
            df_dict['label'].append(result[0]['label'])
            df_dict['review_id'].append(i)
            df_dict['review_stars'].append(review['label']+1)
    df = pd.DataFrame(df_dict)
    df.to_csv(save_path, index=False)
    return df

# Analyze the df maybe on the colab?

# Visualize some samples of yelp reviews
def print_datapoint(dataset, idx):
    print(f"The {idx}th datapoint:")
    print(f"Review: {dataset[idx]['text']}")
    print("")
    print(f"Label: {dataset[idx]['label']}")
    print("")
    for s in split2sentences(dataset[idx]['text']):
        print(f"Sentence: {s}")
        print("")

    return
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--save_path', type=str, default='./yelp_sentiment_score.csv')
    # args = parser.parse_args()
    #save_sentiment_score("./yelp_sentiment_score_newSentence.csv")

    dataset = load_yelp()['test']
    #print_datapoint(dataset, 13813)