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
from build_graph import saveToCSV_overall

# load yelp dataset from huggingface
def load_amazon():
    dataset = load_dataset('amazon_reviews_multi')
    print(dataset)
    return dataset

def length_distrubution(amazon_dataset, yelp_dataset):
    amazon_list = []
    yelp_list = []
    for i in tqdm(range(len(amazon_dataset))):
        amazon_list.append(len(amazon_dataset[i]['content']))
    for i in tqdm(range(len(yelp_dataset))):
        yelp_list.append(len(yelp_dataset[i]['text']))
    amazon_list = np.array(amazon_list)
    yelp_list = np.array(yelp_list)
    print('amazon mean: ', np.mean(amazon_list))
    print('amazon std: ', np.std(amazon_list))
    print('yelp mean: ', np.mean(yelp_list))
    print('yelp std: ', np.std(yelp_list))

from yelp_subsample import split_to_sentences

def reformat_amazon():
    dataset = load_amazon()
    result_list = []
    idx = 0
    #print(dataset['label'][:5])
    for review in dataset['validation']:
        if review['language'] != 'en':
            continue
        if len(split_to_sentences(review['review_title']+". "+review['review_body'])) < 5:
            continue
        if review['stars'] == 3:
            label = 0
        elif review['stars'] > 3:
            label = 1
        else:
            label = -1
        result = {
            'review_id': review['review_id'],
            'review_text': review['review_title']+". "+review['review_body'],
            'true_label': label,
            'stars': review['stars']
        }
        idx += 1
        result_list.append(result)
    #print(result_list[:5])
    for review in dataset['test']:
        if review['language'] != 'en':
            continue
        if len(split_to_sentences(review['review_title']+". "+review['review_body'])) < 5:
            continue
        if review['stars'] == 3:
            label = 0
        elif review['stars'] > 3:
            label = 1
        else:
            label = -1
        result = {
            'review_id': review['review_id'],
            'review_text': review['review_title']+". "+review['review_body'],
            'true_label': label,
            'stars': review['stars']
        }
        idx += 1
        result_list.append(result)

    saveToCSV_overall(result_list, 'amazon_doc_senti_true')

# Calculate sentiment score for each sentence and save to file
def save_sentiment_score(save_path):
    from transformers import pipeline
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', max_length=512)

    def cut_sentence(sentence):
        # use the tokenizer to cut the sentence to maximum length
        # if the sentence is too long, only keep the last 512 tokens
        # return the sentence and whether it is cut
        encoded_input = tokenizer.encode(sentence)
        #print(len(encoded_input))
        if len(encoded_input) > 500:
            #print(tokenizer.decode(encoded_input[-510:]))
            return tokenizer.decode(encoded_input[-500:]), True
        else:
            return sentence, False

    sentiment_pipeline = pipeline("sentiment-analysis", device = 0)
                                  #)# model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer = 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    df_dict = {'sentence':[], 'score':[], 'label':[], 'review_id':[], 'review_stars':[], "cuted":[]}

    dataset = pd.read_csv("/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data/amazon_doc_senti_true.csv")
    #exit(0)
    for i in tqdm(range(len(dataset))):
        review = dataset.iloc[i].to_dict()
        #sentences = split2sentences(review['text'])
        from yelp_subsample import split_to_sentences
        sentences = split_to_sentences(review['review_text'])
        try:
            text, cuted = cut_sentence(review['review_text'])
            result = sentiment_pipeline(text)
        except:
            print(s)
            exit(0)
        df_dict['sentence'].append(review['review_text'])
        df_dict['score'].append(result[0]['score'])
        df_dict['label'].append(result[0]['label'])
        df_dict['review_id'].append(i)
        df_dict['review_stars'].append(review['stars'] + 1)
        df_dict['cuted'].append(cuted)
        for s in sentences:
            try:
                text, cuted = cut_sentence(s)
                result = sentiment_pipeline(text)
            except:
                print(s)
                exit(0)
            df_dict['sentence'].append(s)
            df_dict['score'].append(result[0]['score'])
            df_dict['label'].append(result[0]['label'])
            df_dict['review_id'].append(i)
            df_dict['review_stars'].append(review['stars']+1)
            df_dict['cuted'].append(cuted)
    df = pd.DataFrame(df_dict)
    df.to_csv(save_path, index=False)
    return df

def reformat_amazon_sentence_sentiment():
    df_sentiment = pd.read_csv('amazon_sentiment_score.csv')
    review_id = -1
    sentence_id = 0
    result_list = []
    result_list_ReviewPred = []
    from tqdm import trange
    for i in trange(len(df_sentiment)):
        item = df_sentiment.iloc[i]
        from yelp_analyze import calculate_sentiment_score
        score = calculate_sentiment_score(item['label'], item['score'])
        if item['review_id'] != review_id:
            review_id = item['review_id']
            sentence_id = 0
            result = {
                'review_id': item['review_id'],
                #'sentence_id': sentence_id,
                'review_text': item['sentence'],
                'review_sentiment': score,
                'cut_flag': item['cuted']
            }
            result_list_ReviewPred.append(result)
        result = {
            'review_id': item['review_id'],
            'sentence_id': sentence_id,
            'sentence_text': item['sentence'],
            'sentence_sentiment': score,
            'cut_flag': item['cuted']
        }
        result_list.append(result)
        sentence_id += 1
    saveToCSV_overall(result_list, 'amazon_sent_senti_pred')
    saveToCSV_overall(result_list_ReviewPred, 'amazon_senti_pred')

# TODO: Yelp Analyze(yelp_sent_sentiment_tmp.csv): (review_id, peak_end_avg, all_sent_avg)
def reformat_amazon_analyze():
    def calc_peak_end_avg(senti_list):
        peak, end = 0, senti_list[-1]
        for i in senti_list:
            # abs value of i
            if abs(i) > abs(peak):
                peak = i
        return (peak + end) / 2, peak, end
    df = pd.read_csv('reformated_data/amazon_sent_senti_pred.csv')
    print(df.head())
    import math
    review_id = 0
    result_list = []
    senti_list = []
    for i in range(len(df)):
        item = df.iloc[i]
        if item['review_id'] != review_id:
            if len(senti_list) > 0:
                peak_end_avg, peak, end = calc_peak_end_avg(senti_list)
                all_sent_avg = sum(senti_list) / len(senti_list)
                result_list.append({
                    'review_id': review_id,
                    'peak_end_avg': peak_end_avg,
                    'all_sent_avg': all_sent_avg,
                    'peak': peak,
                    'begin': senti_list[0],
                    'end': end
                })
            review_id = item['review_id']
            senti_list = []
        senti_list.append(item['sentence_sentiment'])
    if len(senti_list) > 0:
        peak_end_avg, peak, end = calc_peak_end_avg(senti_list)
        all_sent_avg = sum(senti_list) / len(senti_list)
        result_list.append({
            'review_id': review_id,
            'peak_end_avg': peak_end_avg,
            'all_sent_avg': all_sent_avg,
            'peak': peak,
            'begin': senti_list[0],
            'end': end
        })
    saveToCSV_overall(result_list, 'amazon_sent_sentiment_tmp')


if __name__ == '__main__':
    #from yelp_split import load_yelp
    #amazonData = load_amazon()['test']
    #yelpData = load_yelp()['test']
    #length_distrubution(amazonData, yelpData)
    #reformat_amazon()

    #df = pd.read_csv("/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data/amazon_doc_senti_true.csv")
    #print(df.head())
    #print(len(df))
    #save_sentiment_score('amazon_sentiment_score.csv')
    reformat_amazon_sentence_sentiment()
    reformat_amazon_analyze()
