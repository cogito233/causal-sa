import pandas as pd
import numpy as np
import os
import re
from cluster_graph import save_to_npy, load_from_npy
from build_graph import saveToCSV_overall
# Done: Original Yelp(yelp_doc_senti_true.csv): (review_id, review_text, true_label(-1, 0, 1))
def reformat_Yelp():
    from yelp_split import load_yelp
    dataset = load_yelp()['test']
    result_list = []
    idx = 0
    #print(dataset['label'][:5])
    for review in dataset:
        if review['label'] == 2:
            label = 0
        elif review['label'] > 2:
            label = 1
        else:
            label = -1
        result = {
            'review_id': idx,
            'review_text': review['text'],
            'true_label': label
        }
        idx += 1
        result_list.append(result)
    #print(result_list[:5])
    saveToCSV_overall(result_list, 'yelp_doc_senti_true')
# Done: Yelp Sentence(yelp_sent_senti_pred.csv): (review_id, sentence_id, sentence_text, sentence_sentiment)
def reformat_Yelp_sentence_sentiment():
    df_sentiment = pd.read_csv('yelp_sentiment_score_newSentence.csv')
    review_id = 0
    sentence_id = 0
    result_list = []
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
            'sentence_id': sentence_id,
            'sentence_text': item['sentence'],
            'sentence_sentiment': score
        }
        sentence_id += 1
        result_list.append(result)
    saveToCSV_overall(result_list, 'yelp_sent_senti_pred')

# TODO: Yelp Analyze(yelp_sent_sentiment_tmp.csv): (review_id, peak_end_avg, all_sent_avg)
def reformat_Yelp_analyze():
    def calc_peak_end_avg(senti_list):
        peak, end = 0, senti_list[-1]
        for i in senti_list:
            # abs value of i
            if abs(i) > abs(peak):
                peak = i
        return (peak + end) / 2
    df = pd.read_csv('reformated_data/yelp_sent_senti_pred.csv')
    import math
    review_id = 0
    result_list = []
    senti_list = []
    for i in range(len(df)):
        item = df.iloc[i]
        if item['review_id'] != review_id:
            if len(senti_list) > 0:
                peak_end_avg = calc_peak_end_avg(senti_list)
                all_sent_avg = sum(senti_list) / len(senti_list)
                result_list.append({
                    'review_id': review_id,
                    'peak_end_avg': peak_end_avg,
                    'all_sent_avg': all_sent_avg
                })
            review_id = item['review_id']
            senti_list = []
        senti_list.append(item['sentence_sentiment'])
    if len(senti_list) > 0:
        peak_end_avg = calc_peak_end_avg(senti_list)
        all_sent_avg = sum(senti_list) / len(senti_list)
        result_list.append({
            'review_id': review_id,
            'peak_end_avg': peak_end_avg,
            'all_sent_avg': all_sent_avg
        })
    saveToCSV_overall(result_list, 'yelp_sent_sentiment_tmp')

# TODO: yelp_doc_senti_pred.csv(review_id, pred_score, pred_label)
# TODO: Sentiment model is not work?
def reformat_Yelp_doc_sentiment():
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", device=0)
    def calc_dict(text):
        from yelp_analyze import calculate_sentiment_score
        try:
            result = sentiment_pipeline(text)
        except Exception as e:
            print(text)
        print("Done")
        print(result)
        exit(0)
        return {
            "pred_score": calculate_sentiment_score(result[0]['label'], result[0]['score']),
            "pred_label": result[0]['label']
        }
    from yelp_split import load_yelp
    dataset = load_yelp()['test']
    result_list = []
    idx = 0
    from tqdm import tqdm
    for review in tqdm(dataset):
        result = calc_dict(review['text'])
        result['review_id'] = idx
        idx += 1
        result_list.append(result)
    saveToCSV_overall(result_list, 'yelp_doc_senti_pred')

if __name__ == '__main__':
    #df = pd.read_csv('yelp_sentiment_score_newSentence.csv')
    #print(df.head())
    #reformat_Yelp_sentence_sentiment()
    #df = pd.read_csv('reformated_data/yelp_sent_senti_pred.csv')
    #print(df.head())
    #print(df.tail())
    #reformat_Yelp_analyze()
    df = pd.read_csv('reformated_data/yelp_sent_sentiment_tmp.csv')
    print(df.head())
    print(df.tail())