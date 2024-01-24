import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/discourse")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/tools")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/data")

output_dir = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data_2024"

import os
import transformers
import pandas as pd
from transformers import pipeline, AutoTokenizer
from yelp_subsample import split_to_sentences
from tqdm import tqdm
from load_datasets import save_to_csv

# Source: data/amazon_analyze.py:save_sentiment_score
# Calculate sentiment score for each sentence and save to file
def save_sentiment_score(metaname = "yelp"):
    save_path = f"{output_dir}/{metaname}_sentiment.csv"
    if os.path.exists(save_path):
        print("File already exists")
        return pd.read_csv(save_path)

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

    dataset = pd.read_csv(f"{output_dir}/{metaname}.csv")
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

# Source: data/amazon_analyze.py:reformat_amazon_sentence_sentiment
def reformat_sentence_sentiment(metaname = "yelp"):
    df_sentiment = pd.read_csv(f"{output_dir}/{metaname}_sentiment.csv")
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
    save_to_csv(result_list, f"{metaname}_sent_senti_pred")
    save_to_csv(result_list_ReviewPred, f"{metaname}_senti_pred")

# Source: data/amazon_analyze.py:reformat_amazon_analyze
def reformat_lambdaScore(metaname = "yelp"):
    def calc_peak_end_avg(senti_list):
        peak, end = 0, senti_list[-1]
        for i in senti_list:
            # abs value of i
            if abs(i) > abs(peak):
                peak = i
        return (peak + end) / 2, peak, end
    df = pd.read_csv(f"{output_dir}/{metaname}_sent_senti_pred.csv")
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
    save_to_csv(result_list, f"{metaname}_sent_sentiment_tmp")

def main_process(metaname = "yelp"):
    save_sentiment_score(metaname = metaname)
    reformat_sentence_sentiment(metaname = metaname)
    reformat_lambdaScore(metaname = metaname)

if __name__ == "__main__":
    main_process(metaname = "yelp")
    main_process(metaname = "app")
