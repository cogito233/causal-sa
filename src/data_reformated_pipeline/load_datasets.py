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

import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/discourse")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/tools")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/data")
from yelp_subsample import split_to_sentences

output_dir = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data_2024"

import pandas as pd

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    print(df.head())
    print("len(df): ", len(df))
    path = f"{output_dir}/{filename}.csv"
    print(f"Saving to {path}")
    df.to_csv(path, index=False)

def process_review_yelp(review, index):
    if len(split_to_sentences(review['text'])) >= 5:
        label = (review['label'] - 3) / 2
        return {
            'review_id': index,
            'review_text': review['text'],
            'true_label': label,
            'stars': review['label']  # Assuming stars are equivalent to labels in Yelp dataset
        }
    return None

def process_review_amazon(review):
    if review['language'] == 'en' and len(split_to_sentences(review['review_title'] + ". " + review['review_body'])) >= 5:
        label = (review['stars'] - 3) / 2
        return {
            'review_id': review['review_id'],
            'review_text': review['review_title'] + ". " + review['review_body'],
            'true_label': label,
            'stars': review['stars']
        }
    return None

def process_review_app(review, index):
    if len(split_to_sentences(review['review'])) >= 5:
        label = (review['star'] - 3) / 2
        return {
            'review_id': index,
            'review_text': review['review'],
            'true_label': label,
            'stars': review['star']
        }
    return None

def load_and_reformat(datasetname="amazon"):
    result_list = []

    if datasetname == "amazon":
        dataset = load_dataset('amazon_reviews_multi')
        for section in ['validation', 'test']:
            for review in tqdm(dataset[section], desc=f"Processing {datasetname} {section}"):
                processed_review = process_review_amazon(review)
                if processed_review:
                    result_list.append(processed_review)

    elif datasetname == "yelp":
        dataset = load_dataset('yelp_review_full')
        for index, review in tqdm(enumerate(dataset['test']), desc="Processing Yelp test"):
            processed_review = process_review_yelp(review, index)
            if processed_review:
                result_list.append(processed_review)

    elif datasetname == "app":
        dataset = load_dataset('app_reviews')
        for index, review in tqdm(enumerate(dataset['train']), desc="Processing App train"):
            processed_review = process_review_app(review, index)
            if processed_review:
                result_list.append(processed_review)

    else:
        raise NotImplementedError

    save_to_csv(result_list, datasetname)



if __name__ == "__main__":
    # load_and_reformat(datasetname = "amazon")
    # load_and_reformat(datasetname = "yelp")
    load_and_reformat(datasetname = "app")