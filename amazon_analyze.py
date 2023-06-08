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
def load_amazon():
    dataset = load_dataset('amazon_polarity')
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


if __name__ == '__main__':
    from yelp_split import load_yelp
    amazonData = load_amazon()['test']
    yelpData = load_yelp()['test']
    length_distrubution(amazonData, yelpData)
    
