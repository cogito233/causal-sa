import pandas as pd
import numpy as np
import os
import re
import datasets
# set random seed to 0
np.random.seed(0)
from nlp_tool import NLP
from build_graph import saveToCSV_overall
nlper = NLP()
def split_to_sentences(string):
    string = nlper.sent_tokenize(string)
    return string

def filter_by_sentences(yelp_test, output_name = "yelp_test"):
    result_list = []
    from tqdm import trange
    for i in trange(len(yelp_test)):
        result_dict = {}
        # if less than 5 sentences, skip
        if len(split_to_sentences(yelp_test[i]['text'])) < 5:
            continue
        result_dict['review_id'] = i
        result_dict['text'] = yelp_test[i]['text']
        result_dict['label'] = yelp_test[i]['label']
        result_list.append(result_dict)
    # Then random shuffle
    np.random.shuffle(result_list)
    print(len(result_list))
    saveToCSV_overall(result_list, output_name)
    return result_list


if __name__ == '__main__':
    from yelp_split import load_yelp
    yelp = load_yelp()['train']
    yelp = filter_by_sentences(yelp, output_name = "yelp_train")

    yelp_test = load_yelp()['test']
    yelp_test = filter_by_sentences(yelp_test, output_name = "yelp_test")


