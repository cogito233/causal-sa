import numpy as np
import pandas as pd
import datasets

from tqdm import trange

from score_analyze import check_validity

def wrap_imdb():
    datapath = "imdb"
    data = datasets.load_dataset(datapath)
    data = data['train']
    text_list = data['text']
    label_list = data['label']
    # print(set(label_list)) # {0, 1}
    # exit(0)
    # Map 0 to 1, 1 to 5
    label_list = [1 if x == 0 else 5 for x in label_list]
    import random
    # set seed
    random.seed(0)
    # Get a random shuffle list of idx and then shuffle the text_list and label_list
    idx_list = list(range(len(text_list)))
    random.shuffle(idx_list)
    text_list = [text_list[i] for i in idx_list]
    label_list = [label_list[i] for i in idx_list]
    print(text_list[0:5])
    # save to a npy file
    text_list_re, label_list_re = [], []
    for i in trange(len(text_list)):
        if check_validity(text_list[i]):
            text_list_re.append(text_list[i])
            label_list_re.append(label_list[i])
        if len(text_list_re) >= 5000:
            break
    np.save("imdb.npy", text_list_re)
    np.save("imdb_label.npy", label_list_re)
"""
# can not run sentence tokenize, discard
def wrap_tripadvisor():
    datapath = "argilla/tripadvisor-hotel-reviews"
    data = datasets.load_dataset(datapath)
    data = data['train']
    text_list = data['text']
    label_list = data['prediction']
    #print(label_list[0][0]['label'])
    #print(label_list[0][0]['score'])
    score_list = [x[0]['score'] for x in label_list]
    label_list = [x[0]['label'] for x in label_list]
    print(set(label_list)) # {'3', '4', '1', '2', '5'}
    print(set(score_list)) # {1.0}
    # change label to 1-5
    label_list = [int(x) for x in label_list]
    import random
    random.seed(0)
    random.shuffle(text_list)
    print(text_list[0:5])
    # save to a npy file
    text_list_re, label_list_re = [], []
    for i in trange(len(text_list)):
        if check_validity(text_list[i]):
            text_list_re.append(text_list[i])
            label_list_re.append(label_list[i])
        if len(text_list_re) >= 5000:
            break
    np.save("tripadvisor.npy", text_list)
    np.save("tripadvisor_label.npy", label_list)
"""

def wrap_app():
    datapath = "app_reviews"
    data = datasets.load_dataset(datapath)
    data = data['train']
    text_list = data['review']
    label_list = data['star']
    print(set(label_list)) # {1, 2, 3, 4, 5}
    import random
    # set seed
    random.seed(0)
    # Get a random shuffle list of idx and then shuffle the text_list and label_list
    idx_list = list(range(len(text_list)))
    random.shuffle(idx_list)
    text_list = [text_list[i] for i in idx_list]
    label_list = [label_list[i] for i in idx_list]
    print(text_list[0:5])
    # save to a npy file
    text_list_re, label_list_re = [], []
    sentence_list_re = []
    sum = 0
    for i in trange(len(text_list)):
        if check_validity(text_list[i]):
            from yelp_subsample import split_to_sentences
            sentence_list_re.append(len(split_to_sentences(text_list[i])))
            text_list_re.append(text_list[i])
            label_list_re.append(label_list[i])
        #if len(text_list_re) >= 5000:
        #    break
    np.save("app_reviews.npy", text_list_re)
    np.save("app_reviews_label.npy", label_list_re)
    np.save("app_reviews_sentence.npy", sentence_list_re)

def wrap_beer():
    datapath = "arize-ai/beer_reviews_label_drift_neutral"
    data = datasets.load_dataset(datapath)
    data = data['training']
    text_list = data['text']
    label_list = data['label']
    print(set(label_list)) # {0, 1, 2}
    # map 0 to 1, 1 to 3, 2 to 5
    label_list = [1 if x == 0 else 3 if x == 1 else 5 for x in label_list]
    # exit(0)
    import random
    # set seed
    random.seed(0)
    # Get a random shuffle list of idx and then shuffle the text_list and label_list
    idx_list = list(range(len(text_list)))
    random.shuffle(idx_list)
    text_list = [text_list[i] for i in idx_list]
    label_list = [label_list[i] for i in idx_list]
    print(text_list[0:5])
    # save to a npy file
    text_list_re, label_list_re = [], []
    for i in trange(len(text_list)):
        if check_validity(text_list[i]):
            text_list_re.append(text_list[i])
            label_list_re.append(label_list[i])
        if len(text_list_re) >= 5000:
            break
    np.save("beer_reviews.npy", text_list_re)
    np.save("beer_reviews_label.npy", label_list_re)

def wrap_amazon():
    datapath = "amazon_reviews_multi"
    data = datasets.load_dataset(datapath)
    data = data['train']
    text_list = data['review_body']
    label_list = data['stars']
    language_list = data['language']
    print(set(label_list)) # {1, 2, 3, 4, 5}
    import random
    # set seed
    random.seed(0)
    # Get a random shuffle list of idx and then shuffle the text_list and label_list
    idx_list = list(range(len(text_list)))
    random.shuffle(idx_list)
    text_list = [text_list[i] for i in idx_list]
    label_list = [label_list[i] for i in idx_list]
    language_list = [language_list[i] for i in idx_list]
    print(text_list[0:5])
    # save to a npy file
    text_list_re, label_list_re = [], []
    sentence_list_re = []
    sum = 0
    for i in trange(len(text_list)):
        if language_list[i] == 'en'  and check_validity(text_list[i]):
            from yelp_subsample import split_to_sentences
            sentence_list_re.append(len(split_to_sentences(text_list[i])))
            text_list_re.append(text_list[i])
            label_list_re.append(label_list[i])
        #if len(text_list_re) >= 5000:
        #    break
    print(len(text_list_re))
    np.save("amazon.npy", text_list_re)
    np.save("amazon_label.npy", label_list_re)
    np.save("amazon_sentence.npy", sentence_list_re)

def inference_data(name):
    pass

if __name__ == "__main__":
    #wrap_imdb()
    # wrap_tripadvisor()
    #wrap_app()
    #wrap_beer()
    wrap_amazon()