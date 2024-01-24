import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/discourse")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/tools")

from build_graph import calculate_similarity_matrix
from cluster_graph import buildSpanningTree_fromMatrix

# Calculate sentiment score for each sentence and save to file
import numpy as np
import pandas as pd
import transformers

from yelp_subsample import split_to_sentences

def check_validity(text):
    sentences = split_to_sentences(text)
    if len(sentences) < 5:
        return False
    return True


from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", device=0)
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

def sentiment_score(text, label): # The label is a num from 1 to 5
    from yelp_analyze import calculate_sentiment_score
    def calc_peak_end_avg(senti_list):
        peak, end = 0, senti_list[-1]
        for i in senti_list:
            # abs value of i
            if abs(i) > abs(peak):
                peak = i
        return (peak + end) / 2

    mapping = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}
    label = mapping[label]*10

    sentences = split_to_sentences(text)
    score_list = []
    for s in sentences:
        try:
            text, cuted = cut_sentence(s)
            result = sentiment_pipeline(text)
        except:
            # print(s)
            exit(0)
        # print(result)
        score_list.append(calculate_sentiment_score(result[0]['label'], result[0]['score']))

    peak_end_avg = calc_peak_end_avg(score_list)
    all_sent_avg = np.mean(score_list)

    return abs(label-all_sent_avg), abs(label-peak_end_avg), all_sent_avg

def discourse_score(text, label):
    def type1Score(review_meta):
        n = len(review_meta['sentence_list'])
        edge_list = review_meta['edge_list']
        total_MST_weight = 0
        for edge in edge_list:
            total_MST_weight += edge[0]
        total_chain_weight = 0
        for i in range(n-1):
            total_chain_weight += review_meta['similarity_matrix'][i, i+1]
        return total_chain_weight/total_MST_weight
    def type2Score(review_meta):
        n = len(review_meta['sentence_list'])
        edge_list = review_meta['edge_list']
        total_MST_weight = 0
        for edge in edge_list:
            total_MST_weight += edge[0]
        maximum_star_weight = 0
        for i in range(n):
            if i > 1:
                break
            star_weight = 0
            for j in range(n):
                if i == j:
                    continue
                star_weight += review_meta['similarity_matrix'][i, j]
            if star_weight > maximum_star_weight:
                maximum_star_weight = star_weight
        return maximum_star_weight/total_MST_weight

    sentences = split_to_sentences(text)
    #exit(0)
    similarity_diagonal_list, similarity_matrix_list = calculate_similarity_matrix(text, 0, 0)
    #exit(0)
    review_meta = {}
    length = len(sentences)
    similarity_matrix = np.zeros((length, length)) - 10
    # change the type of similarity_matrix to float
    # similarity_matrix = similarity_matrix.astype(np.float)
    # review_meta['avg_similarity'] = sub_df['similarity'].mean()
    review_meta['sentence_list'] = sentences
    for i in range(len(similarity_matrix_list)):
        idx1 = int(similarity_matrix_list[i]['idx1'])
        idx2 = int(similarity_matrix_list[i]['idx2'])
        similarity = similarity_matrix_list[i]['similarity']
        # print(idx1, idx2)
        similarity_matrix[idx1, idx2] = similarity
        similarity_matrix[idx2, idx1] = similarity

    MST_edge_list = buildSpanningTree_fromMatrix(similarity_matrix, length)
    review_meta['edge_list'] = MST_edge_list
    review_meta['similarity_matrix'] = similarity_matrix

    return type1Score(review_meta), type2Score(review_meta)

def calc_two_score(name):
    text_list = np.load(f"{name}.npy")
    label_list = np.load(f"{name}_label.npy")
    sentiment_score_list = []
    discourse_score_list = []
    from tqdm import tqdm
    #print(len(text_list))
    #print(len(label_list))
    #sum = 0
    #for text, label in tqdm(zip(text_list, label_list)):
    #    sum+=1
    #print(sum)
    #exit(0)
    #print(text_list[:10])
    #print(label_list[:10])
    #exit(0)
    for text, label in tqdm(zip(text_list, label_list)):
        # change numpy.str_ to str
        text = text.item()
        # print(text, label)
        if not check_validity(text):
            continue
        sentiment_score_list.append(sentiment_score(text, label))
        discourse_score_list.append(discourse_score(text, label))
    np.save(f"{name}_sentiment.npy", sentiment_score_list)
    np.save(f"{name}_discourse.npy", discourse_score_list)

def calc_distribution(name):
    import re
    def print_subset(subset_idx, name):
        def get_unique_words(text):
            words = re.findall(r'\b\w+\b', text.lower())
            #print(set(words))
            return set(words)
        unique_words = set(['it'])
        l1_sub, l2_sub, average_sub = [], [], []
        d1_sub, d2_sub = [], []
        sentences_num = []
        words_num = []
        for i in subset_idx:
            l1_sub.append(l1[i])
            l2_sub.append(l2[i])
            average_sub.append(average[i])
            d1_sub.append(d1[i])
            d2_sub.append(d2[i])
            unique_words = unique_words.union(get_unique_words(text_list[i]))
            sentences_num.append(sentences_list[i])
            words_num.append(len(re.findall(r'\b\w+\b', text_list[i].lower()))/sentences_list[i])
        print(f"{name} number of data: {len(subset_idx)}")
        print(f"{name} average senti: {np.mean(average_sub):.2f} +- {np.std(average_sub):.2f}")
        print(f"{name} l1: {np.mean(l1_sub):.4f} +- {np.std(l1_sub):.4f}")
        print(f"{name} l2: {np.mean(l2_sub):.4f} +- {np.std(l2_sub):.4f}")
        print(f"{name} d1: {np.mean(d1_sub):.4f} +- {np.std(d1_sub):.4f}")
        print(f"{name} d2: {np.mean(d2_sub):.4f} +- {np.std(d2_sub):.4f}")
        print(f"{name} unique words: {len(unique_words)}")
        print(f"{name} average sentences: {np.mean(sentences_num):.2f} +- {np.std(sentences_num):.2f}")
        print(f"{name} average words: {np.mean(words_num):.2f} +- {np.std(words_num):.2f}")
        print("######################################################")
    import os
    if not os.path.exists(f"{name}_sentiment.npy"):
        calc_two_score(name)
    text_list = np.load(f"{name}.npy")
    sentences_list = np.load(f"{name}_sentence.npy")
    sentiment_score_list = np.load(f"{name}_sentiment.npy")
    discourse_score_list = np.load(f"{name}_discourse.npy")
    l1, l2, average = [], [], []
    for i in sentiment_score_list:
        l1.append(i[0])
        l2.append(i[1])
        average.append(i[2])
    d1, d2 = [], []
    for i in discourse_score_list:
        d1.append(i[0])
        d2.append(i[1])
    # Then select subset by l1<l2
    all_idx = np.arange(len(l1))
    print_subset(all_idx, "all")
    sub_C1 = np.where(np.array(l1) < np.array(l2))[0]
    print_subset(sub_C1, "C1")
    sub_C2 = np.where(np.array(l1) > np.array(l2))[0]
    print_subset(sub_C2, "C2")

if __name__ == "__main__":
    #calc_distribution('imdb')
    calc_distribution('amazon')
    #calc_distribution('beer_reviews')
    exit(0)
    text = "Not worth the price and very bad cap design. Pathetic design of the caps. Very impractical to use everyday. The caps close so tight that everyday we have to wrestle with the bottle to open the cap. With a baby in one hand opening the cap is a night mare. And on top of these extra ordinary features of super secure cap, they are so expensive when compared to other brands. Stay away from these until they fix the cap issues. We have hurt ourselves many time trying to open caps as they have sharp edges on the inner and outer edges. Not worth the price."
    l1 = 1.467251
    l2 = 4.386529
    print(sentiment_score(text, 1))
    # Expected: 4.386529, 1.467251,

    print(discourse_score(text, 1))
    # Expected: 0.629000, 0.839762
