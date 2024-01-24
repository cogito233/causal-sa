import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/discourse")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/tools")

from build_graph import calculate_similarity_matrix
from cluster_graph import buildSpanningTree_fromMatrix

# Calculate sentiment score for each sentence and save to file
import numpy as np
import pandas as pd
import transformers

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

from yelp_subsample import split_to_sentences

def sentiment_score(sentences, label): # The label is a num from 1 to 5
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

    # sentences = split_to_sentences(text)
    score_list = []
    for s in sentences:
        text, cuted = cut_sentence(s)
        result = sentiment_pipeline(text)
        # print(result)
        score_list.append(calculate_sentiment_score(result[0]['label'], result[0]['score']))

    #peak_end_avg = calc_peak_end_avg(score_list)
    #all_sent_avg = np.mean(score_list)

    print(score_list)

def discourse_matrix(sentences, label):
    text = ' '.join(sentences)
    similarity_diagonal_list, similarity_matrix_list = calculate_similarity_matrix(text, 0, 0)
    # exit(0)
    review_meta = {}
    length = len(sentences)
    similarity_matrix = np.zeros((length, length))
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
    print(similarity_matrix.tolist())
    return similarity_matrix.tolist()


if __name__ == "__main__":
    c0_sentences = ["A wonderful gem in the neighborhood.",
"The lady that greeted us was extremely sweet, welcoming and helpful.",
"The place is excellent with children.",
"Medium sized collection of choosable-ready to paint items.",
"Even though the place is a bit pricey, you're free to stay as long as you work on your pieces, and they do the clean up so you don't have to."]
    c0_label = 5
    print(c0_sentences)
    sentiment_score(c0_sentences, c0_label)
    discourse_matrix(c0_sentences, c0_label)

    # c1_sentences = ["This was a great spot to take a break from it all and just people watch.",
    #     "We sat at the bar facing the casino and we were entertained the whole time.",
    #     "The mini grilled cheese (appetizer) was fantastic.",
    #     "It came with a tomato based dipping sauce that was the perfect compliment to the bite sized wedges.",
    #     "Tip - ask for two dipping sauces because one just won't do.",
    # ]
    # c1_label = 4
    # print(c1_sentences)
    # sentiment_score(c1_sentences, c1_label)
    # discourse_matrix(c1_sentences, c1_label)
    #
    # c2_sentences = ["I read the reviews and should have steered away... but it looked interesting.",
    #     "Salad was wilted, menus are on the wall, with no explanation so you are ordering blind, service was NOT with a smile from the bartender to the waitress, to the server who helped the waitress, and the waitress never checked back to see how everything is.",
    #     "Terribly overpriced for what you get, and as an Italian, this does not even pass for a facsimile thereof!",
    #     "Stay away for sure.",
    #     "I only gave them one star, as I had to fill something in, they should get no stars!"
    # ]
    # c2_label = 1
    # print(c2_sentences)
    # sentiment_score(c2_sentences, c2_label)
    # discourse_matrix(c2_sentences, c2_label)



"""
C1:
"This was a great spot to take a break from it all and just people watch."
"We sat at the bar facing the casino and we were entertained the whole time."
"The mini grilled cheese (appetizer) was fantastic."
"It came with a tomato based dipping sauce that was the perfect compliment to the bite sized wedges."
"Tip - ask for two dipping sauces because one just won't do."


C2: 
I read the reviews and should have steered away... but it looked interesting.
Salad was wilted, menus are on the wall, with no explanation so you are ordering blind, service was NOT with a smile from the bartender to the waitress, to the server who helped the waitress, and the waitress never checked back to see how everything is.
Terribly overpriced for what you get, and as an Italian, this does not even pass for a facsimile thereof!
Stay away for sure.
I only gave them one star, as I had to fill something in, they should get no stars!


C0:
A wonderful gem in the neighborhood.
The lady that greeted us was extremely sweet, welcoming and helpful.
The place is excellent with children.
Medium sized collection of choosable-ready to paint items.
Even though the place is a bit pricey, you're free to stay as long as you work on your pieces, and they do the clean up so you don't have to.
"""