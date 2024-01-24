import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/discourse")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/tools")
sys.path.append("/home/yangk/zhiheng/develop_codeversion/causal-prompt/src/data")

output_dir = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data_2024"

import pandas as pd
import numpy as np
from tqdm import tqdm
from build_graph import calculate_similarity_matrix
from tqdm import trange
# def calculate_similarity_matrix(review, label, idx):
from load_datasets import save_to_csv

def save_to_npy(data, path):
    import numpy as np
    np.save(path, data)

def load_from_npy(path):
    import numpy as np
    return np.load(path, allow_pickle=True)

def generate_matrix(metaname):
    df = pd.read_csv(f"{output_dir}/{metaname}.csv")
    similarity_diagonal_list, similarity_matrix_list = [], []
    num = 0

    for i in trange(len(df)):
        review = df.iloc[i].to_dict()
        similarity_diagonal, similarity_matrix = calculate_similarity_matrix(review['review_text'], review['stars'], num)
        num += 1
        similarity_diagonal_list += similarity_diagonal
        similarity_matrix_list += similarity_matrix

        # Debug prints, remove or comment out for production use
        # print(similarity_diagonal)
        # print(similarity_matrix)
    # Save the results to CSV files
    save_to_csv(similarity_diagonal_list, f"{metaname}_similarity_diagonal")
    save_to_csv(similarity_matrix_list, f"{metaname}_similarity_matrix")


def load_similarity_list(metaname):
    # df: review_id, similarity, label, idx1, idx2
    # for each review, build a graph,
    # return a list of dict, {"review_id": review_id, "edge_list": edge_list, "avg_similarity": avg_similarity}
    from yelp_split import load_yelp
    from yelp_subsample import split_to_sentences
    data = pd.read_csv(f"{output_dir}/{metaname}.csv")
    path = f"{output_dir}/{metaname}_similarity_matrix.csv"
    df = pd.read_csv(path)
    print(df.head())

    import os
    if os.path.exists(f"{output_dir}/{metaname}_similarity_matrix.npy"):
        review_meta_list = load_from_npy(f"{output_dir}/{metaname}_similarity_matrix.npy")
        pass
    else:
        from tqdm import trange
        review_meta_list = []
        for i in trange(len(data)):
            review_dict = {"review_id": i, "edge_list": []}
            sentences = split_to_sentences(data.iloc[i]['review_text'])
            length = len(sentences)
            if length < 5:
                raise Exception("length < 5")
                continue
            #print(sentences)
            #print(length)
            similarity_matrix = np.zeros((length, length))-10
            # change the type of similarity_matrix to float
            similarity_matrix = similarity_matrix.astype(np.float64)
            sub_df = df[df['review_id'] == i]
            review_dict['avg_similarity'] = sub_df['similarity'].mean()
            for j in range(len(sub_df)):
                idx1 = int(sub_df.iloc[j]['idx1'])
                idx2 = int(sub_df.iloc[j]['idx2'])
                similarity = sub_df.iloc[j]['similarity']
                #print(idx1, idx2)
                similarity_matrix[idx1, idx2] = similarity
                similarity_matrix[idx2, idx1] = similarity
            from cluster_graph import buildSpanningTree_fromMatrix
            edge_list = buildSpanningTree_fromMatrix(similarity_matrix, length)
            review_dict['edge_list'] = edge_list
            review_dict['sentence_list'] = sentences
            review_dict['similarity_matrix'] = similarity_matrix
            #print(len(sub_df))
            #print(sub_df.head())
            #print(edge_list)
            #print(similarity_matrix)
            #exit(0)
            review_meta_list.append(review_dict)
        save_to_npy(review_meta_list, f"{output_dir}/{metaname}_similarity_matrix.npy")
    print("Finish loading similarity list!")
    print("The length of review_meta_list is: ", len(review_meta_list))
    return review_meta_list

def calc_scores_and_save(metaname):
    def type1Score(review_meta):
        n = len(review_meta['sentence_list'])
        edge_list = review_meta['edge_list']
        total_MST_weight = 0
        for edge in edge_list:
            total_MST_weight += edge[0]
        total_chain_weight = 0
        for i in range(n - 1):
            total_chain_weight += review_meta['similarity_matrix'][i, i + 1]
        return total_chain_weight / total_MST_weight

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

    review_meta_list = load_similarity_list(metaname)
    result_list = []

    for review_meta in review_meta_list:
        score_type1 = type1Score(review_meta)
        score_type2 = type2Score(review_meta)
        result_list.append({
            "review_id": review_meta['review_id'],
            "score_type1": score_type1,
            "score_type2": score_type2
        })

    save_to_csv(result_list, f"{metaname}_discourse")



if __name__ == "__main__":
    # generate_matrix("yelp")
    # generate_matrix("app")
    calc_scores_and_save("yelp")
    calc_scores_and_save("app")
