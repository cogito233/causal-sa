import pandas as pd
import numpy as np

def save_to_npy(obj, path):
    obj_np = np.array(obj)
    np.save(path, obj_np)

def load_from_npy(path):
    obj_np = np.load(path, allow_pickle=True)
    obj = obj_np.tolist()
    return obj

def buildSpanningTree_fromMatrix(similarity_matrix, num):
    # Use Prim's algorithm to build a maximum spanning tree
    # similarity_matrix: a matrix of similarity
    # num: the number of nodes
    # return: a list of (similarity, input_id, output_id)

    # initialize
    edge_list = []
    visited = [False for i in range(num)]
    visited[0] = True
    curr_sim = [-1 for i in range(num)]
    curr_edge_in = [-1 for i in range(num)]
    for i in range(num):
        if similarity_matrix[0, i] != -10:
            curr_sim[i] = similarity_matrix[0, i]
            curr_edge_in[i] = 0

    for i in range(num-1):# add num-1 edges to the tree
        # find the max similarity
        max_sim, max_idx = -1, -1
        for j in range(num):
            if not visited[j] and curr_sim[j] > max_sim:
                max_sim = curr_sim[j]
                max_idx = j
        # add the edge
        edge_list.append((max_sim, curr_edge_in[max_idx], max_idx))
        visited[max_idx] = True
        # update curr_sim and curr_edge_in
        for j in range(num):
            if not visited[j] and similarity_matrix[max_idx, j] != -10 and similarity_matrix[max_idx, j] > curr_sim[j]:
                curr_sim[j] = similarity_matrix[max_idx, j]
                curr_edge_in[j] = max_idx
    # Sort the edge_list by similarity
    edge_list.sort(key=lambda x: x[0], reverse=True)

    #print(edge_list)
    #exit(0)
    return edge_list # list of (similarity, input_id, output_id), make sure (input, output) be unique

def is_stream(edge_list, n):
    # 所有的i->i+1至少要满足(n+1)/2
    edge_dict = {}
    for i in range(len(edge_list)):
        edge_dict[(edge_list[i][1], edge_list[i][2])] = edge_list[i][0]
    count = 0
    for i in range(n-1):
        if (i, i+1) in edge_dict:
            count += 1
    return count >= (n+1)/2

def is_star(edge_list, n):
    # 树的重心距离1以内包括一半以上的节点；alternative：存在一个点和一半以上的点距离1以内/出度大于n/2
    count_dict = {}
    for i in range(n):
        count_dict[i] = 0
    for i in range(len(edge_list)):
        count_dict[edge_list[i][1]] += 1
        count_dict[edge_list[i][2]] += 1
    for i in range(n):
        if count_dict[i] > (n+1)/2:
            #print(edge_list)
            #exit(0)
            return True, i/(n-1) # Return the center of the star
    return False, -1

def load_similarity_list():
    # df: review_id, similarity, label, idx1, idx2
    # for each review, build a graph,
    # return a list of dict, {"review_id": review_id, "edge_list": edge_list, "avg_similarity": avg_similarity}
    from yelp_split import load_yelp
    from yelp_subsample import split_to_sentences
    data = pd.read_csv("/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data/amazon_doc_senti_true.csv")
    path = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/reformated_data/amazon_similarity_matrix_test.csv"
    df = pd.read_csv(path)
    print(df.head())
    #exit(0)

    import os
    if os.path.exists("analyze_data/review_meta_list.npyy"):
        review_meta_list = load_from_npy("analyze_data/review_meta_list.npy")
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
            similarity_matrix = similarity_matrix.astype(np.float)
            sub_df = df[df['review_id'] == i]
            review_dict['avg_similarity'] = sub_df['similarity'].mean()
            for j in range(len(sub_df)):
                idx1 = int(sub_df.iloc[j]['idx1'])
                idx2 = int(sub_df.iloc[j]['idx2'])
                similarity = sub_df.iloc[j]['similarity']
                #print(idx1, idx2)
                similarity_matrix[idx1, idx2] = similarity
                similarity_matrix[idx2, idx1] = similarity
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
        save_to_npy(review_meta_list, "analyze_data/review_meta_list.npy")
    print("Finish loading similarity list!")
    print("The length of review_meta_list is: ", len(review_meta_list))
    return review_meta_list
    """
    count_stream = 0
    count_star = 0
    count_both = 0
    star_list = []
    center_list = []
    from tqdm import trange
    for i in trange(len(review_meta_list)):
        review_meta_list[i]['is_stream'] = is_stream(review_meta_list[i]['edge_list'], len(review_meta_list[i]['sentence_list']))
        review_meta_list[i]['is_star'], center = is_star(review_meta_list[i]['edge_list'], len(review_meta_list[i]['sentence_list']))
        if review_meta_list[i]['is_stream']:
            count_stream += 1
        if review_meta_list[i]['is_star']:
            count_star += 1
            star_list.append(review_meta_list[i]['review_id'])
            center_list.append(center)
        if review_meta_list[i]['is_stream'] and review_meta_list[i]['is_star']:
            count_both += 1
    print(star_list[:10])
    print(center_list[:10])
    save_to_npy(star_list, "analyze_data/star_list.npy")
    save_to_npy(center_list, "analyze_data/center_list.npy")

    print("count_stream: ", count_stream)
    print("count_star: ", count_star)
    print("count_both: ", count_both)
    """

def save_sentence_list():
    from yelp_split import load_yelp
    from yelp_subsample import split_to_sentences
    data = load_yelp()['test']

    result_list = []
    from tqdm import trange
    for i in trange(len(data)):
        sentences = split_to_sentences(data[i]['text'])
        idx = 0
        for sentence in sentences:
            result_list.append({
                "review_id": i,
                "sentence_id": idx,
                "sentence": sentence
            })
            idx += 1
            pass
    from build_graph import saveToCSV_overall
    saveToCSV_overall(result_list, "sentence_list_test")

def calc_type1Score(): # -> review_discourse_type1.csv; return a list of dict
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
    review_meta_list = load_similarity_list()
    output_path = "analyze_data/review_discourse_type1.npy"
    import os
    if os.path.exists(output_path) and False:
        result_list = load_from_npy(output_path)
    else:
        result_list = []
        for review_meta in review_meta_list:
            result_list.append({
                "review_id": review_meta['review_id'],
                "score_type1": type1Score(review_meta)
            })
        save_to_npy(result_list, output_path)
    count = 0
    for result in result_list:
        if result['score_type1'] > 0.9:
            count += 1
    print("Type1 count: ", count)
    return result_list
def calc_type2Score(): # -> review_discourse_type2.csv return a list of dict
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

    review_meta_list = load_similarity_list()
    output_path = "analyze_data/review_discourse_type2.npy"
    import os
    if os.path.exists(output_path) and False:
        result_list = load_from_npy(output_path)
    else:
        result_list = []
        for review_meta in review_meta_list:
            result_list.append({
                "review_id": review_meta['review_id'],
                "score_type2": type2Score(review_meta)
            })
        save_to_npy(result_list, output_path)
    count = 0
    for result in result_list:
        if result['score_type2'] > 0.6:
            count += 1
    print("Type2 count: ", count)
    return result_list

def merge_type1_type2(type1_list, type2_list): # -> review_discourse.csv
    from build_graph import saveToCSV_overall
    for i in range(len(type1_list)):
        type1_list[i]['score_type2'] = type2_list[i]['score_type2']
    print(len(type1_list))
    saveToCSV_overall(type1_list, "amazon_discourse")

# export MST Graph to visualize on colab
def export_graph():
    review_meta_list = load_similarity_list()
    print(review_meta_list[0])
    result_list = [] # {"review_id": int, "sentence_id1": int, "sentence_id2": int, "weight": float}
    for review_meta in review_meta_list:
        for edge in review_meta['edge_list']:
            result_list.append({
                "review_id": review_meta['review_id'],
                "sentence_id1": edge[1],
                "sentence_id2": edge[2],
                "weight": edge[0]
            })
    from build_graph import saveToCSV_overall
    saveToCSV_overall(result_list, "yelp_MST_graph_0609")


if __name__ == '__main__':
    type1_score = calc_type1Score()
    type2_score = calc_type2Score()
    print("len(type1_score): ", len(type1_score))
    merge_type1_type2(type1_score, type2_score)
    #export_graph()

# Step 1: Test Save and Load npy
# Step 2: Test build_graph and visualize graph?
# Step 3: Test is_connect, is_stream, is_star
