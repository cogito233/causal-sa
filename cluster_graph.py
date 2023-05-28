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

def load_similarity_list(df):
    # df: review_id, similarity, label, idx1, idx2
    # for each review, build a graph,
    # return a list of dict, {"review_id": review_id, "edge_list": edge_list, "avg_similarity": avg_similarity}
    from yelp_split import load_yelp, custom_sentence_tokenize
    data = load_yelp()['test']
    import os
    if os.path.exists("analyze_data/review_meta_list.npy"):
        review_meta_list = load_from_npy("analyze_data/review_meta_list.npy")
        pass
    else:
        from tqdm import trange
        review_meta_list = []
        for i in trange(len(data)):
            review_dict = {"review_id": i, "edge_list": []}
            sentences = custom_sentence_tokenize(data[i]['text'])
            length = len(sentences)
            if length<=5:
                continue
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
            review_meta_list.append(review_dict)
        save_to_npy(review_meta_list, "analyze_data/review_meta_list.npy")
    print("Finish loading similarity list!")
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


def save_sentence_list():
    from yelp_split import load_yelp, custom_sentence_tokenize
    data = load_yelp()['test']

    result_list = []
    from tqdm import trange
    for i in trange(len(data)):
        sentences = custom_sentence_tokenize(data[i]['text'])
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


if __name__ == '__main__':
    path = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/analyze_data/similarity_matrix_test.csv"
    df = pd.read_csv(path)
    #print(df[0:20])
    load_similarity_list(df)
    #save_sentence_list()
    #path = "/home/yangk/zhiheng/develop_codeversion/causal-prompt/analyze_data/sentence_list_test.csv"
    #df_sentence = pd.read_csv(path)
    #print(df_sentence[:10])
    pass
# Step 1: Test Save and Load npy
# Step 2: Test build_graph and visualize graph?
# Step 3: Test is_connect, is_stream, is_star
