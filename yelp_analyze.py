import pandas as pd
import numpy as np

def analyze_similarity(path):
    # Return the distribution of average diagnal similarity - average similarity
    # Select the top 10% of the distribution (~5000) and return their idx
    df = pd.read_csv(path)
    similarity = df["similarity"]
    # print the mean and std of the similarity
    print("mean: ", np.mean(similarity))
    print("std: ", np.std(similarity))

def subsampling_similarity(path_diagonal, path_overall):
    # Return is a list of idx, with is seems to be type 1
    #filename = "idx_list_another_0.05"
    filename = "idx_list"
    def check_type1(similarity_diagonal, similarity_overall):
        # input is two list of similarity
        # output is a boolean, whether it is type 1
        if len(similarity_diagonal) == 0 or len(similarity_overall) == 0:
            return False
        ave_diagonal = np.mean(similarity_diagonal)
        ave_overall = np.mean(similarity_overall)
        if ave_diagonal+0.05 < ave_overall:
            return True
        else:
            return False
    # load idx list from idx_list.npy
    import os
    if os.path.exists(f"analyze_data/{filename}.npy"):
        return np.load(f"analyze_data/{filename}.npy").tolist(), np.load("analyze_data/all_valid_idx.npy").tolist()
    #if os.path.exists("analyze_data/idx_list.npy"):
    #    return np.load("analyze_data/idx_list.npy").tolist(), np.load("analyze_data/all_valid_idx.npy").tolist()

    df_diagonal = pd.read_csv(path_diagonal)
    df_overall = pd.read_csv(path_overall)
    idx_list = []
    all_valid_idx = []
    idx_overall = 0
    list_overall = []
    list_diagonal = []
    review_id = -1
    from tqdm import trange
    for idx_diagonal in trange(len(df_diagonal)):
        item = df_diagonal.iloc[idx_diagonal]
        if item["review_id"] != review_id:
            # get all review_id in df_overall
            while df_overall.iloc[idx_overall]["review_id"] == review_id:
                list_overall.append(df_overall.iloc[idx_overall]["similarity"])
                idx_overall += 1
            # check and save the idx
            if check_type1(list_diagonal, list_overall):
                idx_list.append(review_id)
            all_valid_idx.append(review_id)
            # new review
            review_id = item["review_id"]
            list_overall = []
            list_diagonal = []
        list_diagonal.append(item["similarity"])
        if idx_diagonal == 10000:
            break
    # the last review
    while df_overall.iloc[idx_overall]["review_id"] == review_id:
        list_overall.append(df_overall.iloc[idx_overall]["similarity"])
        idx_overall += 1
        if idx_overall >= len(df_overall):
            break
    if check_type1(list_diagonal, list_overall):
        idx_list.append(review_id)
    all_valid_idx.append(review_id)

    # save idx_list
    np.save(f"analyze_data/{filename}.npy", np.array(idx_list))
    #np.save("analyze_data/all_valid_idx.npy", np.array(all_valid_idx))
    return idx_list, all_valid_idx

def calculate_sentiment_score(label, score):
    def reverse_sigmoid(x):
        return -np.log((1 / x) - 1)
    if label == "NEGATIVE":
        score = 1 - score
    return reverse_sigmoid(score)

def calculate_sentiment_distribution(path, output_name):
    def calculate_sentiment(sentiment_list):
        # input is a list of sentiment
        # output is a tuple of (first_half, last_half)
        first_half = np.mean(sentiment_list[:len(sentiment_list)//2])
        last_half = np.mean(sentiment_list[len(sentiment_list)//2:])
        return first_half, last_half
    # sentence,score,label,review_id,review_stars
    # input is a path with csv file, output another csv file with the follow column:
    # review_id, sentiment_first_half, sentiment_last_half, review_stars
    df = pd.read_csv(path)
    review_id = 0
    review_star = df.iloc[0]["review_stars"]
    result_list = []
    sentiment_list = []
    from tqdm import trange
    for idx in trange(len(df)):
        item = df.iloc[idx]
        if review_id != item["review_id"]:
            # save new review
            first_half, last_half = calculate_sentiment(sentiment_list)
            review_dict = {
                "review_id": review_id,
                "sentiment_first_half": first_half,
                "sentiment_last_half": last_half,
                "sentiment_overall": np.mean(sentiment_list),
                "review_stars": review_star
            }
            result_list.append(review_dict)
            # new review
            review_id = item["review_id"]
            review_star = item["review_stars"]
            sentiment_list = []
        sentiment_list.append(calculate_sentiment_score(item["label"], item["score"]))
    # the last review
    first_half, last_half = calculate_sentiment(sentiment_list)
    review_dict = {
        "review_id": review_id,
        "sentiment_first_half": first_half,
        "sentiment_last_half": last_half,
        "sentiment_overall": np.mean(sentiment_list),
        "review_stars": review_star
    }
    result_list.append(review_dict)

    from build_graph import saveToCSV_overall
    saveToCSV_overall(result_list, output_name)
    #return sentiment_first_half, sentiment_last_half

def print_distribution_crossID(path_sentiment_distribution, idx_list = None, center_list = None):
    # if idx_list is None, print the distribution of all idx
    df = pd.read_csv(path_sentiment_distribution)
    # Translate df to a dict with review_id->(first_half, last_half)
    dict_review_id = {}
    for idx in range(len(df)):
        item = df.iloc[idx]
        review_id = item["review_id"]
        if review_id not in dict_review_id:
            dict_review_id[review_id] = {
                "first_half": item["sentiment_first_half"],
                "last_half": item["sentiment_last_half"],
                "overall": item["sentiment_overall"],
                "review_stars": item["review_stars"]
            }
        else:
            print("Error: duplicate review_id")
    # print the distribution
    if idx_list is None:
        idx_list = list(dict_review_id.keys())
    result_dict = {
        "first_half": [],
        "last_half": [],
        "overall": [],
        "sentiment_label": []
    }
    for i in range(len(idx_list)):
        idx = idx_list[i]
        center = center_list[i]
        if idx not in dict_review_id:
            continue
        item = dict_review_id[idx]
        print(item)
        exit(0)
        if item['review_stars'] == 2:
            continue
        result_dict["first_half"].append(item["first_half"])
        result_dict["last_half"].append(item["last_half"])
        result_dict["overall"].append(item["overall"])
        result_dict["sentiment_label"].append(1 if item["review_stars"] > 2 else -1)

    # Calculate the correlation of first_half and sentiment_label
    from scipy.stats import pearsonr
    print(len(result_dict["first_half"]))
    print("first_half and sentiment_label: ", pearsonr(result_dict["first_half"], result_dict["sentiment_label"]))
    print("last_half and sentiment_label: ", pearsonr(result_dict["last_half"], result_dict["sentiment_label"]))
    print("overall and sentiment_label: ", pearsonr(result_dict["overall"], result_dict["sentiment_label"]))



if __name__ == "__main__":
    """
    path_diagonal = "analyze_data/similarity_diagonal_test.csv"
    path_overall = "analyze_data/similarity_matrix_test.csv"
    # analyze_similarity(path_diagonal)
    # analyze_similarity(path_overall)
    idx, all_idx = subsampling_similarity(path_diagonal, path_overall)
    print(len(idx))
    print(len(all_idx))
    #print(idx)
    #print(len(subsampling_similarity(path_diagonal, path_overall)))
    #calculate_sentiment_distribution("yelp_sentiment_score_newSentence.csv", "sentiment_distribution")
    path_sentiment = "analyze_data/sentiment_distribution.csv"

    #df = pd.read_csv(path_sentiment)
    #print(df[:10])
    #print(len(df))
    print_distribution_crossID(path_sentiment, idx)
    print_distribution_crossID(path_sentiment, all_idx)
    """
    path_sentiment = "analyze_data/sentiment_distribution.csv"

    idx = np.load("analyze_data/star_list.npy").tolist()
    center_idx = np.load("analyze_data/center_list.npy").tolist()
    print(len(idx))
    print_distribution_crossID(path_sentiment, idx, center_idx)




