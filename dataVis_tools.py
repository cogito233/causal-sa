

def load_sample_point(review_id):
    from yelp_split import load_yelp
    dataset = load_yelp()['test']
    print(dataset[review_id])
    return dataset[review_id]

def compare_union_set(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1.union(set2)

if __name__ == "__main__":
    # load_sample_point(1321)
    import numpy as np
    a = np.load('/home/yangk/zhiheng/develop_codeversion/causal-prompt/analyze_data/idx_list.npy')
    b = np.load('/home/yangk/zhiheng/develop_codeversion/causal-prompt/analyze_data/star_list.npy')
    print(len(a)+len(b)-len(compare_union_set(a,b)))
    print(len(a))
    print(len(b))