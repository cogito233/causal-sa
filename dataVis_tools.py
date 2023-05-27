

def load_sample_point(review_id):

    from yelp_split import load_yelp
    dataset = load_yelp()['test']
    print(dataset[review_id])
    return dataset[review_id]

if __name__ == "__main__":
    load_sample_point(233)
