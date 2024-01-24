def calc_corr():
    import numpy as np
    X = np.load("amazon_sentiment.npy")
    X_1 = X[:, 0]
    X_2 = X[:, 1]
    Y = np.load("amazon_discourse.npy")
    Y_1 = Y[:, 0]
    Y_2 = Y[:, 1]
    from scipy.stats import pearsonr

    print(pearsonr(X_1, Y_1), pearsonr(X_1, Y_2))
    print(pearsonr(X_2, Y_1), pearsonr(X_2, Y_2))

if __name__ == "__main__":
    calc_corr()