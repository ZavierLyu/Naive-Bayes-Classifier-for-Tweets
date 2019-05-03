import pandas as pd
import random
from collections import Counter
from nltk.tokenize import TweetTokenizer
import numpy as np
import datetime as dt
random.seed(42)
np.set_printoptions(precision=3)


def read_data(filename):
    raw_df = pd.read_csv(filename,
                         names=["index", "tweets", "category", "type"])
    raw_df = raw_df[["tweets", "category", "type"]]
    return raw_df


def binary_classifier(c1, c2, df, tweet):
    sub_df = df[df["category"].isin([c1, c2])]
    c1_df = df[df["category"] == c1]
    c2_df = df[df["category"] == c2]
    c1_sentences = list(c1_df["tweets"])
    c2_sentences = list(c2_df["tweets"])
    tknzr = TweetTokenizer(preserve_case=False)
    c1_words = []
    c2_words = []
    for sent in c1_sentences:
        c1_words += tknzr.tokenize(sent)
    for sent in c2_sentences:
        c2_words += tknzr.tokenize(sent)
    c1_count = Counter(c1_words)
    c2_count = Counter(c2_words)

    N_word = len(set(c1_words+c2_words))
    for w in c1_count.keys():
        c1_count[w] = (c1_count[w]+1) / (len(c1_words)+N_word)
    c1_count[None] = 1/(len(c1_words)+N_word)
    for w in c2_count.keys():
        c2_count[w] = (c2_count[w]+1) / (len(c2_words)+N_word)
    c2_count[None] = 1/(len(c2_words)+N_word)

    tweet_words = tknzr.tokenize(tweet)

    len_lines = len(sub_df["category"])
    N = 2
    cnt = Counter(list(df["category"]))
    for i in cnt:
        cnt[i] = (cnt[i]+1) / (len_lines+1*N)

    c1_prob = calculate_prob(tweet_words, c1, c1_count, cnt)
    c2_prob = calculate_prob(tweet_words, c2, c2_count, cnt)

    # assert c1_prob != c2_prob, "IMPOSSIBLE!"
    if c1_prob > c2_prob:
        return (c1, c1_prob-c2_prob)
    else:
        return (c2, c2_prob-c1_prob)


def calculate_prob(tweet_words, category, count, cnt):
    prob = cnt[category]
    for word in tweet_words:
        if word in count:
            prob *= count[word]
        else:
            prob *= count[None]
    return prob


def OvO_predict_1(df, tweet):
    tags = ["Energy", "Food", "Medical", "Water", "None"]
    result_df = pd.DataFrame(np.zeros([5, 5]), columns=tags, index=tags)
    round_score_list = []
    for c_1 in range(len(tags)):
        for c_2 in range(c_1+1, len(tags)):
            category_1 = tags[c_1]
            category_2 = tags[c_2]
            prediction = binary_classifier(category_1, category_2, df, tweet)
            if prediction[0] == category_1:
                result_df[category_1][category_2] = 1
                result_df[category_2][category_1] = -1
            elif prediction[0] == category_2:
                result_df[category_2][category_1] = 1
                result_df[category_1][category_2] = -1
            round_score_list.append(prediction[1])
    category_score = [(tags[i], sum(list(result_df[tags[i]])))
                      for i in range(5)]
    random.shuffle(category_score)
    category_score = sorted(category_score, key=lambda x: x[1], reverse=True)
    # print(category_score)
    return category_score[0][0]


def OvO_predict_Massey(df, tweet):
    tags = ["Energy", "Food", "Medical", "Water", "None"]
    round_score_list = []
    B = np.zeros([10, 5])
    cnt = 0
    for c_1 in range(len(tags)):
        for c_2 in range(c_1+1, len(tags)):
            category_1 = tags[c_1]
            category_2 = tags[c_2]
            prediction = binary_classifier(category_1, category_2, df, tweet)
            if prediction[0] == category_1:
                B[cnt][c_1] = 1
                B[cnt][c_2] = -1
            elif prediction[0] == category_2:
                B[cnt][c_1] = -1
                B[cnt][c_2] = 1
            round_score_list.append(prediction[1])
            cnt += 1
    round_score_array = np.array(round_score_list)
    if np.linalg.norm(round_score_array) == 0:
        return tags[random.randint(0, 4)] # When all the elements of v vector are 0
    else:
        round_score_array = round_score_array / \
            np.linalg.norm(round_score_array, ord=1)  # Normalization

    v = round_score_array
    # Using Massey method to rank the categories.
    G = np.dot(B.T, B)
    P = np.dot(B.T, v)
    G[-1] = 1
    P[-1] = 0
    G_I = np.linalg.inv(G)
    r = np.dot(G_I, P) # Get the index of the max value.
    index = np.where(r == r.max())[0][0]
    return tags[index]



def testing(test_file, train_file, output_file, predict_func):
    print("OvO strategy")
    timeStart = dt.datetime.now()
    test_df = read_data(test_file)
    train_df = read_data(train_file)
    test_df["prediction"] = test_df["tweets"].apply(
        lambda row: predict_func(train_df, row))
    total_amount = len(test_df.tweets)
    correct_amount = len(test_df[test_df["category"] == test_df["prediction"]])
    print("Accuracy:", correct_amount/total_amount)
    test_df.drop("type", axis=1, inplace=True)
    test_df.to_csv(output_file)
    timeEnd = dt.datetime.now()
    print("Running Time:", str(timeEnd-timeStart))


if __name__ == "__main__":
    testing(
        test_file="data/labeled-data-singlelabels-test.csv",
        train_file="data/labeled-data-singlelabels-test.csv",
        output_file="prediction/OvO_laplace.csv",
        predict_func=OvO_predict_Massey # Change function to OvO_predict_Massey to use graph ranking.
    )
