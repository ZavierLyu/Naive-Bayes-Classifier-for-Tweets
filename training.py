import pandas as pd
from collections import Counter
from nltk.tokenize import TweetTokenizer


def read_data(filename):
    raw_df = pd.read_csv(filename)
    raw_df = pd.read_csv("data\\labeled-data-singlelabels-train.csv",
                         names=["index", "tweets", "category", "type"])
    raw_df.drop(["index"], axis=1, inplace=True)
    return raw_df


def categorize_tweets(df):
    df_energy = df[df["category"] == "Energy"]
    energy_list = list(df_energy.tweets)
    df_energy = df[df["category"] == "Food"]
    food_list = list(df_energy.tweets)
    df_energy = df[df["category"] == "Medical"]
    medical_list = list(df_energy.tweets)
    df_energy = df[df["category"] == "Water"]
    water_list = list(df_energy.tweets)
    df_energy = df[df["category"] == "None"]
    none_list = list(df_energy.tweets)
    tweets_list = [energy_list, food_list,
                   medical_list, water_list, none_list]
    return tweets_list


def tokenize(tweets_list):
    '''
    Return a dictionary {"Energy":[words], "Food":[words]}
    '''
    tknzr = TweetTokenizer(preserve_case=True)
    energy_words, food_words, medical_words, water_words, none_words = [], [], [], [], []
    words_list = [energy_words, food_words,
                  medical_words, water_words, none_words]
    for i in range(len(words_list)):
        for j in tweets_list[i]:
            words_list[i] += tknzr.tokenize(j)
    dic = {"Energy": energy_words, "Food": food_words,
           "Medical": medical_words, "Water": water_words,
           "None": none_words}
    return dic


def unigram(filename):
    df = read_data(filename)
    tweets_list = categorize_tweets(df)
    words_dic = tokenize(tweets_list)

    prob_dic = {}
    for key, value in words_dic.items():
        cnt = Counter(value)
        len_cnt = len(value)
        for word in cnt:
            cnt[word] /= float(len_cnt)
        prob_dic[key] = cnt
    # print(prob_dic["Energy"])
    return prob_dic


def calculate_prob(words, category, train_dic):
    prob = 1
    for i in words:
        if i in train_dic[category]:
            prob *= train_dic[category][i]
        else:
            prob = 0  # without Laplace Smoothing
            break
    return (category, prob)


# train_dic = unigram("data\\labeled-data-singlelabels-train.csv")


def predict(tweets, train_dic):
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(tweets)
    energy_prob = calculate_prob(words, "Energy", train_dic)
    food_prob = calculate_prob(words, "Food", train_dic)
    medical_prob = calculate_prob(words, "Medical", train_dic)
    water_prob = calculate_prob(words, "Water", train_dic)
    none_prob = calculate_prob(words, "None", train_dic)
    prob_list = [energy_prob, food_prob, medical_prob, water_prob, none_prob]
    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    print(prob_list)
    return prob_list[0]

