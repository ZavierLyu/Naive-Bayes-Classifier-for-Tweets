import pandas as pd
import random
from collections import Counter
from nltk.tokenize import TweetTokenizer
import datetime as dt
random.seed(42)


def read_data(filename):
    raw_df = pd.read_csv(filename,
                         names=["index", "tweets", "category", "type"])
    raw_df = raw_df[["tweets", "category", "type"]]
    return raw_df


def tokenize(tweets_list):
    '''
    Return a dictionary {"Energy":[words], "Food":[words]}
    '''
    tknzr = TweetTokenizer(preserve_case=False)
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


def testing(filename, train_dic, priori_dic, calculate_prob_func, output_file):
    test_df = read_data(filename)
    test_df["prediction"] = test_df["tweets"].apply(
        lambda row: predict(row, train_dic, priori_dic, calculate_prob_func))
    # print(test_df.head(10))
    total_amount = len(test_df.tweets)
    correct_amount = len(test_df[test_df["category"] == test_df["prediction"]])
    print("Accuracy",correct_amount/total_amount)
    test_df.drop("type", axis=1, inplace=True)
    test_df.to_csv(output_file)


def predict(tweets, train_dic, priori_dic, calculate_prob):
    tknzr = TweetTokenizer(preserve_case=False)
    words = tknzr.tokenize(tweets)
    energy_prob = calculate_prob(words, "Energy", train_dic, priori_dic)
    food_prob = calculate_prob(words, "Food", train_dic, priori_dic)
    medical_prob = calculate_prob(words, "Medical", train_dic, priori_dic)
    water_prob = calculate_prob(words, "Water", train_dic, priori_dic)
    none_prob = calculate_prob(words, "None", train_dic, priori_dic)
    prob_list = [none_prob, energy_prob, food_prob, medical_prob, water_prob]
    # Randomize the default prediction if all probability is 0
    # to test the robustness of the algorithm.
    random.shuffle(prob_list)

    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    # print(prob_list)
    return prob_list[0][0]


def calculate_priori_laplace(filename, k=1):
    df = read_data(filename)
    len_number = len(df["category"])
    len_kind = len(set(list(df["category"])))
    cnt = Counter(list(df["category"]))
    for i in cnt:
        cnt[i] = (cnt[i]+k) / (len_number+k*len_kind)
    # print(cnt)
    return cnt


def calculate_prob_laplace(words, category, train_dic, priori_dic):
    prob = priori_dic[category]
    for i in words:
        if i in train_dic[category]:
            prob *= train_dic[category][i]
        else:
            prob *= train_dic[category][None]
    prob = float("{0:.3e}".format(prob))
    return (category, prob)


def unigram_laplace(filename, k=1):
    df = read_data(filename)
    tweets_list = categorize_tweets(df)
    words_dic = tokenize(tweets_list)

    train_dic = {}
    len_set = len(set([x for j in words_dic.keys() for x in words_dic[j]]))
    for key, value in words_dic.items():
        cnt = Counter(value)
        len_cnt = len(value)
        for word in cnt:
            cnt[word] = (cnt[word]+k) / (float(len_cnt)+k*len_set)
        cnt[None] = k / (float(len_cnt) + k*len_set)
        train_dic[key] = cnt
    # print(train_dic["Energy"])
    return train_dic


def main():
    timeStart = dt.datetime.now()
    print("UNIGRAM WITH LAPLACE")
    priori_dic_2 = calculate_priori_laplace(
        "data/labeled-data-singlelabels-train.csv", k=1)
    train_dic_2 = unigram_laplace(
        "data/labeled-data-singlelabels-train.csv", k=1)
    testing("data/labeled-data-singlelabels-test.csv", train_dic_2, priori_dic_2, calculate_prob_laplace,
            "prediction/unigram_with_laplace.csv")
    timeEnd = dt.datetime.now()
    print("Running Time:", str(timeEnd-timeStart))

if __name__ == "__main__":
    main()
