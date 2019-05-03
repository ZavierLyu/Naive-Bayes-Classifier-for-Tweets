from nltk import bigrams
import string
from collections import defaultdict
import pandas as pd
import random
from collections import Counter
from nltk.tokenize import TweetTokenizer
import datetime as dt
random.seed(42)


def removePunctuation(tweet):
    translator = tweet.maketrans('', '', string.punctuation)
    return tweet.translate(translator)


def read_data(filename):
    raw_df = pd.read_csv(filename,
                         names=["index", "tweets", "category", "type"])
    raw_df = raw_df[["tweets", "category", "type"]]
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


def bigramPacker(tweetString):
    tknzr = TweetTokenizer(preserve_case=False)
    words = tknzr.tokenize(tweetString)
    bigram_list = list(bigrams(words, pad_right=True, pad_left=True))[1:-1]
    return bigram_list


def bigram_tokenize_2(tweets_list):
    energy_words, food_words, medical_words, water_words, none_words = [], [], [], [], []
    words_list = [energy_words, food_words,
                  medical_words, water_words, none_words]
    for i in range(5):
        for j in tweets_list[i]:
            words_list[i] += bigramPacker(j)

    energy_counts = bigram_counter_2(words_list[0])
    food_counts = bigram_counter_2(words_list[1])
    medical_counts = bigram_counter_2(words_list[2])
    water_counts = bigram_counter_2(words_list[3])
    none_counts = bigram_counter_2(words_list[4])

    dic = {"Energy": energy_counts, "Food": food_counts,
           "Medical": medical_counts, "Water": water_counts,
           "None": none_counts}
    return dic


def bigram_counter_2(bigram_list):
    bigram_counts = defaultdict(lambda: Counter())
    for w1, w2 in bigram_list:
        bigram_counts[w1][w2] += 1
    return bigram_counts


def calculate_priori_laplace(filename, k=1):
    df = read_data(filename)
    len_number = len(df["category"])
    len_kind = len(set(list(df["category"])))
    cnt = Counter(list(df["category"]))
    for i in cnt:
        cnt[i] = (cnt[i]+k) / (len_number+k*len_kind)
    # print(cnt)
    return cnt


def output_train_dic(filename):
    df = read_data(filename)
    tweets_list = categorize_tweets(df)
    train_dic = bigram_tokenize_2(tweets_list)
    return train_dic


def calculate_prob_bigram_2(words, category, train_dic, priori_dic):

    prob = priori_dic[category]
    start_word = words[0][0]
    total_amount = 0
    start_amount = 0
    for c in train_dic.keys():
        for w1 in train_dic[c].keys():
            total_amount += sum(train_dic[c][w1].values())
            if (w1 == start_word) and (c == category):
                start_amount += sum(train_dic[c][w1].values())

    P0 = (start_amount + 1) / total_amount
    prob = prob * P0
    for bigram_tuple in words:
        w1 = bigram_tuple[0]
        w2 = bigram_tuple[1]
        N = 1000
        if w1 in train_dic[category].keys():
            prob *= (train_dic[category][w1][w2] + 1) / \
                (sum(train_dic[category][w1].values())+N)
        else:
            prob *= 1 / N
    prob = float("{0:.3e}".format(prob))
    return (category, prob)


def bigram_predict_2(tweets, train_dic, priori_dic, calculate_prob):
    words = bigramPacker(tweets)
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


def testing(filename, train_dic, priori_dic, calculate_prob_func, output_file):
    test_df = read_data(filename)
    test_df["prediction"] = test_df["tweets"].apply(
        lambda row: bigram_predict_2(row, train_dic, priori_dic, calculate_prob_func))
    total_amount = len(test_df.tweets)
    correct_amount = len(test_df[test_df["category"] == test_df["prediction"]])
    print("Accuracy:", correct_amount/total_amount)
    test_df.drop("type", axis=1, inplace=True)
    test_df.to_csv(output_file)


def main():
    timeStart = dt.datetime.now()
    print("BIGRAM WITH LAPLACE METHOD 2")
    priori_dic = calculate_priori_laplace(
        "data/labeled-data-singlelabels-train.csv")
    train_dic = output_train_dic("data/labeled-data-singlelabels-train.csv")
    testing("data/labeled-data-singlelabels-test.csv", train_dic, priori_dic,
            calculate_prob_bigram_2, "prediction/bigram_2_laplace.csv")
    timeEnd = dt.datetime.now()
    print("Running Time:", str(timeEnd-timeStart))

if __name__ == "__main__":
    main()
