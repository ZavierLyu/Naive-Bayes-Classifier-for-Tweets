from nltk import bigrams
import string
import random
from basis import read_data, categorize_tweets
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter
from laplace import calculate_priori_laplace, calculate_prob_laplace
import datetime as dt


def removePunctuation(tweet):
    translator = tweet.maketrans('', '', string.punctuation)
    return tweet.translate(translator)


def bigramReturner(tweetString):
    # tweetString = tweetString.lower()
    # tweetString = removePunctuation(tweetString)
    tknzr = TweetTokenizer(preserve_case=False)
    words = tknzr.tokenize(tweetString)
    bigramFeatureVector = []
    for item in bigrams(words):
        bigramFeatureVector.append(' '.join(item))
    return bigramFeatureVector


def bigram_tokenize(tweets_list):
    '''
    Return a dictionary {"Energy":[words], "Food":[words]}
    '''
    energy_words, food_words, medical_words, water_words, none_words = [], [], [], [], []
    words_list = [energy_words, food_words,
                  medical_words, water_words, none_words]
    for i in range(len(words_list)):
        for j in tweets_list[i]:
            words_list[i] += bigramReturner(j)
    dic = {"Energy": energy_words, "Food": food_words,
           "Medical": medical_words, "Water": water_words,
           "None": none_words}
    return dic


def bigram_laplace(filename, k=1):
    df = read_data(filename)
    tweets_list = categorize_tweets(df)
    words_dic = bigram_tokenize(tweets_list)

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


def bigram_predict(tweets, train_dic, priori_dic, calculate_prob):
    words = bigramReturner(tweets)
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


def bigram_testing(filename, train_dic, priori_dic, calculate_prob_func, output_file):
    test_df = read_data(filename)
    test_df["prediction"] = test_df["tweets"].apply(
        lambda row: bigram_predict(row, train_dic, priori_dic, calculate_prob_func))
    # print(test_df.head(10))
    total_amount = len(test_df.tweets)
    correct_amount = len(test_df[test_df["category"] == test_df["prediction"]])
    print("Accuracy:", correct_amount/total_amount)
    test_df.drop("type", axis=1, inplace=True)
    test_df.to_csv(output_file)


def main_1():
    timeStart = dt.datetime.now()
    print("BIGRAM WITH LAPLACE METHOD 1")
    priori_dic_3 = calculate_priori_laplace(
        "data/labeled-data-singlelabels-train.csv")
    train_dic_3 = bigram_laplace("data/labeled-data-singlelabels-train.csv")
    bigram_testing("data/labeled-data-singlelabels-test.csv", train_dic_3, priori_dic_3,
                   calculate_prob_laplace, "prediction/bigram_1_laplace.csv")
    timeEnd = dt.datetime.now()
    print("Running Time:", str(timeEnd-timeStart))

if __name__ == "__main__":
    main_1()