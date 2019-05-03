from training import *


train_dic = unigram("data\\labeled-data-singlelabels-train.csv")


def testing(filename, train_dic):
    test_df = read_data(filename)
    test_df["prediction"] = test_df.apply(lambda row: predict())