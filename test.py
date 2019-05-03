from nltk.corpus import reuters
from collections import Counter
import pandas as pd 

df = pd.read_csv("data\\labeled-data-singlelabels-train.csv", names=["index", "tweets", "category", "type"])
df.drop(["index"], axis=1, inplace=True)

df_energy = df[df["category"]=="Energy"]
energy_list = list(df_energy.tweets)

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <-- !"
s1 = "This is where we stand! Hail House Mormont!"
a = []
words_list = [a]
tokens_0 = tknzr.tokenize(s0)
tokens_1 = tknzr.tokenize(s1)
words_list[0] = tokens_0 + tokens_1
print(Counter(words_list[0]))

def foo(a, b):
    print(a+b)

def bar(func):
    foo(1,2)

bar(foo)