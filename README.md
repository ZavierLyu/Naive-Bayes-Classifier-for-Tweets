# Naive Bayes Classifier for Tweets
This a attempt to use Naive Bayes Classifier to do the classification job described in [Sailors2017](https://github.com/abisee/sailors2017)

Before you run the program, please follow the instructions of configuring environment.
A Python3 is a must, anyway, make a directory as workplace and open the terminal.
``` shell
# Using pipenv as module management tool.
pip install pipenv
# Make a virtual environment.
pipenv --three
# Install relevant modules.
pipenv install pandas nltk numpy
# If pipenv does not work, use pip
pip install pandas nltk numpy
# Activate the virtual environment
pipenv shell
```
Afterwards, input `python` or `python3` to enter the interactive mode of python.
``` python
>>>import nltk
>>>nltk.download("punkt")
>>>exit()
```
And...have fun!