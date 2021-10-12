!pip install laserembeddings

!python -m laserembeddings download-models

import nltk
nltk.download('punkt')
from nltk import tokenize

from laserembeddings import Laser

laser = Laser()


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import codecs

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                label = int(line.split("\t")[2])
                y.append(label)
                corpus.append(tweet)

    return corpus, y

def parse_features(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    features = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
              line = line.rstrip()
              values = line.split(" ")
              coefs = np.asarray(values[0:], dtype='float32')
              features.append(coefs)
    return features


DIR_TRAIN = "english-balanced.tsv"
DIR_TEST = "spanish_pereira.tsv"

dataTrain, labelTrain = parse_training(DIR_TRAIN)
dataTest, labelTest = parse_training(DIR_TEST)

print("read dataset finished")

featureTrain = []
featureTest = []

#featureTrain = parse_features("english-features.txt")
#featureTest = parse_features("german_ross.txt")

 CorpusFile = './english-features.txt'
 CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")

 for tweet in dataTrain:
#   #sents = tokenize.sent_tokenize(tweet)
   embeddings = laser.embed_sentences(tweet, lang='en')
   for idx in embeddings[0] :
     CorpusFileOpen.write(str(idx))
     CorpusFileOpen.write(" ")
   CorpusFileOpen.write("\n")
   featureTrain.append(embeddings[0])
 CorpusFileOpen.close()

CorpusFile = './spanish-pereira.txt'
CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")

for tweet in dataTest:
  #sents = tokenize.sent_tokenize(tweet)
  embeddings = laser.embed_sentences(tweet, lang='es')
  for idx in embeddings[0] :
    CorpusFileOpen.write(str(idx))
    CorpusFileOpen.write(" ")
  CorpusFileOpen.write("\n")
  featureTest.append(embeddings[0])
CorpusFileOpen.close()

print("featurize finished")

model = LogisticRegression()

print("fitting")

model.fit(featureTrain,labelTrain)

print("fitted")
print("predict")

CorpusFile = './prediction.txt'
CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")

predictions = model.predict(featureTest)

for pred in predictions:
  CorpusFileOpen.write(str(pred))
  CorpusFileOpen.write("\n")
CorpusFileOpen.close()

acc = accuracy_score(labelTest,predictions)
p0 = precision_score(labelTest,predictions, pos_label=0)
r0 = recall_score(labelTest,predictions, pos_label=0)
f0 = f1_score(labelTest,predictions, pos_label=0)
p1 = precision_score(labelTest,predictions, pos_label=1)
r1 = recall_score(labelTest,predictions, pos_label=1)
f1 = f1_score(labelTest,predictions, pos_label=1)

print(p0)
print(r0)
print(f0)
print(p1)
print(r1)
print(f1)
print(acc)