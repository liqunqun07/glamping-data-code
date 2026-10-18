"""Historical word-mover distance procedure; downloads external word vectors."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
import gensim.downloader as api
import pyemd
from pyemd import emd
model = api.load('word2vec-google-news-300')
import sklearn
import pandas as pd
import numpy as np
def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]
sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')
sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)
distance = model.wmdistance(sentence_obama, sentence_president)
print('distance = %.4f' % distance)

data=pd.read_csv(ROOT / 'data/lineage_2023/archived_outputs/03_campingkeyword_combind60%_5_20231225.csv',encoding='utf-8')
words= data.word
dis=pd.DataFrame()
count =len(words)
for r in range(count):
    for j in range(count):
        distance = model.wmdistance(preprocess(words[r]),preprocess(words[j]))
        print('distance:', words[r],words[j], distance)
        dis.loc[words[r],words[j]] = distance
dis.fillna(0)
dis.to_csv(OUTPUT / 'tag_wmd_distance_new.csv')


