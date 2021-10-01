# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:35:46 2020

@author: Manas Joshi
"""

import pandas as pd
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")

combdf = traindf.append(testdf, ignore_index=True)

combdf["tweet"] = combdf["tweet"].str.lower()
combdf.sample(5)

def remove_pat(s1, pat,nstr):
    return re.sub(pat, nstr, s1)
combdf["new_tweet"]=np.vectorize(remove_pat)(combdf["tweet"],"@[\w]*", "")
combdf.sample(5)

def m1(s1,pat):
    return " ".join(re.findall(pat, s1))
combdf["hash"] = np.vectorize(m1)(combdf["new_tweet"], r"#(\w+)")
combdf["new_tweet"] = np.vectorize(remove_pat)(combdf["new_tweet"], r"#[a-z]+", "")
combdf.sample(5)

combdf["new_tweet"] = np.vectorize(remove_pat)(combdf["new_tweet"], r"\b[a-z]{1,2}\b", " ")

all_words = ' '.join([text for text in combdf['new_tweet']])
df2 = combdf['new_tweet'][combdf["label"]==1]
all_words = ' '.join([text for text in df2])
df2 = combdf['hash'][combdf["label"]==1]
all_hash = ' '.join([text for text in df2])
all_w = all_words + all_hash

a = nltk.FreqDist(all_hash.split())
d= pd.DataFrame({'Hashtag':list(a.keys()),'Count' :list(a.values())})
df = combdf[combdf["label"].isin([0,1])]
df["new_tweet"] += df["hash"]
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=2, max_features=3000 , stop_words='english',ngram_range=(1,2))
bow = tfidf_vectorizer.fit_transform(df['new_tweet'])
bow.shape

X = bow
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

yp=model.predict(X_test)
cm= confusion_matrix(yp, y_test)
print(np.shape(yp==0))
y_test[y_test==1].count()
cm

print(y_test[y_test==1].count())

threshold = 0.10
yb2_prob = model.predict_proba(X_test)
yp2 = (yb2_prob [:,1] >= threshold).astype('int')
cm = confusion_matrix(yp2, y_test)
y_test[y_test==1].count()
cm

s1= "white and black are not same"
s1new = tfidf_vectorizer.transform([s1])
p = model.predict(s1new)
print(p)
new_prob = model.predict_proba(s1new)
p2 = (new_prob [:,1] >= threshold).astype('int')
print(p)
