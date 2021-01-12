# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 01:35:45 2021

@author: Lenovo
"""

import pandas as pd
messages=pd.read_csv("SMSSpamCollection",sep='\t',names=["label","message"])
import nltk
import re #regular expressions 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer=PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

sentences=nltk.sent_tokenize(paragraph) 

corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages["message"][i])
    review=review.lower()
    review=review.split()
    review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(messages["label"])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training with naive bayes
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test, y_pred)
