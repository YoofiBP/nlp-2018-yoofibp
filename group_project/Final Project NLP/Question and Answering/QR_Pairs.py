#!/usr/bin/env python
# coding: utf-8

# In[48]:


import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
nltk.download('wordnet')


# In[49]:


def removeNum(text):
    numReg = re.compile(r'^(\d+\.?)')
    for i in range(len(text)-1):
        text[i] = numReg.sub('',text[i])
    return text


# In[50]:


def readandclean(file):
    file = open(file,'r',errors = 'ignore')
    raw = file.read()
    raw = raw.lower()
    raw = raw.replace('\t','')
    raw = raw.replace('�','')
    raw = raw.split('\n')
    raw = removeNum(raw)
    return raw


# In[51]:


def createdict(key, values):
    qrzip = zip(key, values)
    qr_pairs = dict(qrzip)
    return qr_pairs


# In[52]:


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[58]:


def response(user_response,base_corpus):
    qr_response = ''
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(base_corpus)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf == 0):
        qr_response=qr_response + "I am sorry! I don't understand you"
        return qr_response
    else:
        qr_response = qr_response+base_corpus[idx]
        return qr_response


# In[54]:


questions = readandclean('../FAQs/Questions.txt')
answers = readandclean('../FAQs/Answers.txt')


# In[55]:


qr_pairs = createdict(questions, answers)


# In[59]:


def ansques(question):
    #global word_tokens
    q = open(question,'r')
    questions_raw = q.read()
    questions_raw = questions_raw.replace('\t','')
    questions_raw = questions_raw.split('\n')
    questions_raw = removeNum(questions_raw)
    for i in questions_raw:
        if(i == ''):
            questions_raw.remove(i)
    print(questions_raw)
    results = open('questions_file.txt','w')
    for i in questions_raw:
        questions.append(i)
        results.write(qr_pairs[response(i,questions)] + '\n')
        print(qr_pairs[response(i,questions)])
        questions.remove(i)
    results.close()


# In[63]:


def main(questionfile):
    questions = readandclean('../FAQs/Questions.txt')
    answers = readandclean('../FAQs/Answers.txt')
    qr_pairs = createdict(questions, answers)
    ansques(questionfile)


# In[64]:



#main('qatest.txt')


# In[ ]:




