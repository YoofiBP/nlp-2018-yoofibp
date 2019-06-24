#!/usr/bin/env python
# coding: utf-8

# In[104]:


import nltk
from textblob import TextBlob
import csv
import re
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import time
import string
from sklearn.feature_extraction.text import TfidfTransformer
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[145]:


def sentiment_analyzer(classifier, norm, textfile):
    X, Y = datasplitter('amazon_cells_labelled_training.txt')
    
    '''Based on whether it is normalised or the vectorizer converts
    the text into a matrix of token(word) counts. Each word in
    the entire corpus is counted and the word count in each review
    is recorded. The four lines below 'train' the vectorizer on the 
    corpus of reviews X. Hence anything passed through the vectorizer
    will be counted against the corpus X. The analyzer parameter
    will run the function stated on each entry before outputting tokens'''
    if(norm == "n"):
        transformer = CountVectorizer(analyzer=normalize)
        
    else:
        transformer = CountVectorizer(analyzer=splitup)
        
    '''Tokenizing each word in X against the entire corpus'''
    myfile = open(textfile)
    myfile_data = myfile.read()
    myfile_data = myfile_data.split('\n')
    #myfile_data = splitup(myfile_data)
    fitted= transformer.fit_transform(X)
    real_x = transformer.transform(myfile_data)
    real_f = TfidfTransformer()
    tf_transform = real_f.fit_transform(fitted)
    
    '''Splitting our data into training and test data. X_train 
    contains part of reviews while y_train contains the their 
    corresponding sentiments. X_test is the test corpus with y_test
    its corresponding sentiments. The test_size parameter specifies
    the percentage of the entire corpus (real_x) to make testing data'''
    X_train, X_test, y_train, y_test = train_test_split(fitted, Y, test_size=0.3, random_state=101)
    
    '''A Multinomial Naive Bayes (MultinomialNB) is used as it is a 
    version of Naive Bayes designed more for text documents.
    Logisitic Regression classifier (LogisitcRegression) trains and 
    classifies using Logisitic Regression'''
    
    if(classifier == "nb"):
        class_type = MultinomialNB()
    else:
        class_type = LogisticRegression()
        
    '''Training the classifier with the training data X_train
    and y_train. Think of training as the classifier learning
    the sentiments each word expresses with the goal of predicting
    the sentiment of a new word.'''
    class_type.fit(X_train,y_train)
    
    '''Predicting the class of test data having been learned
    with training data'''
    pred = class_type.predict(real_x)
    
    '''Writing results to a text file'''
    newfile = open('results-'+classifier+'-'+norm+'.txt','w')
    for i in pred:
        newfile.write(i+'\n')
    newfile.close()
    #result_breakdown(y_test,pred)


# In[146]:


def datasplitter(textfile):
    #Opening and reading the textfile
    data = open(textfile)
    text = data.read()
    
    '''Creating a list of sentences by splitting based on
    the new line character and removing spaces that appear
    as individual elements'''
    text2 = text.split('\n')
    text2.remove('')
    sentiments = []
    reviews = []
    pos_neg=[]
    
    '''Removing sentiment scores from each list element using
    regular expressions. It identifies a tab character 
    following by a number, puts it into a list and removes it
    from the sentence it was found. All tab characters are 
    then removed from the sentiment list to leave only the 
    scores'''
    regex = re.compile(r'\t.{1}')
    for i in text2:
        if(regex.search(i)!=None):
            match = regex.search(i).group()
            sentiments.append(match)
            reviews.append(i.replace(match,''))
    for i in sentiments:
        pos_neg.append(i.replace('\t',''))
        
    '''Returns a list of reviews and a corresponding list
    of sentiments expressed'''
    return reviews, pos_neg


# In[147]:


'''Function to normalize text by removing stopwords: common 
words like the, a, an etc. It also removes punctuations and 
splits up words to return a list of remaining words'''
def normalize(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[148]:


'''Function to just split up into words and do nothing more'''
def splitup(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc.split()


# In[149]:


'''Function to print out the summary of the precision, recall
and F1-score of the classification'''
def result_breakdown(y_test, classifier):
    print(confusion_matrix(y_test, classifier))
    print('\n')
    print(classification_report(y_test, classifier))


# In[138]:


sentiment_analyzer(sys.argv[1],sys.argv[2],sys.argv[3])


# In[150]:


sentiment_analyzer("lr", "u", "test_sentences.txt")


# In[ ]:




