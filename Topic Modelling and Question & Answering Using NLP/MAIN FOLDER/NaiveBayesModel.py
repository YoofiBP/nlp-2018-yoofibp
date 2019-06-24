
# coding: utf-8

# In[86]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import sys
import nltk
import random
import sklearn
import string
from sklearn.metrics.pairwise import cosine_similarity
import re


# In[87]:


topic_tags = []

#the derive_unique_topics() function is meant to derive the unique topics in the data set and return a dictionary filled with the 
#keys being the various topics and the various values being the unique identifiers.
def derive_unique_topics(topics_info):
    tag_of_info = 0
    number_tags = []
    topics = open(topics_info,"r")
    clean_data = open("clean_data.txt","w")
    for i in topics:
        sentence = i.split(".")
        if sentence[0].isdigit():
            k = sentence[1].rstrip("\n")
            k = k.rstrip()
            k = k.lstrip('ï»¿')
            k = k.lstrip()
            clean_data.write(k+"\n")
            if not(k in topic_tags):
                topic_tags.append(k)
        else:
            k = sentence[0].rstrip("\n")
            k = k.rstrip()
            k = k.lstrip('ï»¿')
            k = k.lstrip()
            clean_data.write(k+"\n")
            if not(k in topic_tags):
                topic_tags.append(k)
    clean_data.close()
    topics.close()
    #converting the list into a dictionary
    for i in range(len(topic_tags)+1):
        number_tags.append(i)
    #create zip object
    zipObj = zip(topic_tags,number_tags)
    dictOfWords = dict(zipObj)
    return dictOfWords

def create_tag_files(filename):
    dictionary = derive_unique_topics(filename)
    labelled_topics = open("labelled_topics.txt","w")
    clean_data = open("clean_data.txt","r")
    for line in clean_data:
        line = line.rstrip("\n")
        value = dictionary[line]
        labelled_topics.write(str(value)+"\n")
    clean_data.close()
    labelled_topics.close()

    



# In[88]:


#train_and_testLR() trains a logistic regression model on the data given and calculates the accuracy of the logistic regression classifier
def train_and_testLR(features_array, features_test, classes, classifier,version):
    #the train_test_split() function is responsible for splitting a matrix of features into the matrix of features to be used for either training or testing. 
    X_trainset, X_testset,Y_trainset, Y_testset = train_test_split(features_array,classes,train_size = 0.80,random_state = 1234)
    #the fit() funtion is responsible for training the logistic regression classifier on the matrix of features extracted for training.
    lr = LogisticRegression().fit(X_trainset,Y_trainset)
    #the predict() function is responsble for predicting the class values of the test file given
    predictionValue = lr.predict(features_test)
    


# In[89]:


#train_and_testNB() trains a naive bayes model on the data given and calculates the accuracy of the naive bayes classifier
def train_and_testNB(features_array,features_test, classes,classifier,version):
    #the train_test_split() function is responsible for splitting a matrix of features into the matrix of features to be used for either training or testing.
    X_trainset, X_testset,Y_trainset, Y_testset = train_test_split(features_array,classes,train_size = 0.80,random_state = 1234)
    #the fit() funtion is responsible for training the naive bayes classifier on the matrix of features extracted for training.
    nb = MultinomialNB().fit(X_trainset,Y_trainset)
    #the predict() function is responsble for predicting the class values of the test file given
    predictionValue = nb.predict(features_test)
    To = open("topic_results.txt","a")
    To.write(" -------------------> Topic Modelling using Naive Bayes \n <--------------------")
    #To.write("Using Naive bayes \n")
    for i in predictionValue:
        To.write(topic_tags[i]+" \n")
    To.close()
    #evaluate = sklearn.metrics.accuracy_score(features_test, predictionValue)
    #print(evaluate)
    


# In[90]:


#transformer() is used to scale down the impact of tokens that occur very frequently in a given corpus and that are hence
#empirically less informative than features that occur in a small fraction of the training corpus.
def transformer(features):
    t_transform = TfidfTransformer()
    transform_features = t_transform.fit_transform(features)
    return (transform_features)


# In[91]:


#vectorizer() extracts features from the data being used and vectorizes these features.
def vectorizer(s,test):
    vectorizer_object = CountVectorizer()
    extracted_features = vectorizer_object.fit_transform(s)
    test_features = vectorizer_object.transform(test)
    transformed_extracted_features = transformer(extracted_features)
    return[transformed_extracted_features,test_features]


# In[92]:


#unnormalize() is meant to just obtain the unnormalized information from the data given, which is to be used in training and test files.
def topic_answering(filename,classifier_type,version):
    prediction_values = []
    sentence_values = []
    test_sentences = []
    file_info = open("../FAQs/Questions.txt", "r")
    create_tag_files("../FAQs/Topics.txt")    
    labelled_topics = open("labelled_topics.txt","r")
    test_file = open(filename,"r")
    #append all of the numerical answer to the prediction values list
    for line1 in labelled_topics:
        line1 = line1.rstrip("\n")
        line1 = line1.lstrip("ï»¿")
        prediction_values.append(int(line1))
    for line in file_info:
        #strip the newline character from the line
        line = line.rstrip("\n")
        sentence_values.append(line)
    for line in test_file:
        test_sentences.append(line)
    vectorized_info = vectorizer(sentence_values,test_sentences)
    if classifier_type == "lr":
        train_and_testLR(vectorized_info[0],vectorized_info[1],prediction_values,classifier_type,version)
    if classifier_type == "nb":
        train_and_testNB(vectorized_info[0],vectorized_info[1],prediction_values,classifier_type,version)

#topic_answering("new_Questions.txt","nb","u")
    
    

