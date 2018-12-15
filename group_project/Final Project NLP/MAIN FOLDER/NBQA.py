
# coding: utf-8

# In[2]:


import nltk
nltk.download('averaged_perceptron_tagger')
import sklearn
#Import  CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Import Multinominal Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB

#Import train_test_split to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
import re
#import pyrata.re as pyrata_re


# In[ ]:


f=open('../FAQs/Questions.txt','r',errors = 'ignore')
t=open('../FAQs/Answers.txt','r',errors = 'ignore')


# In[16]:


#listObj = line.lower().split()

def createSentence(T):
   # ray = ["what","when","where","how","where","which","how","who","whose","whom","is","are","the","an","it","is","\n",'ï»¿',"\t",".","!"]
    Fo = open(T,"r")
    listObj = []
    newObj = []
    sent = ""
    line = Fo.readline()
    while(len(line) != 0 ):
        listObj.append(line.replace("\t","").replace("\n","").replace("\r",""))

        line = Fo.readline()
    Fo.close()
    #print(newObj)
    return(listObj)

 
def removeNum(text):
    numReg = re.compile(r'^(\d+\.?)')
    for i in range(len(text)-1):
        text[i] = numReg.sub('',text[i])
    return text
    



def naiveNormalized(test):
    # normalize data
    #review_vec = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
    review_vec = CountVectorizer(
        analyzer = 'word',
        lowercase=True,
        encoding='utf-8',
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,)
    
    Question = createSentence('../FAQs/Questions.txt')
    
    Answers = createSentence('../FAQs/Answers.txt')
    #Question = removeNum(Question)
    #Answers = removeNum(Answers)
    #print(Question)

    testdoc = createSentence(test)
    print(testdoc)
   #counterAnswer = review_vec.fit
    counterQuestion = review_vec.fit_transform(Question)
    counterTestdoc = review_vec.transform(testdoc)
   
    
   
        
    
    
    
    #Converting raw frequency counts into TF-IDF values
    tfidf_transformer = TfidfTransformer()
    review_tfidf = tfidf_transformer.fit_transform(counterQuestion)
    
    
    review_tfidf2 = tfidf_transformer.fit_transform(counterTestdoc)
    

    # Split data into training and test sets
    X_trainset, X_testset,Y_trainset, Y_testset  = train_test_split(review_tfidf, Answers, train_size = 0.80, random_state = 12)
    #testd = train_test_split(review_tfidf, testdoc, test_size = 1, random_state = 12)

    
    #print(docs_test)
    # Train Naive Bayes classifier
    trainer = MultinomialNB().fit(X_trainset, Y_trainset)

    # Predicting the Test set results
    pred = trainer.predict(review_tfidf2)

    #calculate accuracy
    #print('Accuracy on the test subset for Naive Bayes is: {:.3f}'.format(sklearn.metrics.accuracy_score(y_test, y_pred)))


    
    print(pred)

    
naiveNormalized("qatest.txt")


# In[ ]:




