
import sys
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import nltk
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
import re


import LogisticsRModel
import NaiveBayesModel
import LDAModel



##def TestClassifer(testfile,vectorizer,tfidf_transformer,model):
##    reviews_new = []
##    with open(testfile) as f:
##        for i in f:
##            reviews_new.append(i[:-1])
##
##    reviews_new_counts = vectorizer.transform(reviews_new)
##    reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)
##
##    # Have classifier make a prediction
##    print("   ")
##    print("---> Making Prediction")
##    pred = model.predict(reviews_new_tfidf)
##    return pred
##
### use title array and implement this
##def WriteResults(results,Model):
##    file = open("results-"+Model+".txt", "w")
##    for result in results:
##        file.write(result)
##        file.write("\n")
##
##
##

if __name__ == "__main__":
    
    if sys.argv[1] == "topic":
        NaiveBayesModel.topic_answering(sys.argv[2],"nb","u")
        LogisticsRModel.MakePrediction(sys.argv[2])
        LDAModel.passTestFile(sys.argv[2])
    
    
else:
    print("Try again and add file name")

