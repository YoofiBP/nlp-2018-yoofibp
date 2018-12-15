# Implemented by Emmanuel Jojoe Ainoo (LOGISTICS REGRESSION CLASSFIER)

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


# Initializing and Managing Datasets
print("---> Initializing and Managing Datasets")
print(" ")

def ManageTrainData(trainDoc):
    data = []
    with open(trainDoc) as f:
        for doc in f:
            data.append(doc)
    return data

def ManageTrainClass(trainClass):
    data_labels = []
    with open(trainClass) as file:
        for c in file:
            data_labels.append(c)
    return data_labels

# Vectorizing Data for Normalized
print("---> Transforming/Vectorizing Data(Normalized Version) ")
print(" ")
#Function to Transform data into into counts and normalize it
def VectorizeNorm(data):
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
    )
    features = vectorizer.fit_transform(data)
    features_nd = features.toarray()

    # Convert raw frequency counts into TF-IDF values
    tfidf_transformer = TfidfTransformer()
    sent_tfidf = tfidf_transformer.fit_transform(features).toarray()
    return[features,features_nd,tfidf_transformer,sent_tfidf,vectorizer]
#
# Spliting Data for Training and Testing
print("---> Spliting Data for Training and Testing ")
print(" ")

def Split(features_nd,data_labels):
    X_train, X_test, y_train, y_test  = train_test_split(
            features_nd,
            data_labels,
            train_size=0.8,
            random_state=1234)
    return[X_train, X_test, y_train, y_test]

# Train a Logistics Regression classifier
print("---> Training Logistics Regression Classifier ")
print(" ")

def LogisticsRegressionTrainer(X_train, y_train):
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)# Call Train Data on Naive Bayes
    return log_model

# Predicting the Test set results, find accuracy
print(" ")
print("---> Predicting Test set Results ")
def PredictResults(model,X_test,y_test):
    y_pred = model.predict(X_test)
    sklearn.metrics.accuracy_score(y_test, y_pred)
    evaluate = sklearn.metrics.accuracy_score(y_test, y_pred)
    return [y_pred,evaluate]

# #Function to evaluate the Models
def Evaluate(evaluate):
    return evaluate


def main():
    print("Starting")

def TestClassifer(testfile,vectorizer,tfidf_transformer,model):
    reviews_new = []
    with open(testfile) as f:
        for i in f:
            reviews_new.append(i[:-1])

    reviews_new_counts = vectorizer.transform(reviews_new)
    reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)

    # Have classifier make a prediction
    print("   ")
    print("----> Making Prediction")
    pred = model.predict(reviews_new_tfidf)
    return pred


def MakePrediction(file):
    traindoc = "../FAQs/Questions.txt"
    trainClass = "../FAQs/Topics.txt"

    data = ManageTrainData(traindoc)
    classes = ManageTrainClass(trainClass)
    vector = VectorizeNorm(data)

    split = Split(vector[1],classes)
    trainLRModel = LogisticsRegressionTrainer(split[0], split[2])
    predLR = PredictResults(trainLRModel,split[1],split[3])
    # logEv =  Evaluate(predLR[1])
    # print(logEv)

    logTest = TestClassifer(file,vector[4],vector[2],trainLRModel)
    To = open("topic_results.txt","a")
    To.write(" ------------------> Topic Modelling using Logistic Regression <------------------------------ \n")
    #To.write("Using Naive bayes \n")
    for i in logTest:
        To.write(i+" \n")
    To.close()
