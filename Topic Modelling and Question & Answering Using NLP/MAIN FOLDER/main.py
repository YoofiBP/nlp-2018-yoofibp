import sys
import QR_Pairs
import LogisticsRModel
import NaiveBayesModel
import LDAModelQuestionAndAnswer
import LDAModel

if __name__=="__main__":
    if sys.argv[1] == "qa":
        print("Using the idea of QR-pairs...")
        QR_Pairs.main(sys.argv[2])
        print("Using the idea of LDA with Rules...")
        LDAModelQuestionAndAnswer.passTestFile(sys.argv[2])
    elif sys.argv[1] == "topic":
        print("Using the idea of Naive Bayes...")
        NaiveBayesModel.topic_answering(sys.argv[2],"nb","u")
        print("Using the idea of Logistic Regression...")
        LogisticsRModel.MakePrediction(sys.argv[2])
        print("Using the idea of LDA...")
        LDAModel.passTestFile(sys.argv[2])
    else:
        print("Try again and add a file name")
