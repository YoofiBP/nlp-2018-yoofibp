# Sentiment Classification with Naive Bayes and Logisitic Regression
For my Natural Language Processing class, I developed Naïve Bayes and Logistic Regression classifiers to predict whether a statement is positive or negative training data provided.
library to build the classifiers. 

There are two versions for each classifier – one that includes text normalization and one that does not.


This is very useful in a scenario where an organisation would like to determine whether, across numerous reviews, they are thought of in a good way or not. 

### Requirements
python3 installed

### How to run
```
python3 lab4.py <classifier-type> <version> <testfile>
```

#### Classifier Types
nb - Naive Bayes
lr - Logistic Regression

#### Versions
n - Normalized
u - Unormalized

Example 
```
python lab4.py nb u test.txt
```

### Built With 
- Python (NLTK and scikit Learn)

NLTK was used for normalization as it has a library of common words like ‘the’, ‘a’ and ‘an’ that are removed when normalizing.

Sci-kit was the more important library used. It contained the Vectorizer Object (CountVectorizer) which was necessary for converting the reviews into a matrix of word counts. This was necessary because the classifiers (Multinomial Naive Bayes and Logistic Regression) need vector features in order to perform the classification task. It also contained Multinomial and Logistic Regression objects that were used to build models and fit to the training set.