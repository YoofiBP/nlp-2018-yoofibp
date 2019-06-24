import sys
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

import QR_Pairs


if __name__=="__main__":
    if sys.argv[1] == "qa":
        print(sys.argv[2])
        QR_Pairs.main(sys.argv[2])
    else:
        print("Try again and add a file name")
