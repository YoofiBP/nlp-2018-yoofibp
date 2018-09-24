#!/usr/bin/env python
# coding: utf-8

# In[6]:


import math
import os
import re
import string
import math


# In[7]:


os.getcwd()


# In[8]:


arr = ['amazon_cells_labelled_training.txt','imdb_labelled_training.txt', 'yelp_labelled.txt']
test = 'IMDB.txt'


# In[9]:


def naiveBayes(test_file,array_of_training_files):
    for b in array_of_training_files:
        totalfile = open(b)
        whole = totalfile.read()
        whole = whole.translate(None, string.punctuation)
        whole = whole.replace('\t', ' ')
        wholelist = whole.split('\n')
        positiveDocs = []
        negativeDocs = []
        wholelist.pop()
        wholelist.pop()
        wholelist
        numberOfDocuments = len(wholelist)
        numberOfDocuments
        for i in wholelist:
            if(i[-1] == '1'):
                positiveDocs.append(i)
            else:
                negativeDocs.append(i)
        numberOfPositive = float(len(positiveDocs))
        numberOfNegative = float(len(negativeDocs))
        posppc = numberOfPositive/numberOfDocuments
        negppc = numberOfNegative/numberOfDocuments
        NeuVoc = []
        NegVoc = []
        PosVoc = []
        for i in wholelist:
            for i in i.split(' '):
                if(i!='1' and i!='0'):
                    NeuVoc.append(i.lower())
        NegVoc = []
        for i in negativeDocs:
            for i in i.split(' '):
                if(i!='1' and i!='0'):
                    NegVoc.append(i.lower())
        PosVoc = []
        for i in positiveDocs:
            for i in i.split(' '):
                if(i!='1' and i!='0'):
                    PosVoc.append(i.lower())
        PosBag = dict([i, PosVoc.count(i)]for i in PosVoc)
        NegBag = dict([i, NegVoc.count(i)]for i in NegVoc)
        VocBag = dict([i, NeuVoc.count(i)]for i in NeuVoc)
        del VocBag['']
        del NegBag['']
        posloglike = {}
        denom = sum(PosBag.values()) + len(VocBag)
        for key in VocBag:
            if(key in PosBag):
                poscount = PosBag[key] + 1
            else: 
                poscount = 1
            ##print(key + " " + str(poscount))
            posloglike[key] = float(poscount)/float(denom)
        negloglike = {}
        denom = sum(NegBag.values()) + len(VocBag)
        for key in VocBag:
            if(key in NegBag):
                negcount = NegBag[key] + 1
            else: 
                negcount = 1
            ##print(key + " " + str(negcount))
            negloglike[key] = float(negcount)/float(denom)
        negloglike
    source = open(test_file)
    alldata = source.read()
    alldata = alldata.translate(None, string.punctuation)
    alldata = alldata.replace('\t', ' ')
    alldata = alldata.replace(' 0','')
    alldata = alldata.replace(' 1','')
    alllist = alldata.split('\n')
    alllist
    alllist.pop()
    newfile = open('results.txt','w')
    sentencenegprobs = {}
    sentenceposprobs = {}
    for i in alllist:
        sentenceposlog = 1
        sentenceneglog = 1
        for j in i.split():
            if(j in negloglike):
                sentenceneglog*=negloglike[j]
            if(j in posloglike):
                sentenceposlog*=posloglike[j]
            totnegprob = float(sentenceneglog) *float(negppc)
            totposprob = float(sentenceposlog) * float(posppc)
            sentencenegprobs[i] = totnegprob
            sentenceposprobs[i] = totposprob
        if(sentencenegprobs[i] > sentenceposprobs[i]):
            newfile.write(i + ' 0\n')
        else:
            newfile.write(i + ' 1\n')
    newfile.close()
    print(sentencenegprobs)
    print(sentenceposprobs)


# In[10]:


naiveBayes(test,arr)

