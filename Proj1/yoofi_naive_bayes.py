def naiveBayes(file):
    source = open(file)
    alldata = test1.read()
    alldata = alldata.translate(None, string.punctuation)
    alldata = alldata.replace('\t', ' ')
    alldata = alldata.replace(' 0','')
    alldata = alldata.replace(' 1','')
    alllist = alldata.split('\n')
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
        print(totnegprob)
        totposprob = float(sentenceposlog) * float(posppc)
        print(totposprob)
        sentencenegprobs[i] = totnegprob
        print(" ")
        print(sentencenegprobs)
        sentenceposprobs[i] = totposprob
        if(sentencenegprobs[i] > sentenceposprobs[i]):
            newfile.write(i + ' 0\n')
        else:
            newfile.write(i + ' 1\n')
    newfile.close()
