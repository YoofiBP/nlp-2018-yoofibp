#!/usr/bin/env python
# coding: utf-8

# In[28]:


def min_edit(source_word,target_word):
    n = len(source_word)
    m = len(target_word)
    matri = {}
    for i in range(n+1):
        for j in range(m+1):
            matri[str(i)+":"+str(j)]=0
        matri['0:0']=0
    for i in range(1,n+1):
        matri[str(i)+":0"] = matri[str(i-1)+":0"] + 1
    for j in range(1,m+1):
        matri["0:"+str(j)] = matri["0:"+str(j-1)] + 1
    for i in range(1,n+1):
        for j in range(1, m+1):
            if(source_word[i-1]==target_word[j-1]):
                sub = 0
            else:
                sub = 2
            matri[str(i)+":"+str(j)]=min((matri[str(i-1)+":"+str(j)])+1,(matri[str(i-1)+":"+str(j-1)])+sub,(matri[str(i)+":"+str(j-1)])+1)
    print("Minimum edit distance between "+source_word + "and " + target_word + " is "+str(matri[str(n)+":"+str(m)]))


# In[29]:


min_edit('intention','execution')


# In[ ]:




