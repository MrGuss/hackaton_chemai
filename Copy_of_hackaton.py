#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!wget https://drive.google.com/uc?id=1LL6moSzpUVxJUTMeXihWvUxBJNjvj6EH -O sample_submission.csv
!wget https://drive.google.com/uc?id=1Ui2t87X3in-Wu-pnjkDXa_VtPsVafi0l -O test.csv
!wget https://drive.google.com/uc?id=159PZX3X5rpUO-WbzWyC9whnc8B4mNqJl -O train.csv

# In[17]:


import pandas as pd

sample_sub = pd.read_csv("sample_submission.csv").set_index("index")
test = pd.read_csv("test.csv").set_index("index")
train = pd.read_csv("train.csv").set_index("index")

# In[18]:


train.info()
train.describe()

# In[19]:


train.head()

# In[ ]:



