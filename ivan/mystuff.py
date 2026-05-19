#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import stuff
from urllib.request import urlretrieve
import pandas as pd
import os
import gc
import math


# In[2]:


FILES = {
    "test.csv": "1Ui2t87X3in-Wu-pnjkDXa_VtPsVafi0l",
    "train.csv": "159PZX3X5rpUO-WbzWyC9whnc8B4mNqJl",
    "sample_submission.csv": "1LL6moSzpUVxJUTMeXihWvUxBJNjvj6EH",
}

for filename, file_id in FILES.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {filename}...")
    urlretrieve(url, filename)

print("Done.")

# In[3]:


sample_sub = pd.read_csv("sample_submission.csv").set_index("index")
test = pd.read_csv("test.csv").set_index("index")
train = pd.read_csv("train.csv").set_index("index")

# In[4]:


train.info()
train.describe()

# In[5]:


train.head()

# In[6]:


# check missing
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if len(missing) else "No missing values")

# In[7]:


# targets destribution
target_col = train.columns[0:3]
print(f"Target: {target_col}")
print(train[target_col].describe())
print()
# if classification:
print(train[target_col].value_counts())
