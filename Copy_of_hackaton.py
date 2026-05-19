#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from urllib.request import urlretrieve


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



