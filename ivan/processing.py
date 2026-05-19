#!/usr/bin/env python
# coding: utf-8

# # Процессинг

# In[294]:


from urllib.request import urlretrieve

# In[295]:


# CORE
import warnings
from pathlib import Path

# DATA
import numpy as np
import pandas as pd

# PREPROCESSING
from sklearn.impute import SimpleImputer

# METRICS
from sklearn.metrics import (
    mean_squared_error,
)

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# In[296]:


FILES = {
    "test.csv": "1Ui2t87X3in-Wu-pnjkDXa_VtPsVafi0l",
    "train.csv": "159PZX3X5rpUO-WbzWyC9whnc8B4mNqJl",
    "sample_submission.csv": "1LL6moSzpUVxJUTMeXihWvUxBJNjvj6EH",
}

TARGETS = ['IC50', 'CC50', 'SI']

for filename, file_id in FILES.items():
    if not Path(filename).is_file():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        urlretrieve(url, filename)

sumbission_test = pd.read_csv("test.csv").set_index("index")
df = pd.read_csv("train.csv").set_index("index")
df.rename(columns={"IC50, mM": "IC50", "CC50, mM": "CC50"}, inplace=True)
Y = df[TARGETS]
X = df.drop(TARGETS, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# In[297]:


class DropConstantColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_cols_ = X.columns[X.nunique() > 1]
        return self

    def transform(self, X):
        return X[self.keep_cols_]

class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        seen = {}
        self.keep_cols_ = []

        for col in X.columns:
            key = tuple(X[col].values)
            if key not in seen:
                seen[key] = col
                self.keep_cols_.append(col)

        return self

    def transform(self, X):
        return X[self.keep_cols_]

class DropCorrelatedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.threshold)
        ]

        self.keep_cols_ = [c for c in X.columns if c not in to_drop]
        return self

    def transform(self, X):
        return X[self.keep_cols_]

# In[298]:


x_pipeline = Pipeline([
    ("constants", DropConstantColumns()),
    ("duplicates", DropDuplicateColumns()),
    ("correlation", DropCorrelatedColumns(0.95)),
    ("imputer", SimpleImputer(strategy="median")),
])

# In[ ]:


# Модель чисто рандом, чтобы посмотреть есть ли вообще результат
#TODO: подобрать параметры + попробовать другие модельки (catboost, xdgboost...)

base_gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
)

model = MultiOutputRegressor(base_gb)

X_train_p = x_pipeline.fit_transform(X_train)
X_test_p = x_pipeline.transform(X_test)

model.fit(X_train_p, y_train)

y_pred = model.predict(X_test_p)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

for i, col in enumerate(y_test.columns):
    rmse_i = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print(f"{col}: {rmse_i}")


# In[300]:


sumbission_test_p = x_pipeline.transform(sumbission_test)

y_pred = model.predict(sumbission_test_p)

y_pred = pd.DataFrame(y_pred, index=sumbission_test.index, columns=y_test.columns)

y_pred.to_csv("submission.csv")
