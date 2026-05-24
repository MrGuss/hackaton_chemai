#!/usr/bin/env python
# coding: utf-8

# Что нашел:
# 1. Таргет SI считается как отношение CC50 к IC50. Можно было бы обучить модели на CC50 и IC50, а SI посчитать, но по условиям хакатона для SI тоже нужна модель;
# 2. Длинные хвосты у таргетов;
# 3. Дубликаты признаков с разными таргетами;
# 4. Константные признаки, которые нужно удалить;
# 5. Между признаками много корреляций. Линейные модели без регуляризации не подойдут.
# 
# Предлагаю:
# 1. Удалить index из признаков;
# 2. Заполнять пропуски медианой;
# 3. Удалить константные признаки через `VarianceThreshold`;
# 4. Обучать модели на логарифмированных таргетах;
# 5. Схлопнуть дубликаты признаков в одну строку, а таргеты агрегировать через медиану;
# 6. Использовать `GroupKFold`, чтобы дубликаты не попали в обучение и проверку;
# 7. Сравнить минимум несколько моделей: линейные с регуляризацией, пару деревьев и бустинги (как минимум catboost и lgbm);
# 8. Для feature engineering использовать простые отношения и агрегации.

# ## Загрузка библиотек и данных

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, ks_2samp

pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 100)

# In[2]:


train_path = Path("datasets/train.csv")
test_path = Path("datasets/test.csv")

if not train_path.exists():
    train_path = Path("/mnt/data/train.csv")
if not test_path.exists():
    test_path = Path("/mnt/data/test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_cols = ["IC50, mM", "CC50, mM", "SI"]
id_col = "index"

feature_cols = [col for col in train.columns if col not in target_cols + [id_col]]

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Number of features:", len(feature_cols))

print()
print("Типы данных в train:")
display(train.dtypes.value_counts())

display(train.head())

# ## EDA

# In[3]:


print("Train missing values:", train.isna().sum().sum())
print("Test missing values:", test.isna().sum().sum())

missing_train = train.isna().sum()
missing_test = test.isna().sum()

display(missing_train[missing_train > 0].sort_values(ascending=False))
display(missing_test[missing_test > 0].sort_values(ascending=False))

# In[4]:


si_check = train["CC50, mM"] / train["IC50, mM"]
diff = (train["SI"] - si_check).abs()

print("Максимальное отличие:", diff.max())
print("Среднее отличие:", diff.mean())

# In[18]:


display(train[target_cols].describe().T)

for col in target_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(train[col], bins=40)
    plt.title(f"Распределение {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(np.log(train[col]), bins=40)
    plt.title(f"Log распределение {col}")
    plt.xlabel(f"log({col})")
    plt.ylabel("Count")
    plt.show()

# In[6]:


# Константные признаки

nunique = train[feature_cols].nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()

print("Количество константных признаков:", len(constant_cols))
constant_cols

# In[7]:


# Доля нулей в признаках

zero_frac = (train[feature_cols] == 0).mean().sort_values(ascending=False)

display(zero_frac.head(60))

high_zero_cols = zero_frac[zero_frac > 0.95].index.tolist()
print("Признаков с долей нулей > 95%:", len(high_zero_cols))

# In[8]:


# Дубликаты по признакам

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

train_hash = pd.util.hash_pandas_object(X, index=False)
test_hash = pd.util.hash_pandas_object(X_test, index=False)

print("Дубликатов по признакам в train:", X.duplicated().sum())
print("Дубликатов по признакам в test:", X_test.duplicated().sum())

test_matches = test_hash.isin(set(train_hash))
print("Строк test, которые точно совпадают с train по признакам:", test_matches.sum())

# In[9]:


# Корреляции признаков с логарифмами таргетов

log_targets = np.log(train[target_cols])

corr_rows = []

for feature in feature_cols:
    x = train[feature]

    for target in target_cols:
        y = log_targets[target]
        mask = x.notna() & y.notna()

        if x[mask].nunique() > 1:
            corr = spearmanr(x[mask], y[mask]).correlation
        else:
            corr = np.nan

        corr_rows.append({
            "feature": feature,
            "target": target,
            "spearman_corr": corr,
            "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
        })

corr_df = pd.DataFrame(corr_rows)

for target in target_cols:
    print(target)
    display(
        corr_df[corr_df["target"] == target]
        .sort_values("abs_corr", ascending=False)
        .head(20)
    )

# In[10]:


# Train/test drift

drift_rows = []

for feature in feature_cols:
    train_values = train[feature].dropna()
    test_values = test[feature].dropna()

    if train_values.nunique() > 1 or test_values.nunique() > 1:
        ks_stat, p_value = ks_2samp(train_values, test_values)
    else:
        ks_stat, p_value = np.nan, np.nan

    drift_rows.append({
        "feature": feature,
        "ks_stat": ks_stat,
        "p_value": p_value,
        "train_mean": train_values.mean(),
        "test_mean": test_values.mean(),
    })

drift_df = pd.DataFrame(drift_rows)

display(
    drift_df
    .sort_values("ks_stat", ascending=False)
    .head(30)
)

# In[11]:


# Связь между таргетами

target_ratio_check = train["CC50, mM"] / train["IC50, mM"]
si_diff = (train["SI"] - target_ratio_check).abs()

print("Максимальная разница между SI и CC50 / IC50:", si_diff.max())
print("Средняя разница между SI и CC50 / IC50:", si_diff.mean())

log_target_corr = np.log(train[target_cols]).corr()
display(log_target_corr)

plt.figure(figsize=(7, 5))
plt.scatter(
    np.log(train["IC50, mM"]),
    np.log(train["CC50, mM"]),
    alpha=0.45,
)
plt.title("Связь log(IC50) и log(CC50)")
plt.xlabel("log(IC50, mM)")
plt.ylabel("log(CC50, mM)")
plt.grid(True, alpha=0.3)
plt.show()

# In[12]:


# Хвосты и выбросы в таргетах

target_quantiles = train[target_cols].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

display(target_quantiles)

for col in target_cols:
    plt.figure(figsize=(8, 4))
    plt.boxplot(np.log(train[col]), vert=False)
    plt.title(f"Boxplot для log({col})")
    plt.xlabel(f"log({col})")
    plt.grid(True, alpha=0.3)
    plt.show()

top_si_cols = [
    id_col,
    "IC50, mM",
    "CC50, mM",
    "SI",
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "qed",
]

display(
    train[top_si_cols]
    .sort_values("SI", ascending=False)
    .head(10)
)

# In[13]:


# Соединения с низким IC50 и высоким CC50 

ic50_q25 = train["IC50, mM"].quantile(0.25)
cc50_q75 = train["CC50, mM"].quantile(0.75)
si_q75 = train["SI"].quantile(0.75)

candidate_mask = (train["IC50, mM"] <= ic50_q25) & (train["CC50, mM"] >= cc50_q75)
high_si_mask = train["SI"] >= si_q75
low_si_mask = train["SI"] <= 1

print("Порог низкого IC50, 25% квантиль:", ic50_q25)
print("Порог высокого CC50, 75% квантиль:", cc50_q75)
print("Порог высокого SI, 75% квантиль:", si_q75)
print("Кандидатов с низким IC50 и высоким CC50:", candidate_mask.sum())
print("Соединений с SI выше 75% квантиля:", high_si_mask.sum())
print("Соединений с SI <= 1:", low_si_mask.sum())

candidate_cols = [
    id_col,
    "IC50, mM",
    "CC50, mM",
    "SI",
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "qed",
]

display(
    train.loc[candidate_mask, candidate_cols]
    .sort_values("SI", ascending=False)
    .head(15)
)

core_cols = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "qed",
]
core_cols = [col for col in core_cols if col in feature_cols]

comparison = pd.DataFrame({
    "all_median": train[core_cols].median(),
    "good_candidates_median": train.loc[candidate_mask, core_cols].median(),
    "high_si_median": train.loc[high_si_mask, core_cols].median(),
    "low_si_median": train.loc[low_si_mask, core_cols].median(),
})

display(comparison)

# In[14]:


# Дубликаты по признакам и разброс таргетов внутри них

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

train_hash = pd.util.hash_pandas_object(X, index=False)
test_hash = pd.util.hash_pandas_object(X_test, index=False)

train_with_hash = train.copy()
train_with_hash["feature_hash"] = train_hash

hash_counts = train_with_hash["feature_hash"].value_counts()
duplicate_hashes = hash_counts[hash_counts > 1].index

print("Групп дубликатов в train:", len(duplicate_hashes))
print("Строк train, входящих в группы дубликатов:", hash_counts[hash_counts > 1].sum())
print("Максимальный размер одной группы дубликатов:", hash_counts.max())
print("Дубликатов по признакам в test:", X_test.duplicated().sum())
print("Строк test, полностью совпадающих с train по признакам:", test_hash.isin(set(train_hash)).sum())

dup_group_stats = (
    train_with_hash
    .groupby("feature_hash")
    .agg(
        size=("feature_hash", "size"),
        ic50_min=("IC50, mM", "min"),
        ic50_max=("IC50, mM", "max"),
        cc50_min=("CC50, mM", "min"),
        cc50_max=("CC50, mM", "max"),
        si_min=("SI", "min"),
        si_max=("SI", "max"),
    )
)

dup_group_stats = dup_group_stats[dup_group_stats["size"] > 1].copy()

dup_group_stats["ic50_ratio"] = dup_group_stats["ic50_max"] / dup_group_stats["ic50_min"]
dup_group_stats["cc50_ratio"] = dup_group_stats["cc50_max"] / dup_group_stats["cc50_min"]
dup_group_stats["si_ratio"] = dup_group_stats["si_max"] / dup_group_stats["si_min"]

print()
print("Разброс таргетов внутри групп дубликатов:")
display(dup_group_stats[["size", "ic50_ratio", "cc50_ratio", "si_ratio"]].describe())

display(
    dup_group_stats
    .sort_values("si_ratio", ascending=False)
    .head(10)
)

# In[15]:


# Сильно коррелирующие признаки между собой

nunique = train[feature_cols].nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()

non_constant_features = [col for col in feature_cols if col not in constant_cols]
feature_corr = train[non_constant_features].corr(method="spearman").abs()
upper_mask = np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
upper_corr = feature_corr.where(upper_mask)

high_corr_pairs = (
    upper_corr
    .stack()
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_spearman_corr"})
    .sort_values("abs_spearman_corr", ascending=False)
)

print("Пар признаков с abs Spearman corr > 0.95:", (high_corr_pairs["abs_spearman_corr"] > 0.95).sum())
display(high_corr_pairs.head(25))

# **Что видно:** сильного катастрофического сдвига между train и test по базовым химическим признакам не видно. Test чуть отличается по признакам размера молекулы, но не настолько, чтобы считать выборки из разных миров. Это хороший знак: можно использовать обычную валидацию, но с группами по дубликатам.

# In[16]:


# Feature engineering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hetero_ratio"] = df["NumHeteroatoms"] / (df["HeavyAtomCount"] + 1)
    df["donor_acceptor_sum"] = df["NumHDonors"] + df["NumHAcceptors"]
    df["donor_acceptor_ratio"] = df["NumHDonors"] / (df["NumHAcceptors"] + 1)
    df["rotatable_per_heavy"] = df["NumRotatableBonds"] / (df["HeavyAtomCount"] + 1)
    df["rings_per_heavy"] = df["RingCount"] / (df["HeavyAtomCount"] + 1)
    df["aromatic_ratio"] = df["NumAromaticRings"] / (df["RingCount"] + 1)
    df["tpsa_per_heavy"] = df["TPSA"] / (df["HeavyAtomCount"] + 1)
    df["mol_logp_per_heavy"] = df["MolLogP"] / (df["HeavyAtomCount"] + 1)
    df["mol_mr_per_molwt"] = df["MolMR"] / (df["MolWt"] + 1)

    fr_cols = [col for col in df.columns if col.startswith("fr_")]
    vsa_cols = [col for col in df.columns if "VSA" in col]
    estate_cols = [col for col in df.columns if "EState" in col]

    df["fr_total"] = df[fr_cols].sum(axis=1)
    df["fr_nonzero_count"] = (df[fr_cols] > 0).sum(axis=1)
    df["vsa_total"] = df[vsa_cols].sum(axis=1)
    df["vsa_nonzero_count"] = (df[vsa_cols] > 0).sum(axis=1)
    df["estate_abs_sum"] = df[estate_cols].abs().sum(axis=1)

    return df

X_base = train[feature_cols].copy()
X_test_base = test[feature_cols].copy()

X_fe = add_features(X_base)
X_test_fe = add_features(X_test_base)

new_cols = [col for col in X_fe.columns if col not in X_base.columns]

print("Было признаков:", X_base.shape[1])
print("Стало признаков:", X_fe.shape[1])
print("Новых признаков:", len(new_cols))

display(X_fe[new_cols].head())

# In[ ]:



