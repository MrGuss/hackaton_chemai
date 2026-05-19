#!/usr/bin/env python
# coding: utf-8

# # Загрузка данных

# In[171]:


from urllib.request import urlretrieve

# In[172]:


# CORE
import warnings


# DATA
import numpy as np
import pandas as pd


# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns


# STATISTICS / DATA ANALYSIS
from scipy.stats import (
    ks_2samp,
)


# PREPROCESSING
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
)

# DIMENSIONALITY REDUCTION
from sklearn.decomposition import PCA

# UMAP for structure analysis
import umap


# CLUSTERING
from sklearn.cluster import (
    KMeans,
)


# MODEL VALIDATION
from sklearn.model_selection import (
    cross_val_score
)


# METRICS
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)


# MODELS
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier
)
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression
)

from sklearn.multioutput import MultiOutputRegressor


# BOOSTING
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor


# INTERPRETABILITY
import shap

warnings.filterwarnings("ignore")

# In[173]:


FILES = {
    "test.csv": "1Ui2t87X3in-Wu-pnjkDXa_VtPsVafi0l",
    "train.csv": "159PZX3X5rpUO-WbzWyC9whnc8B4mNqJl",
    "sample_submission.csv": "1LL6moSzpUVxJUTMeXihWvUxBJNjvj6EH",
}

for filename, file_id in FILES.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {filename}...")
    urlretrieve(url, filename)


# # Обзор данных

# ## Обзор данных train.csv

# In[174]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# In[175]:


# загрузка
df = pd.read_csv('./train.csv')

# In[176]:


# первые строки
df.head()

# In[177]:


df.head().T

# In[178]:


print(f'Размер данных: {df.shape}')

# In[179]:


df.info()

# In[180]:


df.select_dtypes(include='object').columns

# In[181]:


# названия столбцов
print(df.columns)

# In[182]:


# пропуски
df.isnull().sum().sort_values(ascending=False)

# In[183]:


# Дубликаты
df.duplicated().sum()

# In[184]:


# Проверка уникальных значений
df.nunique()

# In[185]:


# колонки с одним уникальным значением
constant_cols = df.columns[df.nunique()==1]

print("Количество:", len(constant_cols))
print(constant_cols.tolist())

# In[186]:


targets = ['IC50, mM', 'CC50, mM', 'SI']

df[targets].describe()

# In[187]:


constant_cols = df.columns[df.nunique()==1]

for col in constant_cols:
    print(col)
    print(df[col].value_counts())
    print('-'*30)

# In[188]:


df[constant_cols].head()

# Вывод по первичному анализу данных:
# 
# Датасет содержит 751 объект и 214 признаков.
# Пропуски практически отсутствуют: обнаружены только единичные пропуски (по 2 значения в нескольких столбцах).
# Дубликаты отсутствуют.
# Все признаки числовые: типы данных представлены float64 и int64, текстовые признаки отсутствуют.
# Найдено 18 признаков с одним уникальным значением. Такие признаки не несут информации для различения объектов и на этапе предобработки могут быть удалены.
# В датасете присутствует большое количество признаков вида fr_*.
# Вероятно, данные признаки являются химическими фрагментными дескрипторами, где значения работают как логические признаки: 0 = фрагмент отсутствует 1 = фрагмент присутствует
# То есть большое количество нулей не является ошибкой или пропущенными значениями. Нули здесь отражают отсутствие соответствующего структурного элемента в молекуле. Также некоторые признаки содержат больше двух уникальных значений, поэтому часть дескрипторов может отражать не только наличие/отсутствие структуры, но и количество определённых фрагментов.
# 
# При анализе уникальных значений были обнаружены 18 константных признаков. Все они содержат одно значение (0) для всех 751 объектов.
# 
# Большинство таких признаков относятся к группе химических фрагментных дескрипторов (fr_*), что может указывать на отсутствие соответствующих структурных элементов во всех молекулах выборки.
# 
# Поскольку константные признаки не содержат информации для различения объектов, они не участвуют в обучении модели и могут быть удалены на этапе предобработки данных.

# ## Обзор данных test.csv

# In[189]:


test = pd.read_csv('test.csv')

print(test.shape)
test.head()

# In[190]:


test.info()
test.describe()

# In[191]:


test.isna().sum().sort_values(ascending=False)

# In[192]:


print("Дубликаты:", test.duplicated().sum())

# In[193]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train:", train.shape)
print("Test:", test.shape)

# dafaq?
set(train.columns) - set(test.columns)
set(test.columns) - set(train.columns)

# In[194]:


X_train = train.drop(
    columns=targets + ['index']
)

# целевые переменные
y_train = train[targets]

# признаки test
X_test = test.drop(
    columns=['index']
)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)

# Датасет содержит молекулярные дескрипторы и три целевые переменные: IC50, CC50 и SI.
# Для задач обучения без учителя целевые переменные не использовались. Из обучающей выборки были отделены признаки и таргеты, а из обеих выборок удалён технический столбец index, не несущий смысловой информации.

# In[195]:


# объединение признаков для анализа структуры данных

full_features = pd.concat(
    [X_train, X_test],
    axis=0,
    ignore_index=True
)

print(full_features.shape)

# Для кластеризации и анализа структуры признакового пространства использовалась объединённая выборка train + test без целевых переменных. Такой подход позволяет использовать больше объектов и получить более полное представление о распределении молекулярных дескрипторов.

# # EDA

# ## Пропуски

# In[196]:


# Пропуски: общий count по train и test

train_missing = train.isna().sum()
test_missing = test.isna().sum()

train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
test_missing = test_missing[test_missing > 0].sort_values(ascending=False)

print("=" * 60)
print("TRAIN MISSING VALUES")
print("=" * 60)
display(train_missing)

print("\n")

print("=" * 60)
print("TEST MISSING VALUES")
print("=" * 60)
display(test_missing)

# In[197]:


train.isna().sum()

# In[198]:


# Пропуски в процентах
train_missing_pct = train.isna().mean() * 100
test_missing_pct = test.isna().mean() * 100

train_missing_pct = train_missing_pct[train_missing_pct > 0].sort_values(ascending=False)
test_missing_pct = test_missing_pct[test_missing_pct > 0].sort_values(ascending=False)

print("=" * 60)
print("TRAIN MISSING %")
print("=" * 60)
display(train_missing_pct)

print("\n")

print("=" * 60)
print("TEST MISSING %")
print("=" * 60)
display(test_missing_pct)

# In[199]:


# Deep analysis: строки train с пропусками

missing_rows = train[train.isna().any(axis=1)]

print("Количество строк с пропусками в train:", missing_rows.shape[0])

display(
    missing_rows[
        ["index", "IC50, mM", "CC50, mM", "SI"]
    ]
)

print("\nПропуски по этим строкам:")
display(
    missing_rows.isna().sum()[missing_rows.isna().sum() > 0]
)

# In[200]:


# Deep analysis: строки test с пропусками

missing_rows_test = test[test.isna().any(axis=1)]

print("Количество строк с пропусками в test:", missing_rows_test.shape[0])

display(
    missing_rows_test[["index"]]
)

print("\nПропуски по этим строкам:")
display(
    missing_rows_test.isna().sum()[missing_rows_test.isna().sum() > 0]
)

# ## Анализ пропущенных значений

# Пропуски в данных встречаются крайне редко и не являются существенной проблемой для датасета.
# 
# В обучающей выборке пропуски обнаружены только у нескольких объектов, при этом отсутствуют не отдельные случайные значения, а повторяющийся блок признаков. Основные пропуски сосредоточены в дескрипторах:
# 
# - `MaxPartialCharge`
# - `MinPartialCharge`
# - `MaxAbsPartialCharge`
# - `MinAbsPartialCharge`
# - группа признаков `BCUT2D_*`
# 
# Такой паттерн больше похож не на случайную потерю данных, а на невозможность расчёта определённой группы молекулярных дескрипторов для некоторых соединений.
# 
# Дополнительная проверка показала, что объекты с пропусками не выглядят явно ошибочными, однако часть из них имеет нетипичные значения целевых переменных. Это говорит о том, что пропуски могут быть структурными, а не полностью случайными.
# 
# Поскольку количество пропусков минимально, удаление объектов нецелесообразно. На этапе подготовки данных достаточно использовать импутацию, например заполнение медианными значениями.

# ## Анализ targets

# In[201]:


TARGETS = ["IC50, mM", "CC50, mM", "SI"]

train[TARGETS].describe()

# In[202]:


train[TARGETS].hist(
    figsize=(15,5),
    bins=30
)

plt.show()

# In[203]:


train[TARGETS].boxplot(
    figsize=(10,5)
)

plt.show()

# In[204]:


log_targets = np.log1p(
    train[TARGETS]
)

log_targets.hist(
    figsize=(15,7),
    bins=30
)

plt.show()

# Распределение целевых переменных
# 
# Распределения целевых переменных имеют выраженную асимметрию и длинные правые хвосты, особенно для `SI`.
# 
# Наибольшая неоднородность наблюдается у `SI`: большинство объектов сосредоточено в области небольших значений, однако присутствуют отдельные экстремально большие значения.
# 
# После применения `log1p` распределения становятся более компактными и стабильными:
# 
# - `IC50` становится заметно более симметричным;
# - `CC50` существенно выравнивается;
# - `SI` также улучшается, хотя сохраняет более выраженную асимметрию.
# 
# Это указывает на наличие сильной скошенности и влияния выбросов в исходных данных.

# In[205]:


calculated_si = train["CC50, mM"] / train["IC50, mM"]

comparison = pd.DataFrame({
    "real_SI": train["SI"],
    "calculated_SI": calculated_si,
    "difference": np.abs(
        train["SI"] - calculated_si
    )
})

comparison.describe()

# Проверка зависимости SI
# 
# Проверка показала, что `SI` практически полностью совпадает со значением:
# 
# SI = CC50 / IC50
# 
# Расхождения находятся на уровне машинной погрешности.
# 
# Это означает, что `SI` не является независимой целевой переменной, а рассчитывается на основе двух других таргетов.

# In[206]:


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

sns.scatterplot(
    x=train["IC50, mM"],
    y=train["CC50, mM"],
    ax=axes[0]
)

axes[0].set_title(
    "IC50 vs CC50"
)

sns.scatterplot(
    x=train["IC50, mM"],
    y=train["SI"],
    ax=axes[1]
)

axes[1].set_title(
    "IC50 vs SI"
)

sns.scatterplot(
    x=train["CC50, mM"],
    y=train["SI"],
    ax=axes[2]
)

axes[2].set_title(
    "CC50 vs SI"
)

plt.tight_layout()
plt.show()

# In[207]:


plt.figure(figsize=(6,5))

plt.scatter(
    train["IC50, mM"],
    train["SI"]
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("IC50")
plt.ylabel("SI")

plt.show()

# Связь между целевыми переменными
# 
# Между `IC50` и `CC50` наблюдается умеренная положительная связь.
# 
# Для пары `IC50` и `SI` наблюдается выраженная нелинейная обратная зависимость: при снижении `IC50` значения `SI` резко возрастают.
# 
# После перехода к логарифмическому масштабу структура зависимости становится более выраженной.
# 
# Это соответствует формуле расчёта `SI`, поскольку `IC50` находится в знаменателе.

# In[208]:


for target in TARGETS:

    print("="*80)
    print(target)

    display(
        train[target].describe()
    )

    q1 = train[target].quantile(0.25)
    q3 = train[target].quantile(0.75)

    iqr = q3-q1

    lower = q1-1.5*iqr
    upper = q3+1.5*iqr

    outliers=train[
        (train[target]<lower) |
        (train[target]>upper)
    ]

    print(
        "Количество:",
        outliers.shape[0]
    )

    display(
        train[
            [
                "index",
                "IC50, mM",
                "CC50, mM",
                "SI"
            ]
        ]
        .sort_values(
            target,
            ascending=False
        )
        .head(10)
    )

# In[209]:


train["CC50, mM"]\
.round(3)\
.value_counts()\
.head(25)

# In[210]:


train["IC50, mM"]\
.round(4)\
.value_counts()\
.head(25)

# Выводы по целевым переменным
# 
# Анализ показал, что целевые переменные имеют выраженную асимметрию и содержат экстремальные значения.
# 
# Наиболее нестабильным таргетом оказался `SI`. Медианное значение составляет около `4`, тогда как максимальное превышает `15000`, что создаёт очень длинный правый хвост распределения.
# 
# Проверка подтвердила, что:
# 
# SI = CC50 / IC50
# 
# Поэтому экстремально большие значения `SI` возникают преимущественно при очень малых значениях `IC50`.
# 
# Также были обнаружены повторяющиеся значения в `CC50`, особенно около:
# 
# - 100
# - 250
# - 300
# 
# Это может быть связано с особенностями лабораторных измерений или округлением.
# 
# Дополнительно обнаружены крайне влиятельные наблюдения, которые потенциально могут существенно влиять на дальнейшее обучение моделей.

# In[211]:


# Распределения таргетов

for target in TARGETS:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        train[target],
        bins=50,
        kde=True
    )

    plt.title(
        f"Распределение {target}"
    )

    plt.show()

# In[212]:


# Логарифмированные распределения таргетов

for target in TARGETS:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        np.log1p(
            train[target]
        ),
        bins=50,
        kde=True
    )

    plt.title(
        f"log1p({target})"
    )

    plt.show()

# Анализ распределений таргетов
# 
# Распределения `IC50`, `CC50` и особенно `SI` сильно скошены вправо.
# 
# На сырых значениях видно, что большинство объектов сосредоточено в области небольших значений, а отдельные наблюдения формируют длинный правый хвост.
# 
# После логарифмирования распределения становятся заметно более компактными и стабильными.
# 
# Это важно для дальнейшего моделирования, потому что экстремальные значения могут сильно влиять на ошибку модели.

# In[213]:


# skew до и после log

for target in TARGETS:

    print("="*50)

    print(target)

    print(
        "До log:",
        train[target].skew()
    )

    print(
        "После log:",
        np.log1p(
            train[target]
        ).skew()
    )

    print()

# Выводы по логарифмированию таргетов
# 
# Количественная оценка асимметрии подтвердила, что логарифмирование существенно улучшает распределения целевых переменных.
# 
# Изменение коэффициента асимметрии:
# 
# - `IC50`: `3.79 → -0.06`
# - `CC50`: `2.06 → -0.90`
# - `SI`: `15.63 → 1.54`
# 
# Наиболее сильный эффект наблюдается для `SI`, который до преобразования имел крайне длинный правый хвост и большое количество экстремальных значений.
# 
# После применения `log1p`:
# 
# - распределения стали значительно более компактными;
# - влияние экстремальных значений уменьшилось;
# - асимметрия заметно снизилась;
# - значения приблизились к более стабильной форме распределения.
# 
# Для `IC50` распределение после преобразования стало практически симметричным, а для `CC50` и `SI`, несмотря на сохраняющуюся асимметрию, ситуация существенно улучшилась.
# 
# Это подтверждает целесообразность использования логарифмического преобразования на этапе моделирования.

# ## Анализ констант

# In[214]:


# исключаем таргеты и index

feature_cols = [
    col for col in train.columns
    if col not in TARGETS + ["index"]
]

print("Количество признаков:", len(feature_cols))

# In[215]:


# поиск константных признаков

constant_cols = []

for col in feature_cols:

    if train[col].nunique() <= 1:

        constant_cols.append(col)

print(
    "Количество константных признаков:",
    len(constant_cols)
)

constant_cols

# In[216]:


train[constant_cols].head()

# In[217]:


combined_features = pd.concat(
    [
        train[feature_cols],
        test[feature_cols]
    ],
    axis=0
)

constant_combined=[]

for col in combined_features.columns:

    if combined_features[col].nunique()==1:

        constant_combined.append(col)

print(
    "Константные в train+test:",
    len(constant_combined)
)

constant_combined

# Анализ константных признаков
# 
# В данных были обнаружены признаки, принимающие одно и то же значение для всех объектов.
# 
# Такие признаки не содержат полезной информации, поскольку не помогают различать молекулы и не могут участвовать в объяснении целевых переменных.
# 
# Дополнительная проверка объединённого пространства `train + test` показала, что часть признаков остаётся константной и за пределами обучающей выборки.
# 
# Подобные признаки не несут аналитической ценности и могут быть безопасно удалены на этапе подготовки данных.
# 
# Удаление константных признаков позволяет уменьшить размерность пространства и убрать заведомо бесполезные дескрипторы.

# ## Анализ дубликатов

# In[218]:


# поиск полностью одинаковых признаков

duplicate_cols = []

for i in range(len(feature_cols)):

    col1 = feature_cols[i]

    for j in range(i + 1, len(feature_cols)):

        col2 = feature_cols[j]

        if train[col1].equals(
            train[col2]
        ):

            duplicate_cols.append(
                (col1, col2)
            )

print(
    "Количество пар дублирующихся признаков:",
    len(duplicate_cols)
)

# In[219]:


duplicate_df = pd.DataFrame(
    duplicate_cols,
    columns=[
        "Feature_1",
        "Feature_2"
    ]
)

display(
    duplicate_df
)

# In[220]:


duplicate_df.head(20)

# Анализ дублирующихся признаков
# 
# Дополнительно была проведена проверка признаков на полное совпадение значений.
# 
# В отличие от высокой корреляции, здесь анализируются признаки, которые совпадают полностью для каждого объекта и фактически содержат идентичную информацию.
# 
# Наличие таких признаков не добавляет новых данных и приводит к избыточности признакового пространства.
# 
# Подобные дубликаты могут возникать из-за разных способов расчёта дескрипторов или из-за присутствия нескольких вариантов представления одной и той же химической характеристики.
# 
# Такие признаки могут быть безопасно удалены на этапе подготовки данных.

# ## Повторяющиеся строки по признакам

# In[221]:


feature_only = train.drop(
    columns=TARGETS + ["index"]
)

duplicate_mask = feature_only.duplicated(
    keep=False
)

duplicates = train.loc[
    duplicate_mask
].copy()

print(
    "Количество объектов с повторяющимися признаками:",
    duplicates.shape[0]
)

# In[222]:


display(
    duplicates[
        [
            "index",
            "IC50, mM",
            "CC50, mM",
            "SI"
        ]
    ]
)

# In[223]:


duplicate_groups = (

    feature_only[
        duplicate_mask
    ]

    .groupby(
        list(
            feature_only.columns
        )
    )

    .size()

    .reset_index(
        name="count"
    )

)

print(
    "Количество групп дубликатов:",
    duplicate_groups.shape[0]
)

print(
    "Максимальный размер группы:",
    duplicate_groups["count"].max()
)

# In[224]:


dup_analysis = (

    train.loc[
        duplicate_mask
    ]

    .groupby(
        list(
            feature_only.columns
        )
    )

    .agg({

        "IC50, mM":[
            "nunique",
            "min",
            "max"
        ],

        "CC50, mM":[
            "nunique",
            "min",
            "max"
        ],

        "SI":[
            "nunique",
            "min",
            "max"
        ]

    })

)

dup_analysis.columns=[

    "_".join(col)

    for col in
    dup_analysis.columns

]

dup_analysis = (
    dup_analysis
    .reset_index(
        drop=True
    )
)

display(

    dup_analysis[

        [

        "IC50, mM_nunique",
        "IC50, mM_min",
        "IC50, mM_max",

        "CC50, mM_nunique",
        "CC50, mM_min",
        "CC50, mM_max",

        "SI_nunique",
        "SI_min",
        "SI_max"

        ]

    ]

    .sort_values(

        by=[

            "IC50, mM_nunique",
            "CC50, mM_nunique"

        ],

        ascending=False

    )

    .head(20)

)

# Анализ повторяющихся объектов
# 
# В данных было обнаружено большое количество объектов с полностью одинаковыми признаками.
# 
# Дополнительный анализ показал, что одинаковым наборам дескрипторов могут соответствовать разные значения целевых переменных.
# 
# Для некоторых групп наблюдается существенный разброс:
# 
# - `IC50` изменяется от очень малых до десятков единиц;
# - `CC50` отличается на порядки;
# - `SI` может меняться от единиц до экстремально больших значений.
# 
# Это означает, что одинаковые наборы признаков не всегда однозначно определяют биологический ответ.
# 
# Возможные причины:
# 
# - повторные экспериментальные измерения одного соединения;
# - шум биологических измерений;
# - ограниченность набора дескрипторов;
# - разные молекулы с одинаковыми агрегированными признаками.
# 
# Это важное свойство датасета, поскольку оно создаёт естественное ограничение качества модели: даже при одинаковых признаках целевая переменная может существенно различаться.

# In[225]:


# Корреляции между признаками

feature_train = train.drop(
    columns=TARGETS + ["index"]
).copy()

corr_matrix = (
    feature_train
    .corr()
    .abs()
)

# In[226]:


# Поиск сильно коррелирующих признаков

upper = corr_matrix.where(

    np.triu(
        np.ones(
            corr_matrix.shape
        ),
        k=1
    ).astype(bool)

)

high_corr_pairs = []

for col in upper.columns:

    highly_corr = upper[col][
        upper[col] > 0.995
    ]

    for idx, corr_val in highly_corr.items():

        high_corr_pairs.append(
            (
                idx,
                col,
                corr_val
            )
        )

high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=[
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]
)

print(
    "Количество сильно коррелирующих пар:",
    len(high_corr_df)
)

display(
    high_corr_df
    .sort_values(
        "Correlation",
        ascending=False
    )
)

# In[227]:


# разброс таргетов внутри одинаковых X

target_variability = (

    train.loc[
        duplicate_mask
    ]

    .groupby(
        list(
            feature_only.columns
        )
    )

    [

        TARGETS

    ]

    .std()

)

display(
    target_variability
    .sort_values(
        by="SI",
        ascending=False
    )

    .head(15)

)

# Анализ корреляций между признаками
# 
# Проверка показала, что среди молекулярных дескрипторов есть признаки с очень высокой корреляцией.
# 
# Это ожидаемо для химических данных, потому что разные дескрипторы могут описывать близкие свойства молекулы: массу, размер, количество атомов, топологию или функциональные группы.
# 
# Такие признаки не являются ошибкой, но указывают на избыточность признакового пространства.
# 
# На этапе подготовки данных часть полностью дублирующих или почти одинаковых признаков можно рассматривать как кандидатов на удаление.
# 
# Даже внутри полностью одинаковых наборов признаков наблюдается заметная вариативность таргетов.
# 
# Это дополнительно подтверждает наличие экспериментального шума и ограничений используемых дескрипторов.

# In[228]:


# Анализ таргетов у объектов с одинаковыми признаками

dup_analysis = (

    train.loc[
        duplicate_mask
    ]

    .groupby(
        list(
            feature_only.columns
        )
    )

    .agg({

        "IC50, mM": [
            "nunique",
            "min",
            "max"
        ],

        "CC50, mM": [
            "nunique",
            "min",
            "max"
        ],

        "SI": [
            "nunique",
            "min",
            "max"
        ]

    })

)

dup_analysis.columns = [
    "_".join(col)
    for col in dup_analysis.columns
]

dup_analysis = (
    dup_analysis
    .reset_index(
        drop=True
    )
)

display(

    dup_analysis[

        [

            "IC50, mM_nunique",
            "IC50, mM_min",
            "IC50, mM_max",

            "CC50, mM_nunique",
            "CC50, mM_min",
            "CC50, mM_max",

            "SI_nunique",
            "SI_min",
            "SI_max"

        ]

    ]

    .sort_values(

        by=[
            "IC50, mM_nunique",
            "CC50, mM_nunique"
        ],

        ascending=False

    )

    .head(20)

)

# Дубликаты с разными таргетами
# 
# В данных обнаружены объекты с полностью одинаковыми признаками, но разными значениями целевых переменных.
# 
# Это одна из самых важных находок EDA.
# 
# Такая ситуация означает, что одинаковый набор молекулярных дескрипторов не всегда однозначно определяет `IC50`, `CC50` и `SI`.
# 
# Возможные причины:
# 
# - повторные экспериментальные измерения;
# - шум биологических данных;
# - ограниченность дескрипторов;
# - разные соединения с одинаковыми агрегированными признаками.
# 
# Это создаёт естественное ограничение качества будущих моделей: даже при одинаковом `X` целевой ответ `y` может отличаться.

# ## Анализ корреляций между целевыми переменными

# In[229]:


# Корреляции между таргетами

target_corr = (
    train[TARGETS]
    .corr()
)

display(
    target_corr
)

# In[230]:


# Связи между таргетами

sns.pairplot(

    train[
        TARGETS
    ]

)

plt.show()

# In[231]:


# IC50 vs SI

plt.figure(
    figsize=(6,5)
)

plt.scatter(

    train[
        "IC50, mM"
    ],

    train[
        "SI"
    ]

)

plt.xscale(
    "log"
)

plt.yscale(
    "log"
)

plt.xlabel(
    "IC50"
)

plt.ylabel(
    "SI"
)

plt.show()

# Выводы по корреляциям таргетов
# 
# Анализ подтвердил наличие зависимости между целевыми переменными.
# 
# Наблюдаются следующие закономерности:
# 
# - между `IC50` и `SI` присутствует выраженная обратная зависимость;
# - между `CC50` и `SI` связь имеет более сложный нелинейный характер;
# - `IC50` и `CC50` демонстрируют умеренную положительную связь.
# 
# После перехода к логарифмическому масштабу структура зависимостей становится более выраженной.
# 
# Также видно, что экстремально высокие значения `SI` возникают преимущественно при очень малых значениях `IC50`, что согласуется с формулой:
# 
# SI = CC50 / IC50

# ## Корреляции признаков

# Анализ корреляции между признаками
# 
# Дополнительно проверим признаки на избыточность.
# 
# Для молекулярных дескрипторов часто встречаются признаки, которые практически дублируют друг друга и содержат очень похожую информацию.
# 
# На данном этапе интерес представляют признаки с почти полной корреляцией, поскольку они могут указывать на повторяющиеся химические характеристики или разные варианты расчёта одного и того же свойства.

# In[232]:


feature_train = train.drop(
    columns=TARGETS + ["index"]
).copy()

corr_matrix = (
    feature_train
    .corr()
    .abs()
)

upper = corr_matrix.where(

    np.triu(
        np.ones(
            corr_matrix.shape
        ),
        k=1
    ).astype(bool)

)

high_corr_pairs=[]

for col in upper.columns:

    highly_corr = upper[col][
        upper[col] > 0.995
    ]

    for idx, corr_val in highly_corr.items():

        high_corr_pairs.append(

            (
                idx,
                col,
                corr_val
            )

        )

high_corr_df = pd.DataFrame(

    high_corr_pairs,

    columns=[
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]

)

print(
    "Количество сильно коррелирующих пар:",
    len(high_corr_df)
)

# In[233]:


display(

    high_corr_df
    .sort_values(
        "Correlation",
        ascending=False
    )

)

# Выводы по корреляции признаков
# 
# Проверка показала, что сильная избыточность признаков присутствует, но в ограниченном масштабе.
# 
# Было найдено 11 пар признаков с очень высокой корреляцией (>0.995), часть из которых фактически дублирует одну и ту же информацию.
# 
# Наиболее показательные примеры:
# 
# - `MaxAbsEStateIndex` и `MaxEStateIndex`
# - `MolWt` и `ExactMolWt`
# - `fr_COO` и `fr_COO2`
# 
# Также были обнаружены признаки с очень высокой, но не полной корреляцией:
# 
# - `MolWt` и `HeavyAtomMolWt`
# - `Chi0 / Chi1` и `HeavyAtomCount`
# 
# В целом набор признаков не выглядит чрезмерно раздутым, однако несколько явно дублирующих дескрипторов присутствуют.
# 
# Такие признаки могут быть кандидатами на удаление на этапе подготовки данных.

# ## Корреляций между признаками и целями

# Анализ связи признаков с целевыми переменными
# 
# Дополнительно проверим, насколько отдельные признаки связаны с целевыми переменными.
# 
# Это позволяет понять:
# 
# - существуют ли особенно информативные дескрипторы;
# - какие типы признаков сильнее связаны с активностью и токсичностью;
# - отсутствуют ли признаки с подозрительно высокой связью, которые могут указывать на утечку информации.

# In[234]:


feature_train = train.drop(
    columns=TARGETS + ["index"]
).copy()

target_corr_results = {}

for target in TARGETS:

    corr = (

        feature_train
        .corrwith(
            train[target]
        )

        .abs()

        .sort_values(
            ascending=False
        )

    )

    target_corr_results[target]=(
        corr.head(15)
    )

for target, corr_series in target_corr_results.items():

    print("="*80)

    print(target)

    print("="*80)

    display(
        corr_series
    )

# In[235]:


for target, corr_series in target_corr_results.items():

    plt.figure(figsize=(8,5))

    corr_series.sort_values().plot.barh()

    plt.title(
        f"Top correlations with {target}"
    )

    plt.show()

# Выводы по связи признаков и таргетов
# 
# Проверка не выявила признаков с подозрительно высокой корреляцией с целевыми переменными.
# 
# Максимальные значения корреляции составили:
# 
# - для `IC50`: около `0.25`
# - для `CC50`: около `0.31`
# - для `SI`: около `0.19`
# 
# Это говорит о том, что явной утечки информации в признаках не наблюдается, а целевые переменные не определяются одним отдельным дескриптором.
# 
# Также видно, что разные таргеты сильнее связаны с различными типами признаков:
# 
# - `IC50` чаще связан с топологическими и lipophilicity-дескрипторами;
# - `CC50` сильнее связан с признаками размера и структурной сложности;
# - `SI` не демонстрирует выраженной линейной зависимости от отдельных признаков.
# 
# Это хороший признак: модель, вероятно, будет использовать комбинации признаков, а не зависеть от одного доминирующего дескриптора.

# ## Обнаружение аномалий и Isolation Forest

# Анализ аномальных объектов
# 
# После анализа отдельных признаков дополнительно проверим сами объекты.
# 
# Важно понять, существуют ли молекулы, которые существенно отличаются от остальных по совокупности признаков.
# 
# Такие объекты не обязательно являются ошибками данных.
# 
# Они могут представлять:
# 
# - редкие химические структуры;
# - соединения с необычной биологической активностью;
# - потенциально сильно влияющие наблюдения.

# In[236]:


from sklearn.ensemble import IsolationForest

X_clean = train.drop(
    columns=TARGETS + ["index"]
).copy()

X_clean = X_clean.fillna(
    X_clean.median()
)

iso = IsolationForest(

    contamination=0.03,

    random_state=42

)

outlier_labels = (
    iso.fit_predict(
        X_clean
    )
)

train["is_outlier"]=(
    outlier_labels==-1
)

print(

    "Количество аномальных объектов:",

    train[
        "is_outlier"
    ].sum()

)

display(

    train.loc[

        train[
            "is_outlier"
        ],

        [

            "index",

            "IC50, mM",

            "CC50, mM",

            "SI"

        ]

    ]

    .sort_values(
        "SI",
        ascending=False
    )

)

# In[237]:


outlier_pct = (

    train[
        "is_outlier"
    ].mean()

    *100

)

print(

    f"Доля аномалий: {outlier_pct:.2f}%"

)

# Выводы по аномальным объектам
# 
# С помощью Isolation Forest было обнаружено 22 объекта, что составляет около 3% выборки.
# 
# Анализ показал, что большинство обнаруженных объектов не выглядят как ошибки данных или некорректные расчёты дескрипторов.
# 
# Среди найденных объектов чаще встречаются:
# 
# - соединения с очень высоким `SI`;
# - объекты с крайне низкими значениями `IC50`;
# - соединения с необычно высоким `CC50`.
# 
# Таким образом, алгоритм в основном выделяет редкие, но биологически правдоподобные соединения, а не явно повреждённые наблюдения.
# 
# На текущем этапе удаление таких объектов не выглядит оправданным.

# ## Train vs Test Drift

# Анализ различий между train и test
# 
# Перед построением моделей важно проверить, насколько обучающая и тестовая выборки принадлежат одному пространству данных.
# 
# Если распределения признаков существенно различаются, модель может столкнуться с проблемой domain shift и хуже работать на тестовых данных.
# 
# Для проверки используем:
# 
# - KS-test;
# - adversarial validation;
# - PCA-визуализацию;
# - UMAP-проекцию.

# ### KS тест

# In[238]:


drift_results=[]

for col in feature_cols:

    stat, p_value = ks_2samp(

        train[col],

        test[col]

    )

    drift_results.append({

        "feature":col,

        "ks_stat":stat,

        "p_value":p_value

    })

drift_df=pd.DataFrame(
    drift_results
)

drift_df=(
    drift_df
    .sort_values(
        "ks_stat",
        ascending=False
    )
)

display(
    drift_df.head(20)
)

# Наибольшие отличия между train и test наблюдаются для признаков, связанных с размером и структурой молекул.
# 
# При этом значения KS-статистики остаются умеренными, что не указывает на критическое расхождение распределений.

# ### Adversarial validation:

# In[239]:


train_adv = train.copy()
test_adv = test.copy()

train_adv["is_test"]=0
test_adv["is_test"]=1

combined = pd.concat(

    [
        train_adv,
        test_adv
    ],

    axis=0

)

X_adv = combined.drop(

    columns=
    ["is_test"]+
    TARGETS

)

y_adv = combined[
    "is_test"
]

clf = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

scores = cross_val_score(

    clf,

    X_adv,

    y_adv,

    cv=5,

    scoring="roc_auc"

)

print(
    "ROC-AUC:",
    scores
)

print(
    "Mean:",
    scores.mean()
)

# Средний ROC-AUC составил около `0.67`.
# 
# Это говорит о наличии умеренного различия между train и test.
# 
# Однако значение далеко от `1`, поэтому модель не может уверенно отделить одну выборку от другой.

# ### PCA

# In[240]:


train_pca = train[
    feature_cols
].copy()

test_pca = test[
    feature_cols
].copy()

train_pca[
    "dataset"
]="train"

test_pca[
    "dataset"
]="test"

combined_pca = pd.concat(

    [
        train_pca,
        test_pca
    ],

    axis=0

)

X = combined_pca.drop(
    columns=["dataset"]
)

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = (
    imputer
    .fit_transform(X)
)

scaler = StandardScaler()

X_scaled = (
    scaler
    .fit_transform(
        X_imputed
    )
)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = (
    pca
    .fit_transform(
        X_scaled
    )
)

pca_df = pd.DataFrame({

    "PC1":
    X_pca[:,0],

    "PC2":
    X_pca[:,1],

    "dataset":
    combined_pca[
        "dataset"
    ].values

})

# In[241]:


plt.figure(
    figsize=(8,6)
)

sns.scatterplot(

    data=pca_df,

    x="PC1",

    y="PC2",

    hue="dataset",

    alpha=0.7

)

plt.show()

# ### UMAP

# In[242]:


umap_model=umap.UMAP(

    n_components=2,

    n_neighbors=15,

    min_dist=0.1,

    metric="euclidean",

    random_state=42

)

X_umap=(
    umap_model
    .fit_transform(
        X_scaled
    )
)

umap_df = pd.DataFrame({

    "UMAP1":
    X_umap[:,0],

    "UMAP2":
    X_umap[:,1],

    "dataset":
    combined_pca[
        "dataset"
    ].values

})

# In[243]:


plt.figure(
    figsize=(8,6)
)

sns.scatterplot(

    data=umap_df,

    x="UMAP1",

    y="UMAP2",

    hue="dataset",

    alpha=0.7

)

plt.show()

# Выводы по train/test различиям
# 
# Проверка несколькими методами показала наличие умеренного различия между train и test.
# 
# KS-test выявил отдельные признаки с отличающимися распределениями.
# 
# Adversarial validation показал ROC-AUC около `0.67`, что указывает на существование некоторого смещения между выборками.
# 
# При этом PCA и UMAP показали, что train и test в целом остаются хорошо перемешанными и не образуют изолированных областей.
# 
# Таким образом, критического domain shift не наблюдается, однако небольшое смещение распределений присутствует и его стоит учитывать при дальнейшем моделировании.

# ## Дополнительная кластеризация

# Кластеризация объектов (дополнительный анализ)
# 
# Дополнительно рассмотрим возможность существования естественных групп соединений в пространстве молекулярных дескрипторов.
# 
# Кластеризация используется исключительно как исследовательский инструмент и не рассматривается как основная стратегия построения модели.
# 
# Цель анализа:
# 
# - проверить наличие внутренних структур;
# - посмотреть, различаются ли биологические свойства между группами;
# - оценить потенциальную неоднородность данных.

# In[244]:


X_cluster = train.drop(
    columns=TARGETS + ["index"]
)

imputer = SimpleImputer(
    strategy="median"
)

X_cluster = imputer.fit_transform(
    X_cluster
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)

# In[245]:


scores=[]

K=range(3,11)

for k in K:

    kmeans=KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    labels=(
        kmeans
        .fit_predict(
            X_scaled
        )
    )

    score=(
        silhouette_score(
            X_scaled,
            labels
        )
    )

    scores.append(score)

plt.figure(
    figsize=(8,5)
)

plt.plot(
    K,
    scores
)

plt.xlabel(
    "Количество кластеров"
)

plt.ylabel(
    "Silhouette"
)

plt.show()

# In[246]:


best_k = K[
    np.argmax(scores)
]

print(
    "Лучшее число кластеров:",
    best_k
)

# In[247]:


kmeans = KMeans(

    n_clusters=best_k,

    random_state=42,

    n_init=10

)

train["cluster"]=(

    kmeans.fit_predict(
        X_scaled
    )

)

# In[248]:


cluster_targets = (

    train

    .groupby(
        "cluster"
    )[TARGETS]

    .median()

)

display(
    cluster_targets
)

# In[249]:


pca_vis = PCA(
    n_components=2,
    random_state=42
)

X_vis=(
    pca_vis
    .fit_transform(
        X_scaled
    )
)

cluster_df = pd.DataFrame({

    "PC1":
    X_vis[:,0],

    "PC2":
    X_vis[:,1],

    "cluster":
    train[
        "cluster"
    ]

})

plt.figure(
    figsize=(8,6)
)

sns.scatterplot(

    data=cluster_df,

    x="PC1",

    y="PC2",

    hue="cluster"

)

plt.show()

# Выводы по кластеризации
# 
# Кластеризация использовалась как дополнительный исследовательский инструмент.
# 
# Анализ показал, что данные не образуют чётко разделённых компактных групп: значения silhouette остаются относительно низкими.
# 
# Это может объясняться:
# 
# - высокой размерностью пространства дескрипторов;
# - шумом в биологических измерениях;
# - наличием одинаковых признаков при разных таргетах;
# - сложной структурой химического пространства.
# 
# При этом отдельные различия между кластерами по медианным значениям целевых переменных всё же наблюдаются.
# 
# Таким образом, кластеризация может использоваться как инструмент интерпретации данных, однако не выглядит основной стратегией для решения задачи прогнозирования.

# ## PCA / UMAP визуализация химического пространства

# Дополнительно визуализируем молекулы в двумерном пространстве.
# 
# Высокоразмерное пространство дескрипторов трудно анализировать напрямую, поэтому используем методы снижения размерности.
# 
# Цель анализа:
# 
# - проверить наличие групп молекул;
# - оценить распределение активных соединений;
# - посмотреть, образуют ли значения таргетов отдельные области.

# In[250]:


# признаки молекул

X_vis = train.drop(
    columns=TARGETS + ["index"]
).copy()

# пропуски

X_vis = X_vis.fillna(
    X_vis.median()
)

# масштабирование

X_scaled = scaler.fit_transform(
    X_vis
)

# In[251]:


# PCA

X_pca = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({

    "PC1":X_pca[:,0],

    "PC2":X_pca[:,1],

    "SI":train["SI"]

})

# In[252]:


# PCA + SI

plt.figure(
    figsize=(8,6)
)

plt.scatter(

    pca_df["PC1"],

    pca_df["PC2"],

    c=np.log1p(
        pca_df["SI"]
    )

)

plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.colorbar(
    label="log(SI)"
)

plt.show()

# In[253]:


# UMAP

X_umap = umap_model.fit_transform(
    X_scaled
)

umap_df = pd.DataFrame({

    "UMAP1":X_umap[:,0],

    "UMAP2":X_umap[:,1],

    "SI":train["SI"]

})

# In[254]:


# UMAP + SI

plt.figure(
    figsize=(8,6)
)

plt.scatter(

    umap_df["UMAP1"],

    umap_df["UMAP2"],

    c=np.log1p(
        umap_df["SI"]
    )

)

plt.colorbar(
    label="log(SI)"
)

plt.show()

# Визуализация химического пространства показала, что молекулы не образуют чётко разделённых компактных групп.
# 
# На PCA-проекции объекты формируют несколько пересекающихся областей без выраженных границ между ними. Аналогичная картина наблюдается и на UMAP: пространство остаётся непрерывным, а изолированные кластеры практически отсутствуют.
# 
# При окрашивании по `SI` видно, что соединения с высокими значениями активности распределены среди остальных объектов и не формируют отдельный компактный кластер.
# 
# Также наблюдаются отдельные удалённые точки и небольшие группы объектов, которые могут соответствовать редким или необычным соединениям. Это согласуется с ранее найденными аномальными объектами.
# 
# Полученные результаты указывают на сложную структуру химического пространства: биологическая активность, вероятно, определяется комбинацией нескольких признаков, а не принадлежностью молекул к простым и хорошо разделимым группам.

# ## Выводы

# In[255]:


# EDA -> modelling

eda_summary = pd.DataFrame({

"Наблюдение":[

"Сильная асимметрия таргетов",

"Есть одинаковые X с разными Y",

"Найдены редкие объекты",

"Обнаружено умеренное смещение train/test",

"Есть коррелирующие признаки"

],

"Действие":[

"log1p для таргетов",

"учитывать шум таргетов",

"не удалять автоматически",

"использовать robust CV",

"CatBoost без агрессивного удаления"

]

})

display(
    eda_summary
)

# Итоговые выводы EDA
# 
# В ходе анализа данных была проведена проверка структуры датасета, целевых переменных, признаков и различий между обучающей и тестовой выборками.
# 
# Основные результаты:
# 
# Структура данных
# - Датасет содержит молекулярные дескрипторы и три целевые переменные: `IC50`, `CC50`, `SI`;
# - пропуски практически отсутствуют и встречаются только у небольшого числа объектов;
# - пропуски имеют структурный характер и связаны с невозможностью расчёта отдельных дескрипторов.
# 
# Целевые переменные
# - распределения имеют выраженную асимметрию и длинные правые хвосты;
# - `SI` содержит экстремально большие значения;
# - подтверждено, что:
# 
# SI = CC50 / IC50
# 
# то есть `SI` не является независимой целевой переменной.
# 
# Анализ признаков
# - были обнаружены константные признаки;
# - найдены полностью дублирующиеся дескрипторы;
# - сильная избыточность признаков выражена умеренно;
# - обнаружено только 11 пар признаков с очень высокой корреляцией.
# 
# Повторяющиеся объекты
# Было найдено большое количество объектов с одинаковыми признаками.
# 
# При этом одинаковым наборам дескрипторов могут соответствовать разные значения таргетов.
# 
# Это указывает на:
# 
# - возможный шум измерений;
# - повторные эксперименты;
# - ограничения используемого набора признаков.
# 
# Данное свойство создаёт естественное ограничение качества будущих моделей.
# 
# Аномальные объекты
# Isolation Forest выявил редкие объекты, однако большинство из них выглядят биологически правдоподобными и не похожи на ошибки данных.
# 
# Удаление таких наблюдений на данном этапе не выглядит оправданным.
# 
# Train/Test различия
# Проверка несколькими методами показала наличие умеренного смещения между train и test.
# 
# Однако критического domain shift обнаружено не было.
# 
# PCA и UMAP показали, что выборки в целом принадлежат одному пространству.
# 
# Кластеризация
# Кластеризация не выявила чётко разделённых групп объектов.
# 
# Низкие значения silhouette показывают, что химическое пространство имеет сложную структуру и плохо разделяется на компактные кластеры.
# 
# ---
# 
# Общий вывод
# 
# Датасет выглядит достаточно качественным и не содержит критических проблем.
# 
# При этом были обнаружены важные особенности:
# 
# - шум в таргетах;
# - одинаковые признаки при разных ответах;
# - умеренное различие train/test;
# - наличие редких биологически необычных соединений.
# 
# Эти особенности необходимо учитывать на этапе построения моделей.

# # Дальнейшая обработка и направления развития
# 
# Проведённый EDA позволил выявить особенности данных и определить возможные направления дальнейшей обработки и улучшения моделей.
# 
# ## 1. Работа с целевыми переменными
# 
# Анализ показал сильную асимметрию распределений:
# 
# - `IC50`: 3.79 → -0.06 после log;
# - `CC50`: 2.06 → -0.90;
# - `SI`: 15.63 → 1.54.
# 
# Поэтому для моделирования целесообразно использовать логарифмическое преобразование целевых переменных (`log1p`), что уменьшает влияние экстремальных значений.
# 
# ---
# 
# ## 2. Работа с одинаковыми X и разными Y
# 
# Были обнаружены объекты с одинаковыми дескрипторами и различающимися таргетами.
# 
# Это может указывать на:
# 
# - шум биологических измерений;
# - экспериментальную вариативность;
# - ограничения набора дескрипторов.
# 
# При обучении моделей это свойство следует учитывать:
# 
# - использовать устойчивую к шуму кросс-валидацию;
# - избегать агрессивной очистки данных;
# - рассматривать ансамблевые методы.
# 
# ---
# 
# ## 3. Работа с признаками
# 
# Были обнаружены:
# 
# - константные признаки;
# - дублирующиеся признаки;
# - группы сильно коррелирующих дескрипторов.
# 
# Возможные варианты обработки:
# 
# - удаление полностью константных признаков;
# - удаление идентичных признаков;
# - сравнение моделей с исходным и очищенным пространством признаков.
# 
# При использовании CatBoost агрессивное удаление коррелирующих признаков может быть необязательным.
# 
# ---
# 
# ## 4. Возможное извлечение дополнительных признаков
# 
# На текущем этапе используются готовые молекулярные дескрипторы.
# 
# Дополнительно можно рассмотреть:
# 
# - генерацию новых RDKit-дескрипторов;
# - взаимодействия признаков;
# - агрегированные отношения между химическими свойствами;
# - снижение размерности (PCA-компоненты как признаки).
# 
# ---
# 
# ## 5. Работа с train/test различиями
# 
# Adversarial validation показал наличие умеренного смещения между train и test.
# 
# В дальнейшем можно:
# 
# - использовать robust CV;
# - анализировать важность признаков в adversarial-модели;
# - учитывать возможный domain shift.
# 
# ---
# 
# ## 6. План дальнейшего моделирования
# 
# На основе результатов EDA дальнейший пайплайн может включать:
# 
# 1. логарифмирование таргетов;
# 2. удаление константных признаков;
# 3. заполнение пропусков;
# 4. обучение baseline-моделей;
# 5. CatBoost / LightGBM / ансамбли;
# 6. анализ важности признаков;
# 7. интерпретацию моделей.
