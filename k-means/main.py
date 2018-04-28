import numpy as np
import pandas as pd
from K_Means import K_Means

df = pd.read_csv("dataset/titanic.csv", index_col=0)

# 1. SHUFFLE DATA
df = df.sample(frac=1).reset_index(drop=True)

# print(df.head(15))

# 2. PREP DATA
# make NaN outliers
NaN_replacement = -999999
df = df.fillna(NaN_replacement)

# replace strings with discrete values in df
def strs_to_nums(column_name):
    # get unique column values from a given column
    unique_col_vals = df[column_name].unique()
    # remove NaN if any
    if NaN_replacement in unique_col_vals:
        unique_col_vals = np.delete(
            unique_col_vals, list(unique_col_vals).index(NaN_replacement)
        )
    # convert to dictionary
    # e.g {"column value 1": 0, "column value 2": 1, ...}
    unique_col_vals_dict = {v:k for k,v in enumerate(unique_col_vals)}
    # update df
    df.replace({column_name: unique_col_vals_dict}, inplace=True)

strs_to_nums('cabin')
strs_to_nums('sex')
strs_to_nums('embarked')
strs_to_nums('home.dest')
strs_to_nums('ticket')
strs_to_nums('boat')

# drop name column
df = df.drop('name', axis=1)

y = df['survived'].tolist()
X = df.drop('survived', axis=1).values.tolist()

# 3. TRAIN MODEL
clf = K_Means(k=2)
clf.train(X)
clf.test(X, y)