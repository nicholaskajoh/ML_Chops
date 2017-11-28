import pandas as pd
import random
from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN

# DATA
df = pd.read_csv("dataset/Iris.csv")
df.drop(['Id'], 1, inplace=True)
data_set = df.values.tolist()

# shuffle data set
random.shuffle(data_set)

# divide set in to training and test data
train_data = {"Iris-setosa": [], "Iris-versicolor": [], "Iris-virginica": []}
test_data = {"Iris-setosa": [], "Iris-versicolor": [], "Iris-virginica": []}
for i in data_set[:120]:
  train_data[i[-1]].append(i[:-1])
for i in data_set[120:]:
  test_data[i[-1]].append(i[:-1])

# MODEL
knn = KNN(train_data, k=7)
knn.test(test_data)