import pandas as pd
import random
from Naive_Bayes import Naive_Bayes

# DATA
df = pd.read_csv("dataset/Iris.csv")
df.drop(['Id'], 1, inplace=True)
data_set = df.values.tolist()

# shuffle data set
random.shuffle(data_set)

# divide set in to training and test data
train_data = pd.DataFrame(data_set[:120])
test_data = pd.DataFrame(data_set[120:])

# CLASSIFIER
nb = Naive_Bayes(train_data)
nb.test(test_data)