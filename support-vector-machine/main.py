import numpy as np
import pandas as pd
import random
from Support_Vector_Machine import Support_Vector_Machine

# TOY DATA
# use this to test out the visualize method
# features = np.array([[5, 4], [5, -1], [3, 3], [7, 9], [6, 7], [7, 11]])
# labels = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# clf = Support_Vector_Machine()
# clf.fit(features, labels)
# xi = [3, 4]
# prediction = clf.predict(xi)
# clf.visualize(features, labels, xi, prediction)

# DATA
df = pd.read_csv("dataset/data.csv", index_col=0)
df.drop(['Unnamed: 32'], 1, inplace=True) # remove last column containing NaNs
num_samples = df.shape[0]
data = df.values.tolist()

# shuffle data
random.shuffle(data)

# replace labels B with 1 and M with -1
for i in range(len(data)):
  if data[i][0] == 'B':
    data[i][0] = 1
  elif data[i][0] == 'M':
    data[i][0] = -1

data = np.array(data)

y = data[:, 0]
X = np.delete(data, 0, axis=1)

# divide samples in to training and test data
# 70% for training and 30% for testing
n = int(round(7/10 * num_samples))
X_train, y_train = X[:n], y[:n]
X_test, y_test = X[n:], y[n:]

# CLASSIFIER
clf = Support_Vector_Machine()
clf.fit(X_train, y_train)
clf.test(X_test, y_test)