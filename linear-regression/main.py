import pandas as pd
import numpy as np
from Linear_Regression import Linear_Regression

# df = pd.read_csv("datasets/train.csv")
# df.drop([''], 1, inplace=True)
# print(df)

# GET (TRAINING) DATA
with open("datasets/train.csv", 'r') as tf:
  xs, ys = [], []
  data = tf.readlines()
  for i in range(1, len(data)):
    d = data[i].rstrip('\n')
    _x, _y = [float(i) for i in d.split(',')]
    xs.append(_x)
    ys.append(_y)
  xs = np.array(xs)
  ys = np.array(ys)

lr = Linear_Regression(xs, ys)

# TEST
x = 77
y = lr.predict(x)

print("Prediction: x=" + str(x) + ", y=" + str(y))
print("Slope=" + str(lr.m) + ", y intercept=" + str(lr.c))
print("R squared=" + str(lr.r2))

# VISUALISE
lr.visualise(x, y)