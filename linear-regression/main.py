import pandas as pd
import numpy as np
from Linear_Regression import Linear_Regression

# DATA
df = pd.read_csv("dataset/train.csv")
xs = df['x']
ys = df['y']

# MODEL
lr = Linear_Regression(xs, ys)

# TEST
x = 77
y = lr.predict(x)

print("Prediction: x=" + str(x) + ", y=" + str(y))
print("Slope=" + str(lr.m) + ", y intercept=" + str(lr.c))
print("R squared=" + str(lr.r2))

# VISUALISE
lr.visualise(x, y)