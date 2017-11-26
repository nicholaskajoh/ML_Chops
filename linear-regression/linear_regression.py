import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


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
  

# BUILD MODEL
# equation of a line: y = mx + c
# m = slope
m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs)**2 - mean(xs * xs))
# c = y intercept
c = mean(ys) - (m * mean(xs))
print("slope: " + str(m) + ", y intercept: " + str(c))
# best fit line
bf_line = [(m * x) + c for x in xs]


# MODEL ACCURACY
# coefficient of determination, r squared
SSy_hat = sum([y * y for y in (bf_line - ys)])
SSy_mean = sum([y * y for y in [y - mean(ys) for y in ys]])
r2 = 1 - (SSy_hat / SSy_mean)
print("r squared: " + str(r2))


# TEST MODEL
in_x = 77
out_y = (m * in_x) + c
print("PREDICTION => x: " + str(in_x) + ", y: " + str(out_y))


# VISUALISE DATA
plt.scatter(xs, ys)
plt.plot(xs, bf_line, color='g')
plt.scatter(in_x, out_y, color='r')
plt.show()