from statistics import mean
import matplotlib.pyplot as plt

class Linear_Regression:
  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys
    self.m = self.get_m(self.xs, self.ys)
    self.c = self.get_c(self.xs, self.ys, self.m)
    self.bfl = self.get_bfl(self.m, self.c, self.xs)
    self.r2 = self.get_r2(self.bfl, self.ys)

  def get_m(self, xs, ys):
    # m = slope
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs)**2 - mean(xs * xs))
    return m

  def get_c(self, xs, ys, m):
    # c = y intercept
    c = mean(ys) - (m * mean(xs))
    return c

  def get_bfl(self, m, c, xs):
    # bfl = best fit line
    # equation of a line: y = mx + c
    bfl = [(m * x) + c for x in xs]
    return bfl

  def get_r2(self, bfl, ys):
    # r squared, r^2 = coefficient of determination
    SSy_hat = sum([y * y for y in (bfl - ys)])
    SSy_mean = sum([y * y for y in [y - mean(ys) for y in ys]])
    r2 = 1 - (SSy_hat / SSy_mean)
    return r2

  def predict(self, x):
    y = (self.m * x) + self.c
    return y

  def visualise(self, x, y):
    # plot a graph with matplotlib
    plt.scatter(self.xs, self.ys)
    plt.plot(self.xs, self.bfl, color='g')
    plt.scatter(x, y, color='r')
    plt.show()