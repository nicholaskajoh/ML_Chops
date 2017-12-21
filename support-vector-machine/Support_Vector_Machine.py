import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers
from matplotlib import style

style.use('ggplot')

class Support_Vector_Machine:
  def fit(self, X, y):
    # The goal in training an SVM is to maximize the distance between
    # the two parallel hyperplanes that separate the two classes of data.
    # The geometric distance between these two hyperplanes is 2 / ||w||
    # where ||w|| is the magnitude of w.
    # w is the normal vector to the hyperplanes.
    # The constraint for the optimization is yi(xi.w + b) >= 1
    # This constraint ensures that each data point must lie on the correct side of the margin.
    # b is the bias or shift.
    # Maximizing 2 / ||w|| implies minimizing ||w||
    # For mathematical convenience, we minimize 1/2 * ||w||^2
    # This is a Quadratic programming problem.
    # We'll use Convex Optimization to solve it with the help of the CVXOPT Python library.
    # The code below is a slight modification of Mathieu Blondel's SVM's fit method.
    # Link: https://gist.github.com/mblondel/586753

    n_samples, n_features = X.shape

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
      for j in range(n_samples):
        K[i,j] = np.dot(X[i], X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h = cvxopt.matrix(np.zeros(n_samples))

    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    a = np.ravel(solution['x'])

    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    self.a = a[sv]
    self.sv = X[sv]
    self.sv_y = y[sv]

    # Intercept
    self.b = 0
    for n in range(len(self.a)):
      self.b += self.sv_y[n]
      self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
    self.b /= len(self.a)

    # Weight vector
    self.w = np.zeros(n_features)
    for n in range(len(self.a)):
      self.w += self.a[n] * self.sv_y[n] * self.sv[n]

  def predict(self, xi):
    # The decision boundary is given by xi.w + b = 0
    # we return the sign of xi.w + b i.e positive or negative
    classification = np.sign(np.dot(xi, self.w) + self.b)
    return classification

  def test(self, X, y):
    correct = 0
    total = X.shape[0]
    for i in range(len(X)):
      if self.predict(X[i]) == y[i]:
        correct += 1
    accuracy = correct / total
    print("Accuracy:", accuracy)

  def visualize(self, X, y, xi, prediction):
    # To simplify things, we'll only visualize data with 2 features.
    # 'xi' is a feature set the SVM classified and
    # 'prediction' is it's prediction (-1 or 1)
    # We plot xi with the training data (X) to see if the SVM classified it correctly,
    # judging from a visual point of view

    # Plot data
    n_features = X.shape[1]
    if n_features == 2:
      colors = {-1: 'r', 1: 'b'}
      for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], c=colors[y[i]])
      # plot xi
      plt.scatter(xi[0], xi[1], c=colors[prediction], marker='*')

      # Plot hyperplanes
      def hyperplane(x, w, b, v):
        return (-w[0] * x - b + v) / w[1]
      
      xs = X[:, 0]
      min_x, max_x = min(xs) * 0.9, max(xs) * 1.1 

      # decision boundary hyperplane
      db_hyp_y1 = hyperplane(min_x, self.w, self.b, 0)
      db_hyp_y2 = hyperplane(max_x, self.w, self.b, 0)
      plt.plot([min_x, max_x], [db_hyp_y1, db_hyp_y2], 'k')

      # +ve support vector hyperplane
      psv_hyp_y1 = hyperplane(min_x, self.w, self.b, 1)
      psv_hyp_y2 = hyperplane(max_x, self.w, self.b, 1)
      plt.plot([min_x, max_x], [psv_hyp_y1, psv_hyp_y2], 'k--')

      # -ve support vector hyperplane
      nsv_hyp_y1 = hyperplane(min_x, self.w, self.b, -1)
      nsv_hyp_y2 = hyperplane(max_x, self.w, self.b,-1)
      plt.plot([min_x, max_x], [nsv_hyp_y1, nsv_hyp_y2], 'k--')
        
      plt.show()
    else:
      print("This function only visualizes data with 2 features.")