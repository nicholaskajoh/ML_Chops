# Support Vector Machine (SVM)
The SVM is a supervised learning algorithm that can be used for classification and regression. Here we use it for classification.

Given a set of training samples, each marked as belonging to one or the other of two categories, the objective is to find the best splitting boundary between the data. This boundary is known as the best separating hyperplane. With this hyperplane in place, we can easily predict an input feature set as being in one of the two categories.

## Implementation
In training an SVM, we're attempting to solve an optimization problem. Essentially, we want to minimize `1/2 * ||w||^2` given the constraint `yi(xi.w + b) >= 1` (see comments in code). In this implementation, the optimization is achieved using Convex Optimization with the help of the CVXOPT library. There are other ways to solve this problem e.g by using the popular Sequential Minimal Optimization (SMO) algorithm.

## Data set
Breast Cancer Wisconsin (diagnostic) data set from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data.