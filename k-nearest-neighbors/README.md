# K Nearest Neighbors (KNN)
KNN is an algorithm which outputs a class membership of a feature set by a majority vote of its closest neighbors. Given a set of features, say height, weight and foot size, we should be able to classify reasonably accurately, a human as either male or female. We can achieve this by finding a number (k) of points closest to the feature set (imagine a 3D graph of training data -- height against weight against foot size). With the nearest points (neighbors) to our feature set, we cast a vote to see which class occurs the most among the neighbors. This is/should be the class of the feature set.

## Implementation
This implementation can handle data in multiple dimensions thanks to Numpy's Euclidean Distance function. Depending on the number of groups/classes in your data set, you may need to adjust the value of k.

## Data set
Iris data set from https://www.kaggle.com/styven/iris-dataset. Read more about the Iris data set: https://en.wikipedia.org/wiki/Iris_flower_data_set.