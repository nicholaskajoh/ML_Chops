import numpy as np

class K_Means:
    def __init__(self, k=2, iterations=25):
        self.k = k
        self.iterations = iterations

    def train(self, data):
        self.centroids = {}
        self.groups = {}

        # set initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        # optimize
        for i in range(self.iterations):
            for j in range(self.k):
                self.groups[j] = []

            for feature_set in data:
                # find closest centroid to a feature set
                distances = [np.linalg.norm(np.array(feature_set) \
                            - np.array(self.centroids[centroid_key])) \
                            for centroid_key in self.centroids]
                group = distances.index(min(distances))
                # add the feature set under the centroid group
                self.groups[group].append(feature_set)

            # you can make the training more efficient
            # by breaking off the loop if there's little
            # or no change in the position of the
            # centroids in a new iteration

    def test(self, X, y):
        clusters = {}
        for i in range(self.k):
            clusters[i] = []

        # sort Xs into k groups and see
        # if the groups contain similar ys
        # if they do, clustering was successful
        for i in range(len(X)):
            for j in range(self.k):
                if X[i] in self.groups[j]:
                    clusters[j].append(y[i])
                    break

        for i in range(self.k):
            print(clusters[i][:25]) # data too large to print all