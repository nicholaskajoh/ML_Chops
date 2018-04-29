import numpy as np

class K_Means:
    def __init__(self, k=2, max_iter=300, tolerance=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tolerance

    def train(self, data):
        self.centroids = {}
        self.groups = {}

        # set initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        # optimize
        for i in range(self.max_iter):
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

            # store current centroids
            old_centroids = dict(self.centroids)

            # calculate new centroids
            for l in range(self.k):
                self.centroids[l] = np.average(self.groups[l], axis=0)

            # check if change in any centroid position is
            # insignificant based on the set tolerance
            # if so, then we're already optimized
            optimized = False
            for centroid_key in self.centroids:
                old_centroid = old_centroids[centroid_key]
                new_centroid = self.centroids[centroid_key]
                a = np.array(new_centroid - old_centroid)
                b = np.array(old_centroid)
                change = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                if abs(np.sum(change * 100.0)) <= self.tol:
                    optimized = True
                    break
            if optimized:
                break

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