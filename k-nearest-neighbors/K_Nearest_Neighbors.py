import numpy as np
from collections import Counter

class K_Nearest_Neighbors:
  def __init__(self, data_set, k):
    self.ds = data_set
    self.k = k

  def predict(self, feature_set):
    distances = []
    for group in self.ds:
      for feature in self.ds[group]:
        # get the euclidean distance (e_d) of each feature and the new feature
        e_d = np.linalg.norm(np.array(feature) - np.array(feature_set))
        distances.append([e_d, group])
    # sort the distances in ascending order (by e_d) and pick the first k elements
    # these are the distances of the feature sets nearest to the new feature set
    nearest = sorted(distances)[:self.k]
    # dispose of the distances (we only need the groups of the nearest feature sets at this point)
    votes = [d[1] for d in nearest]
    # get the group with the highest count in votes
    nearest_group = Counter(votes).most_common(1)[0]
    feature_set_group, self.confidence = nearest_group[0], nearest_group[1] / self.k
    return feature_set_group

  def test(self, test_data):
    correct = 0
    total = 0
    for group in test_data:
      for feature_set in test_data[group]:
        group_prediction = self.predict(feature_set)
        if group_prediction == group:
          correct += 1
        total += 1
    accuracy = correct / total
    print("Accuracy=", accuracy)