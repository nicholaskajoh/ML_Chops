import numpy as np

class Naive_Bayes:
  def __init__(self, data_set):
    self.ds = data_set
    # get the means and variances of each feature for each class
    self.ds_means = self.ds.groupby(4).mean()
    self.ds_variances = self.ds.groupby(4).var()
    # probabilities of selecting each class in the dataset
    self.class_probabilities = self.get_class_probabilities(self.ds)

  def get_class_probabilities(self, data_set):
    class_sizes = data_set.groupby(4).size()
    ds_total = data_set.shape[0]
    probs = {}
    for i in class_sizes.iteritems():
      probs[i[0]] = i[1] / ds_total
    return probs

  def get_probability_density(self, x, mean, variance):
    # gaussian probability density function
    # calculates probablity of alpha given beta
    # pd is calculated for each feature of a feature set for each class
    # mean, variance are of a class/group
    pd = 1 / (np.sqrt(2 * np.pi * variance)) * np.exp((-(x - mean)**2) / (2 * variance))
    return pd

  def predict(self, x):
    feature_class_probabilities = {}
    for group, class_prob in self.class_probabilities.items():
      feature_class_probabilities[group] = class_prob
      for i in range(len(x)):
        feature_class_probabilities[group] *= self.get_probability_density(x[i], self.ds_means.loc[group][i], self.ds_variances.loc[group][i])
    # class of feature set is the one with the highest probability 
    feature_class = max(feature_class_probabilities, key=feature_class_probabilities.get)
    return feature_class

  def test(self, test_data):
    correct = 0
    total = 0
    for row in test_data.itertuples():
      feature_set = row[1:5]
      group = self.predict(feature_set)
      if group == row[5]:
        correct += 1
      else:
        print(feature_set, "prediction=", group, "correct=", row[5])
      total += 1
    accuracy = correct / total
    print("Accuracy=", accuracy)