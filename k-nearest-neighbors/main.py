from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN

data_set = {
  'red': [[1, 2], [2, 2], [2, 1]],
  'green': [[4, 3], [5, 3], [4, 4]],
  'blue': [[5, 6], [6, 6], [6, 7]]
}
k = 5
new_feature = [7, 7]

knn = KNN(data_set, k)
group = knn.predict(new_feature)
print("Feature set=" + str(new_feature), "class=" + group)
print("Confidence=" + str(knn.confidence))