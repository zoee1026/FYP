# import pickle
# from sklearn.neighbors import KNeighborsRegressor

# # Create a KNN model object
# knn = KNeighborsRegressor(n_neighbors=3)

# # Train the model on some data
# X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
# y_train = [0, 0, 1, 1]
# knn.fit(X_train, y_train)

# # Save the trained model to a file
# filename = 'knn_model.sav'
# pickle.dump(knn, open(filename, 'wb'))

# # Load the saved model from a file
# loaded_model = pickle.load(open(filename, 'rb'))

# # Use the loaded model to make predictions on some test data
# X_test = [[0.5, 0.5], [2.5, 2.5], [4, 4]]
# y_pred = loaded_model.predict(X_test)

# # Print the predicted labels
# print(y_pred)

import numpy as np
from sklearn.neighbors import KDTree

# Generate some example data
X = np.random.rand(10000, 2)  # Centroid coordinates
y = np.random.rand(10000)     # Heading angles

# Define the number of buckets and the bucket size
num_buckets = 100
bucket_size = 1.0 / num_buckets

# Compute the bucket indices for each centroid
bucket_indices = np.floor(X / bucket_size).astype(int)

# Build a KD-tree for each bucket
buckets = {}
for i in range(num_buckets):
    indices = np.where(np.all(bucket_indices == i, axis=1))[0]
    if len(indices) > 0:
        buckets[i] = KDTree(X[indices])

# Define a function to predict the heading angle of a new centroid
def predict_heading_angle(centroid):
    bucket_index = tuple(np.floor(centroid / bucket_size).astype(int))
    if bucket_index in buckets:
        _, indices = buckets[bucket_index].query([centroid], k=5)
        return np.mean(y[indices])
    else:
        # No centroids in this bucket, return a default value
        return 0.0