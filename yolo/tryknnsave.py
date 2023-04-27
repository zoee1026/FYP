import pickle
from sklearn.neighbors import KNeighborsRegressor

# Create a KNN model object
knn = KNeighborsRegressor(n_neighbors=3)

# Train the model on some data
X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
y_train = [0, 0, 1, 1]
knn.fit(X_train, y_train)

# Save the trained model to a file
filename = 'knn_model.sav'
pickle.dump(knn, open(filename, 'wb'))

# Load the saved model from a file
loaded_model = pickle.load(open(filename, 'rb'))

# Use the loaded model to make predictions on some test data
X_test = [[0.5, 0.5], [2.5, 2.5], [4, 4]]
y_pred = loaded_model.predict(X_test)

# Print the predicted labels
print(y_pred)