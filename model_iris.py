# Importing the necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load the iris dataset from sklearn datasets
dataset = load_iris()

# Loading features and labels from the dataset
features = dataset.data
labels = dataset.target.astype(str)
for i in range(3):
    labels = np.where(labels == str(i), 'Iris '+ dataset.target_names[i].title(), labels)

# Splitting labels and features to training and testing sets
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialising Logistic Regression Model with maximum iterations as 500
model = LogisticRegression(max_iter=500)

# Fitting Model or Training Model with training features and labels
model.fit(feature_train, label_train)

# Predicting the labels for the testing features
label_pred = model.predict(feature_test)

# Finding the accuracy score
score = accuracy_score(label_test, label_pred)

# Dumping the model object to save it as pkl file
joblib.dump(model, open('model.pkl', 'wb+'))
