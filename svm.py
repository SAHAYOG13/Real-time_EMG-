import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your handcrafted feature dataset, X, and target labels, y
feature_matrix = pd.read_csv('Combined.csv')
X = feature_matrix.iloc[:, :-1].values
y = feature_matrix['label'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM model with your choice of hyperparameters
clf = svm.SVC(kernel='linear', C=0.1, gamma=0.1)

# Train the SVM model on the training set
clf.fit(X_train, y_train)

# Use the trained SVM model to predict on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the SVM model on the testing set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import joblib
# Save the trained SVM model
joblib.dump(clf, 'svm_model.joblib')

# Save the LabelEncoder
joblib.dump(encoder, 'label_encoder.joblib')

