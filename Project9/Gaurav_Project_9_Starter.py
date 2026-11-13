"""Project 9 starter: Single-Layer Classifier vs. Multilayer Neural Network.

Install packages: pip install pandas scikit-learn
Run: python Project_9_Starter.py
Keep Iris.csv in the same folder as this script.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load the data and identify features and target.
csv_file = Path(__file__).with_name('Iris.csv')
df = pd.read_csv(csv_file)
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_columns]
# Binary labels: 0 = Iris-setosa, 1 = other Iris species
y = (df['Species'] != 'Iris-setosa').astype(int)

# STEP 2: Create ONE reproducible train/test split for both models.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# STEP 3: Train a linear sigmoid classifier (no hidden layer).
# Logistic regression is a practical equivalent of one sigmoid output neuron.
slp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
slp.fit(X_train, y_train)
slp_pred = slp.predict(X_test)

# STEP 4: Train a multilayer perceptron with ONE hidden layer (8 neurons).
mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=2000,
                  random_state=42)
)   
mlp.fit(X_train, y_train)
#created a function to handle the number of hidden layer for MLP
def hiddenmlp(hidden):
   return make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(hidden,), activation='relu', max_iter=2000,
                  random_state=42)
)   
    

# STEP 5: Compare the two models.
mlp8=hiddenmlp(8)
mlp8.fit(X_train, y_train)

mlp_pred8 = mlp8.predict(X_test)
mlp16=hiddenmlp(16)
mlp16.fit(X_train, y_train)

mlp_pred16 = mlp16.predict(X_test)
print('Single-layer classifier accuracy:', round(accuracy_score(y_test, slp_pred), 3))
print('Multilayer perceptron accuracy:', round(accuracy_score(y_test, mlp_pred8), 3))
print('Multilayer perceptron accuracy:', round(accuracy_score(y_test, mlp_pred16), 3))
print('\nSingle-layer classification report:')
print(classification_report(y_test, slp_pred, target_names=['Setosa', 'Other'], zero_division=0))
print('Multilayer classification report:')
print("Multilayer classification report with 8 hidden neurons:")
print(classification_report(y_test, mlp_pred8, target_names=['Setosa', 'Other'], zero_division=0))
print("Multilayer classification report with 16 hidden neurons:")
print(classification_report(y_test, mlp_pred16, target_names=['Setosa', 'Other'], zero_division=0))

# TASKS (in addition to running the starter):
# A. Change hidden_layer_sizes=(8,) to (4,) or (16,). Record what happens.
# B. Record both models' accuracies in your submission.
# C. In your own words, explain one hidden layer and why results may be similar.
# D. Do NOT use the Id column as a feature.
