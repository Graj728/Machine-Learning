"""Project 9: Single-Layer Classifier vs. Multilayer Neural Network.
Name: Gaurav    Project: 9

Run: python Project_9_Submission.py   (keep Iris.csv in the same folder)
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load data. The Id column is NOT used as a feature.
csv_file = Path(__file__).with_name('Iris.csv')
df = pd.read_csv(csv_file)
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_columns]
# Binary labels: 0 = Iris-setosa, 1 = other Iris species
y = (df['Species'] != 'Iris-setosa').astype(int)

# STEP 2: ONE reproducible train/test split shared by all models.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# STEP 3: Single-layer sigmoid classifier (logistic regression, no hidden layer).
slp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
slp.fit(X_train, y_train)
slp_pred = slp.predict(X_test)

# STEP 4: MLP with ONE hidden layer.
# ORIGINAL setting: hidden_layer_sizes=(8,)
# CHANGED setting (Task A): hidden_layer_sizes=(16,)
def build_mlp(hidden_neurons):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(hidden_neurons,), activation='relu',
                      max_iter=2000, random_state=42)
    )

mlp8 = build_mlp(8)      # original
mlp8.fit(X_train, y_train)
mlp8_pred = mlp8.predict(X_test)

mlp16 = build_mlp(16)    # modified (changed from 8 to 16)
mlp16.fit(X_train, y_train)
mlp16_pred = mlp16.predict(X_test)

# STEP 5: Compare.
print('Single-layer classifier accuracy:      ', round(accuracy_score(y_test, slp_pred), 3))
print('MLP (8 hidden neurons) accuracy:       ', round(accuracy_score(y_test, mlp8_pred), 3))
print('MLP (16 hidden neurons, changed) acc.: ', round(accuracy_score(y_test, mlp16_pred), 3))

print('\nSingle-layer classification report:')
print(classification_report(y_test, slp_pred, target_names=['Setosa', 'Other'], zero_division=0))
print('MLP (8 neurons) classification report:')
print(classification_report(y_test, mlp8_pred, target_names=['Setosa', 'Other'], zero_division=0))
print('MLP (16 neurons) classification report:')
print(classification_report(y_test, mlp16_pred, target_names=['Setosa', 'Other'], zero_division=0))