import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the dataset
data = pd.read_excel("customer_churn_large_dataset.xlsx")

# Initial data exploration
print(data.head())
print(data.info())
print(data.describe())

data.fillna(data.mean(), inplace=True)

X = data.drop(columns=["churn_column"])
y = data["churn_column"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = data.drop(columns=["churn_column"])
y = data["churn_column"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 7: Model Building
model = RandomForestClassifier()

# Step 8: Train and Validate the Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model using metrics like accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Step 9: Model Optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    # Add other hyperparameters to tune
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

