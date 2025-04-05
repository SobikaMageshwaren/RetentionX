import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Import joblib for saving the model

# Load the dataset
data = pd.read_csv('logistic_ds.csv')
print(data.shape)

# Preprocessing
data['Churn'] = data['Churn'].astype(int)
data = pd.get_dummies(data, columns=['State', 'International plan', 'Voice mail plan'], drop_first=True)

# Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_rep)

# Save the model
joblib.dump(log_reg, 'model.pkl')  # Save the Logistic Regression model
print("Model saved as 'model.pkl'")

# Add predictions to the test data
X_test['Actual_Churn'] = y_test
X_test['Predicted_Churn'] = y_pred

# Save to CSV
X_test.to_csv('churn_predictions.csv', index=False)
print("Predictions saved to 'churn_predictions.csv'")
