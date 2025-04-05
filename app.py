from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset and train the model
data = pd.read_csv('logistic_ds.csv')
data['Churn'] = data['Churn'].astype(int)

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=['State', 'International plan', 'Voice mail plan'], drop_first=True)

# Define feature and target variables
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    account_length = float(request.form['account_length'])
    area_code = int(request.form['area_code'])
    intl_plan = request.form['intl_plan']  # Yes or No
    voice_mail_plan = request.form['voice_mail_plan']  # Yes or No
    number_vmail_messages = int(request.form['number_vmail_messages'])
    total_day_minutes = float(request.form['total_day_minutes'])
    total_day_calls = int(request.form['total_day_calls'])
    total_day_charge = float(request.form['total_day_charge'])
    total_eve_minutes = float(request.form['total_eve_minutes'])
    total_eve_calls = int(request.form['total_eve_calls'])
    total_eve_charge = float(request.form['total_eve_charge'])
    total_night_minutes = float(request.form['total_night_minutes'])
    total_night_calls = int(request.form['total_night_calls'])
    total_night_charge = float(request.form['total_night_charge'])
    total_intl_minutes = float(request.form['total_intl_minutes'])
    total_intl_calls = int(request.form['total_intl_calls'])
    total_intl_charge = float(request.form['total_intl_charge'])
    customer_service_calls = int(request.form['customer_service_calls'])

    # Convert Yes/No to 1/0 for 'International plan' and 'Voice mail plan'
    intl_plan = 1 if intl_plan.lower() == 'yes' else 0
    voice_mail_plan = 1 if voice_mail_plan.lower() == 'yes' else 0

    # Create input DataFrame for prediction
    input_data = pd.DataFrame([[account_length, area_code, intl_plan, voice_mail_plan,
                                 number_vmail_messages, total_day_minutes, total_day_calls,
                                 total_day_charge, total_eve_minutes, total_eve_calls,
                                 total_eve_charge, total_night_minutes, total_night_calls,
                                 total_night_charge, total_intl_minutes, total_intl_calls,
                                 total_intl_charge, customer_service_calls]],
                               columns=['Account length', 'Area code', 'International plan', 
                                        'Voice mail plan', 'Number vmail messages', 'Total day minutes',
                                        'Total day calls', 'Total day charge', 'Total eve minutes',
                                        'Total eve calls', 'Total eve charge', 'Total night minutes',
                                        'Total night calls', 'Total night charge', 'Total intl minutes',
                                        'Total intl calls', 'Total intl charge', 'Customer service calls'])

    # Handle one-hot encoding for categorical features (ensure missing columns are added)
    input_data = pd.get_dummies(input_data, columns=['International plan', 'Voice mail plan'], drop_first=True)
    
    for column in X.columns:
        if column not in input_data.columns:
            input_data[column] = 0  # Ensure that missing columns in the input are added with 0 value

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the trained logistic regression model
    prediction = log_reg.predict(input_data_scaled)
    result = 'Churn' if prediction[0] == 1 else 'No Churn'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
