from flask import Flask, request, jsonify
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained models
with open('model_svm.pkl', 'rb') as f:
    model_svm = pickle.load(f)

with open('model_lr.pkl', 'rb') as f:
    model_lr = pickle.load(f)

with open('model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Define label encoder
label_encoder = LabelEncoder()

@app.route('/predict', methods=['POST'])
def predict_churn():
    # Get input data from request
    data = request.json
    
    # Convert input data to DataFrame
    df = pd.DataFrame(data)
    
    # Apply label encodings
    df['International plan'] = label_encoder.fit_transform(df['International plan'])
    df['Voice mail plan'] = label_encoder.fit_transform(df['Voice mail plan'])
    df['State'] = label_encoder.fit_transform(df['State'])
    
    
    # Predict churn using SVM
    churn_prediction_svm = model_svm.predict(df)
    
    # Predict churn using Random Forest
    churn_prediction_rf = model_rf.predict(df)
    
    # Predict churn using XGBoost
    churn_prediction_xgb = xgb_model.predict(df)
    
    # For Logistic Regression, drop unnecessary features
    df_lr = df.drop(columns=['Voice mail plan', 'Number vmail messages', 
                             'Total day minutes', 'Total day charge', 
                             'Total eve minutes', 'Total night minutes'])
    
    # Predict churn using Logistic Regression
    churn_prediction_lr = model_lr.predict(df_lr)
    
    # Return predictions
    return jsonify({
        'churn_prediction_svm': churn_prediction_svm.tolist(),
        'churn_prediction_lr': churn_prediction_lr.tolist(),
        'churn_prediction_rf': churn_prediction_rf.tolist(),
        'churn_prediction_xgb': churn_prediction_xgb.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)