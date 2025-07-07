######## TASK-3 END-TO END DATA SCIENCE PROJECT USING FLASK API

from flask import Flask, request
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

model_path = "model.pkl"
data_path = "housing.csv"

# Train model from CSV if model file doesn't exist
if not os.path.exists(model_path):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        X = data[['area']]
        y = data['price']
        model = LinearRegression()
        model.fit(X, y)
        pickle.dump(model, open(model_path, 'wb'))
    else:
        raise FileNotFoundError("CSV file 'housing.csv' not found.")
else:
    model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return '''
    <h2>House Price Prediction</h2>
    <form action="/predict" method="post">
      Enter Area (in sq ft): <input type="text" name="area"><br><br>
      <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        prediction = model.predict([[area]])
        return f'<h3>Predicted Price: ₹{prediction[0]:,.2f}</h3>'
    except Exception as e:
        return f'<h3>Error: {str(e)}</h3>'

if __name__ == '__main__':
    app.run(debug=True, port=5070)
