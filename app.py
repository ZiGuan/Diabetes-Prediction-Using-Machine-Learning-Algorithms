from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, IntegerField
from wtforms.validators import DataRequired
import numpy as np
import pickle

app = Flask(__name__)
# adding an additional layer of security to sensitive data such as user authentication credentials, sessions, and cookies.
app.config["SECRET_KEY"] = "mysecret"

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route("/formtest")
def form_page():
    return render_template('predict.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    
    features = [float(x) for x in request.form.values()] 
    test_data = np.array(features)
    data = test_data.reshape(1,-1)
    prediction = model.predict(data)
        
    return render_template('predict.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run()