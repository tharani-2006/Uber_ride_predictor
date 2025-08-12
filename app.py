import numpy as np
from flask import Flask,request,render_template
import math
import pickle

app=Flask(__name__)

model2= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Accept numeric fields as float to handle decimals from the form
    feature_values = [float(x) for x in request.form.values()]
    final_features = np.array(feature_values).reshape(1, -1)
    prediction = model2.predict(final_features)
    output = round(prediction[0], 2)
    return render_template(
        'index.html',
        prediction_text='Number of weekly riders is {}'.format(math.floor(output))
    )

if __name__ == '__main__':
    app.run(debug=True)