import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        return render_template('index.html',
                               prediction_text='Low chances of person buying a car'.format(prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text='High chances of person buying a car'.format(prediction),
                               )


if __name__ == '__main__':
    app.run(port=5000, debug=True)
