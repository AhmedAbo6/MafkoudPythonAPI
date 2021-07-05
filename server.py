# Serve model as a flask application

import pickle
from flask import Flask, request

model = None
app = Flask(__name__)


# def load_model():
    # global model
    # model variable refers to the global variable
    # with open('iris_trained_model.pkl', 'rb') as f:
        # model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


# @app.route('/age-gender', methods=['POST'])
# def getAgeGender():
#     # Works only for a single sample
#     if request.method == 'POST':
#         data = request.get_json()
#         prediction = model.predict(data)
#     return str(prediction[0])


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
