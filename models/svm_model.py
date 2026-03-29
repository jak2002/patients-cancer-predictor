import joblib

class SVMModel:
    def __init__(self):
        self.model = joblib.load("models/svm.pkl")

    def predict(self, data):
        return self.model.predict([data])[0]
