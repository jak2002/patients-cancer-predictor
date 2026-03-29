import joblib

class RandomForestModel:
    def __init__(self):
        self.model = joblib.load("models/random_forest.pkl")

    def predict(self, data):
        return self.model.predict([data])[0]
