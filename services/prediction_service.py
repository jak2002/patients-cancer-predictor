from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from db import connect_db

import sqlite3
import joblib

class PredictionService:
    def __init__(self):
        # Load trained models
        self.svm = joblib.load("models/svm.pkl")
        self.rf = joblib.load("models/random_forest.pkl")

    def consensus_diagnosis(self, rf_pred, rf_conf, svm_pred, svm_conf):
        """
        Decide final diagnosis based on Random Forest and SVM predictions.
        """
        if rf_pred == svm_pred:
            return f"Final Diagnosis (Consensus): {rf_pred} (Agreement)"
        else:
            # Confidence gap threshold
            if abs(rf_conf - svm_conf) >= 15:  # 15% difference
                chosen = rf_pred if rf_conf > svm_conf else svm_pred
                return (f"Final Diagnosis (Consensus): {chosen} "
                        f"(Resolved by higher confidence: RF={rf_conf:.2f}%, SVM={svm_conf:.2f}%)")
            else:
                return (f"Final Diagnosis (Consensus): Disagreement "
                        f"(RF={rf_pred}, {rf_conf:.2f}%; SVM={svm_pred}, {svm_conf:.2f}%) → Doctor review required")

    def predict(self, patient_id):
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT radius_mean, texture_mean, perimeter_mean, area_mean,
                   smoothness_mean, compactness_mean, concavity_mean,
                   concave_points_mean, symmetry_mean, fractal_dimension_mean
            FROM patients WHERE id=?
        """, (patient_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return {"error": "Patient not found"}

        features = [float(x) if x not in (None, '') else 0.0 for x in row]
        X = [features]

        # Predictions
        svm_pred = int(self.svm.predict(X)[0])
        rf_pred = int(self.rf.predict(X)[0])

        # Probabilities
        svm_prob = self.svm.predict_proba(X)[0][svm_pred] * 100
        rf_prob = self.rf.predict_proba(X)[0][rf_pred] * 100

        # Map numeric predictions to labels
        label_map = {0: "Benign", 1: "Malignant"}
        svm_label = label_map[svm_pred]
        rf_label = label_map[rf_pred]

        # Consensus
        final_result = self.consensus_diagnosis(rf_label, rf_prob, svm_label, svm_prob)

        # Save results back into DB
        cursor.execute("""
            UPDATE patients
            SET rf_prediction=?, rf_confidence=?,
                svm_prediction=?, svm_confidence=?,
                consensus_result=?
            WHERE id=?
        """, (rf_label, rf_prob, svm_label, svm_prob, final_result, patient_id))
        conn.commit()
        conn.close()

        return {
            "SVM": svm_label,
            "SVM_prob": round(svm_prob, 2),
            "RandomForest": rf_label,
            "RF_prob": round(rf_prob, 2),
            "Consensus": final_result
        }
