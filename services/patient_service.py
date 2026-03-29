from db import connect_db
import sqlite3

class PatientService:
    def __init__(self):
        # Keep a connection open for read operations
        self.conn = connect_db()
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def add_patient(self, name, age, email, gender, phone,
                    radius_mean=None, texture_mean=None, perimeter_mean=None, area_mean=None,
                    smoothness_mean=None, compactness_mean=None, concavity_mean=None,
                    concave_points_mean=None, symmetry_mean=None, fractal_dimension_mean=None,
                    medical_data=None):
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (
                name, age, email, gender, phone,
                radius_mean, texture_mean, perimeter_mean, area_mean,
                smoothness_mean, compactness_mean, concavity_mean,
                concave_points_mean, symmetry_mean, fractal_dimension_mean,
                medical_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, age, email, gender, phone,
              radius_mean, texture_mean, perimeter_mean, area_mean,
              smoothness_mean, compactness_mean, concavity_mean,
              concave_points_mean, symmetry_mean, fractal_dimension_mean,
              medical_data))
        conn.commit()
        conn.close()

    def get_all_patients(self):
        self.cursor.execute("""
            SELECT id, name, age, gender, email, phone, medical_data,
                   rf_prediction, rf_confidence,
                   svm_prediction, svm_confidence,
                   consensus_result
            FROM patients
        """)
        rows = self.cursor.fetchall()
        patients = []
        for row in rows:
            patients.append({
                "id": row["id"],
                "name": row["name"],
                "age": row["age"],
                "gender": row["gender"],
                "email": row["email"],
                "phone": row["phone"],
                "medical_data": row["medical_data"] if row["medical_data"] else "Not yet filled",
                "rf_prediction": row["rf_prediction"] if row["rf_prediction"] else "Not yet predicted",
                "rf_confidence": row["rf_confidence"] if row["rf_confidence"] else "-",
                "svm_prediction": row["svm_prediction"] if row["svm_prediction"] else "Not yet predicted",
                "svm_confidence": row["svm_confidence"] if row["svm_confidence"] else "-",
                "consensus_result": row["consensus_result"] if row["consensus_result"] else "Not yet predicted"
            })
        return patients

    def get_patient_by_id(self, patient_id):
        conn = connect_db()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def update_patient(self, patient_id, name, age, email, gender, phone,
                       radius_mean, texture_mean, perimeter_mean, area_mean,
                       smoothness_mean, compactness_mean, concavity_mean,
                       concave_points_mean, symmetry_mean, fractal_dimension_mean,
                       medical_data):
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE patients
            SET name=?, age=?, email=?, gender=?, phone=?,
                radius_mean=?, texture_mean=?, perimeter_mean=?, area_mean=?,
                smoothness_mean=?, compactness_mean=?, concavity_mean=?,
                concave_points_mean=?, symmetry_mean=?, fractal_dimension_mean=?,
                medical_data=?
            WHERE id=?
        """, (name, age, email, gender, phone,
              radius_mean, texture_mean, perimeter_mean, area_mean,
              smoothness_mean, compactness_mean, concavity_mean,
              concave_points_mean, symmetry_mean, fractal_dimension_mean,
              medical_data, patient_id))
        conn.commit()
        conn.close()

    def delete_patient(self, patient_id):
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients WHERE id=?", (patient_id,))
        conn.commit()
        conn.close()
