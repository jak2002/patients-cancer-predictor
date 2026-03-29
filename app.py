from datetime import datetime
import base64
import io
import os
import sqlite3

from flask import Flask, flash, render_template, request, redirect, url_for, make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import joblib
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from xhtml2pdf import pisa

from db import connect_db, get_db_path
import doctor_helpers
from services.patient_service import PatientService
from services.prediction_service import PredictionService
import os, sqlite3
from db import connect_db
from db import get_db_path
print("Flask using DB:", get_db_path())

REQUIRED_MEDICAL_FIELDS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
]

app = Flask(__name__)
# Set a unique secret key for sessions
app.secret_key = os.environ.get("SECRET_KEY", "9f2c8a7d3b4e5f6a8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1")
patient_service = PatientService()
prediction_service = PredictionService()


# --- Flask-Login setup ---
login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = "login"   # redirect unauthenticated users to /login

class User(UserMixin):
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

# --- User loader function ---
@login_manager.user_loader
def load_user(user_id):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors WHERE id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return User(row["id"], row["email"], row["password"])
    return None

# Load model globally so routes can access it
model_rf = joblib.load("models/random_forest.pkl")
model_svm = joblib.load("models/svm.pkl")

def generate_report(patient_id):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    patient = cursor.fetchone()

    cursor.execute("SELECT note, created_at FROM doctor_notes WHERE patient_id=?", (patient_id,))
    notes = cursor.fetchall()

    cursor.execute("SELECT model, result, confidence, timestamp FROM predictions WHERE patient_id=?", (patient_id,))
    predictions = cursor.fetchall()

    conn.close()

    # Render HTML using your Jinja template
    html_out = render_template("reports.html", patient=patient, notes=notes, predictions=predictions)

    # Save PDF using pdfkit
    pdf_path = os.path.join(REPORTS_DIR, f"patient_{patient_id}_report.pdf")
    pdfkit.from_string(html_out, pdf_path)

    print(f"✅ Report generated: {pdf_path}")
    return pdf_path

import matplotlib.pyplot as plt
import io
import base64

def get_feature_importance_plot(rf_model, svm_model, feature_names):
    plt.figure(figsize=(10, 6))

    # --- Random Forest importances ---
    if hasattr(rf_model, "feature_importances_"):
        rf_importances = rf_model.feature_importances_
        plt.bar(range(len(feature_names)), rf_importances, alpha=0.7, label="Random Forest")

    # --- SVM coefficients (only if linear kernel) ---
    if hasattr(svm_model, "coef_"):
        # Take absolute value of coefficients for importance
        svm_importances = abs(svm_model.coef_[0])
        plt.bar(range(len(feature_names)), svm_importances, alpha=0.7, label="SVM (linear)")
    else:
        print("⚠️ SVM does not provide feature importances unless using a linear kernel.")

    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature Importance Comparison")
    plt.legend()

    # Save plot to base64 string for embedding in HTML
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    importance_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    return importance_img



# --- Dashboard route ---
@app.route('/')
@login_required
def dashboard():
    doctor_id = current_user.id  # from Flask-Login

    # Patients linked to this doctor
    patients = doctor_helpers.get_doctor_dashboard(doctor_id)

    # Global statistics across all patients
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    all_patients = cursor.fetchall()
    conn.close()

    ages = {"labels": ["<30", "30-50", "50+"], "values": [0, 0, 0]}
    genders = {"labels": ["Male", "Female"], "values": [0, 0]}
    outcomes = {"labels": ["Benign", "Malignant"], "values": [0, 0]}

    for p in all_patients:
        # Age distribution
        if p["age"] < 30:
            ages["values"][0] += 1
        elif p["age"] <= 50:
            ages["values"][1] += 1
        else:
            ages["values"][2] += 1

        # Gender distribution
        if p["gender"] == "Male":
            genders["values"][0] += 1
        elif p["gender"] == "Female":
            genders["values"][1] += 1

        # Prediction outcomes
        if "diagnosis" in p.keys():
            if p["diagnosis"] == "Benign":
                outcomes["values"][0] += 1
            elif p["diagnosis"] == "Malignant":
                outcomes["values"][1] += 1

    return render_template(
        'dashboard.html',
        doctor=current_user,
        patients=patients,   # doctor-specific patients
        ages=ages,
        genders=genders,
        outcomes=outcomes
    )
@app.route('/register_patient', methods=['GET', 'POST'])
@login_required
def register_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        gender = request.form.get('gender')
        phone = request.form.get('phone')

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (name, age, email, gender, phone)
            VALUES (?, ?, ?, ?, ?)
        """, (name, age, email, gender, phone))
        conn.commit()
        conn.close()

        flash("Patient registered successfully. You can add medical data later by editing the patient record.")
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/patients')
@login_required
def patients():
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    patients = cursor.fetchall()
    conn.close()
    return render_template("patients.html", patients=patients)


@app.route('/search_patients')
@login_required
def search_patients():
    # implement search logic here
    return render_template("search_patients.html")



@app.route('/run_predictions')
@login_required
def run_predictions():
    # Example: run predictions for all patients missing results
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE rf_prediction IS NULL OR svm_prediction IS NULL")
    patients = cursor.fetchall()

    for patient in patients:
        features = [
            patient["radius_mean"], patient["texture_mean"], patient["perimeter_mean"],
            patient["area_mean"], patient["smoothness_mean"], patient["compactness_mean"],
            patient["concavity_mean"], patient["concave_points_mean"],
            patient["symmetry_mean"], patient["fractal_dimension_mean"]
        ]
        X = [features]

        # Run models
        pred_rf = model_rf.predict(X)[0]
        pred_svm = model_svm.predict(X)[0]

        label_map = {0: "Benign", 1: "Malignant"}
        pred_rf_label = label_map[pred_rf]
        pred_svm_label = label_map[pred_svm]

        rf_conf = model_rf.predict_proba(X)[0][pred_rf] * 100
        svm_conf = model_svm.predict_proba(X)[0][pred_svm] * 100

        if pred_rf_label == pred_svm_label:
            final_prediction = pred_rf_label
        else:
            final_prediction = f"Disagreement: RF={pred_rf_label}, SVM={pred_svm_label}"

        cursor.execute("""
            UPDATE patients
            SET rf_prediction=?, rf_confidence=?, svm_prediction=?, svm_confidence=?, consensus_result=?
            WHERE id=?
        """, (pred_rf_label, rf_conf, pred_svm_label, svm_conf, final_prediction, patient["id"]))

    conn.commit()
    conn.close()
    flash("Predictions updated for all patients.")
    return redirect(url_for("dashboard"))

@app.route('/prediction_history')
@login_required
def prediction_history():
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT p.id                         AS patient_id,
                          p.name                       AS patient_name,
                          pr.model,
                          pr.prediction_result         AS result,
                          COALESCE(pr.confidence, 0.0) AS confidence,
                          COALESCE(pr.created_at, '') AS timestamp
                   FROM patients p
                       LEFT JOIN predictions pr
                   ON p.id = pr.patient_id
                   ORDER BY pr.created_at DESC
                   """)
    rows = cursor.fetchall()
    conn.close()

    # Convert to plain dicts so templates can use patient.result or patient['result']
    history = [dict(r) for r in rows]

    return render_template("prediction_history.html", history=history)


@app.route('/predict/<int:patient_id>')
@login_required
def predict(patient_id):
    force = request.args.get("force") == "1"

    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()

    if row is None:
        conn.close()
        flash("Patient not found.", "danger")
        return redirect(url_for("dashboard"))

    patient = dict(row)

    # build features list
    features = [
        patient.get("radius_mean"),
        patient.get("texture_mean"),
        patient.get("perimeter_mean"),
        patient.get("area_mean"),
        patient.get("smoothness_mean"),
        patient.get("compactness_mean"),
        patient.get("concavity_mean"),
        patient.get("concave_points_mean"),
        patient.get("symmetry_mean"),
        patient.get("fractal_dimension_mean")
    ]
    X = [features]

    # Run models
    try:
        pred_rf = model_rf.predict(X)[0]
        pred_svm = model_svm.predict(X)[0]
    except Exception as e:
        conn.close()
        flash(f"Model prediction failed: {e}", "danger")
        return redirect(url_for("dashboard"))

    label_map = {0: "Malignant", 1: "Benign"}
    pred_rf_label = label_map.get(pred_rf, str(pred_rf))
    pred_svm_label = label_map.get(pred_svm, str(pred_svm))

    def safe_confidence(model, X, pred_index):
        try:
            probs = model.predict_proba(X)[0]
            if 0 <= pred_index < len(probs):
                val = float(probs[pred_index]) * 100.0
                if np.isnan(val):
                    return 0.0
                return val
        except Exception:
            pass
        return 0.0

    rf_conf = safe_confidence(model_rf, X, int(pred_rf))
    svm_conf = safe_confidence(model_svm, X, int(pred_svm))

    # Consensus
    if pred_rf_label == pred_svm_label:
        final_prediction = pred_rf_label
    else:
        final_prediction = f"Disagreement: RF={pred_rf_label}, SVM={pred_svm_label}"

    # Save predictions to patients table
    try:
        cursor.execute("""
            UPDATE patients
            SET rf_prediction=?, rf_confidence=?, svm_prediction=?, svm_confidence=?, consensus_result=?
            WHERE id=?
        """, (pred_rf_label, rf_conf, pred_svm_label, svm_conf, final_prediction, patient_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        flash(f"Failed to save prediction: {e}", "danger")
        return redirect(url_for("dashboard"))

    # Feature importance and plotting
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
    ]
    n_features = len(feature_names)

    try:
        rf_model = model_rf.named_steps.get("rf", None) if hasattr(model_rf, "named_steps") else None
        rf_importances = np.asarray(rf_model.feature_importances_, dtype=float) if rf_model is not None else np.zeros(n_features)
    except Exception:
        rf_importances = np.zeros(n_features)

    try:
        svm_model = model_svm.named_steps.get("svc", None) if hasattr(model_svm, "named_steps") else None
        svm_coeffs = np.asarray(svm_model.coef_[0], dtype=float) if (svm_model is not None and hasattr(svm_model, "coef_")) else np.zeros(n_features)
    except Exception:
        svm_coeffs = np.zeros(n_features)

    def ensure_length(arr, n):
        arr = np.asarray(arr, dtype=float)
        if arr.size < n:
            return np.concatenate([arr, np.zeros(n - arr.size)])
        if arr.size > n:
            return arr[:n]
        return arr

    rf_importances = ensure_length(rf_importances, n_features)
    svm_coeffs = ensure_length(svm_coeffs, n_features)

    rf_importances = np.nan_to_num(rf_importances, nan=0.0, posinf=0.0, neginf=0.0)
    svm_coeffs = np.nan_to_num(svm_coeffs, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, rf_importances, label="Random Forest")
        ax.barh(feature_names, svm_coeffs, alpha=0.5, label="SVM Coeffs")
        ax.set_xlabel("Importance / Coefficient")
        ax.legend()
        plt.tight_layout()

        os.makedirs("static", exist_ok=True)
        plot_path = os.path.join("static", f"feature_importance_{patient_id}.png")
        plt.savefig(plot_path)
        plt.close(fig)
    except Exception as e:
        plot_path = None
        flash(f"Feature importance plot failed: {e}", "warning")

    conn.close()

    return render_template(
        "results.html",
        patient_id=patient_id,
        pred_rf=pred_rf_label,
        pred_svm=pred_svm_label,
        svm_conf=svm_conf,
        rf_conf=rf_conf,
        results=final_prediction,
        importance_img=plot_path
    )
@app.route('/predict/<int:patient_id>/pdf')
def predict_pdf(patient_id):
    results = prediction_service.predict(patient_id)

    # Render HTML template to a string
    html = render_template(
        "results_pdf.html",
        patient_id=patient_id,
        results=results,
        error=results.get("error"),
        current_date="2026-02-10"  # or use datetime.now().strftime("%Y-%m-%d")
    )

    # Create a PDF file in memory
    pdf_stream = io.BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_stream)

    if pisa_status.err:
        return "Error generating PDF", 500

    # Return PDF as response
    response = make_response(pdf_stream.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=patient_{patient_id}_results.pdf'
    return response

@app.route('/edit/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        medical_data = request.form.get('medical_data')

        # Collect all medical features
        radius_mean = request.form.get('radius_mean')
        texture_mean = request.form.get('texture_mean')
        perimeter_mean = request.form.get('perimeter_mean')
        area_mean = request.form.get('area_mean')
        smoothness_mean = request.form.get('smoothness_mean')
        compactness_mean = request.form.get('compactness_mean')
        concavity_mean = request.form.get('concavity_mean')
        concave_points_mean = request.form.get('concave_points_mean')
        symmetry_mean = request.form.get('symmetry_mean')
        fractal_dimension_mean = request.form.get('fractal_dimension_mean')

        patient_service.update_patient(
            patient_id, name, age, email, gender, phone,
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean,
            medical_data
        )
        flash("Patient record updated successfully.")
        return redirect(url_for('dashboard'))

    patient = patient_service.get_patient_by_id(patient_id)
    return render_template('edit.html', patient=patient)

@app.route('/delete/<int:patient_id>', methods=['POST', 'GET'])
def delete_patient(patient_id):
    patient_service.delete_patient(patient_id)
    return redirect(url_for('dashboard'))

@app.route('/register_doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO doctors (name, email, password) VALUES (?, ?, ?)",
                       (name, email, password))
        conn.commit()
        conn.close()

        flash("Doctor registered successfully. Please log in.")
        return redirect(url_for('login'))

    return render_template('register_doctor.html')

from flask_login import login_user

from flask_login import login_user
from werkzeug.security import check_password_hash

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = connect_db()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors WHERE email=?", (email,))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row["password"], password):
            # User expects id as first argument
            user = User(row["id"], row["email"], row["password"])
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for('login'))



@app.route("/reports")
@login_required
def reports():
    conn = connect_db()
    # Make rows accessible by column name
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM reports")
        rows = cursor.fetchall()
        # Convert sqlite3.Row objects to plain dicts for template safety
        reports = [dict(r) for r in rows]

        # Pick a patient_id from the first report if any exist
        selected_patient_id = reports[0].get("patient_id") if reports else None

    except Exception as e:
        # Log or flash an error and return an empty list so template still renders
        flash(f"Failed to load reports: {e}", "danger")
        reports = []
        selected_patient_id = None

    finally:
        conn.close()

    return render_template(
        "reports.html",
        reports=reports,
        selected_patient_id=selected_patient_id
    )
# example
@app.route('/profile')
@login_required
def profile():
    # current_user is provided by Flask-Login
    return render_template("profile.html", user=current_user)

@app.route('/change_password', methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_pw = request.form.get("current_password")
        new_pw = request.form.get("new_password")
        confirm_pw = request.form.get("confirm_password")

        if new_pw != confirm_pw:
            flash("New passwords do not match.", "auth")
            return redirect(url_for("change_password"))

        # Verify current password
        if not current_user.check_password(current_pw):
            flash("Current password is incorrect.", "auth")
            return redirect(url_for("change_password"))

        # Update password
        current_user.set_password(new_pw)
        db.session.commit()
        flash("Password updated successfully.", "auth")
        return redirect(url_for("dashboard"))

    return render_template("change_password.html")
@app.route('/feature_importance')
@login_required
def feature_importance():
    import pandas as pd
    import joblib

    # Load the trained Random Forest pipeline
    rf_pipeline = joblib.load("models/random_forest.pkl")

    # Extract feature importances
    feature_names = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension"
    ]
    importances = rf_pipeline.named_steps["rf"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Convert to list of dicts for Jinja
    importance_records = importance_df.to_dict(orient="records")

    return render_template("feature_importance.html", importance_records=importance_records)
@app.route('/help', endpoint='help')
@login_required
def help_page():
    return render_template("help.html")

@app.route('/about')
@login_required
def about():
    return render_template("about.html")

@app.route('/timeline/<int:patient_id>', methods=["GET", "POST"])
@login_required
def timeline(patient_id):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Fetch patient
    cursor.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    patient = cursor.fetchone()
    if not patient:
        flash("Patient not found.", "danger")
        return redirect(url_for("patients"))

    # Handle new doctor note submission
    if request.method == "POST":
        note_text = request.form.get("note")
        if note_text:
            cursor.execute(
                "INSERT INTO doctor_notes (patient_id, note) VALUES (?, ?)",
                (patient_id, note_text)
            )
            conn.commit()
            flash("Doctor note added successfully.", "success")
            return redirect(url_for("timeline", patient_id=patient_id))

    # Build timeline events
    timeline = []

    # Predictions
    cursor.execute("SELECT * FROM predictions WHERE patient_id=?", (patient_id,))
    for pred in cursor.fetchall():
        timeline.append({
            "date": pred.get("timestamp", "N/A"),
            "type": "Prediction",
            "details": "Prediction run",
            "prediction": pred.get("consensus_result"),
            "confidence": pred.get("rf_confidence"),
            "report_id": pred.get("report_id")
        })

    # Doctor notes
    cursor.execute("SELECT * FROM doctor_notes WHERE patient_id=? ORDER BY created_at ASC", (patient_id,))
    for note in cursor.fetchall():
        timeline.append({
            "date": note["created_at"],
            "type": "Doctor Note",
            "details": note["note"],
            "prediction": None,
            "confidence": None,
            "report_id": None
        })

    conn.close()

    return render_template("timeline.html", patient_name=patient["name"], timeline=timeline)

@app.route("/add_note/<int:patient_id>", methods=["POST"])
def add_note(patient_id):
    note = request.form["note"]

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO doctor_notes (patient_id, note) VALUES (?, ?)", (patient_id, note))
    conn.commit()
    conn.close()

    pdf_path = generate_report(patient_id)
    return f"Note added for patient {patient_id}. Report saved at {pdf_path}"
@app.route("/check_db")
def check_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return f"Tables in DB: {tables}"

if __name__ == '__main__':
    app.run(debug=True)
