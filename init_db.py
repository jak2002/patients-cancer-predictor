from db import connect_db, get_db_path
import shutil
import os

DB_FILE = get_db_path()
print("DB path being used:", DB_FILE)

def ensure_tables(cursor):
    # Patients table (base schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT NOT NULL,
            gender TEXT,
            phone TEXT
        );
    """)

    # Doctors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );
    """)

    # Reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            report_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
    """)

    # Predictions table (base)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            prediction_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
    """)

    # Doctor notes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctor_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            note_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(doctor_id) REFERENCES doctors(id),
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
    """)

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table});")
    cols = [row[1] for row in cursor.fetchall()]  # row[1] is column name
    return column in cols

def add_column_if_missing(cursor, table, column_def):
    # column_def example: "radius_mean REAL"
    column_name = column_def.split()[0]
    if not column_exists(cursor, table, column_name):
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_def};")
            print(f"Added column {table}.{column_name}")
        except Exception as e:
            print(f"Failed to add column {table}.{column_name}: {e}")
    else:
        print(f"Column {table}.{column_name} already exists")

def migrate_schema(cursor):
    # Medical features to add to patients (nullable REAL)
    medical_columns = [
        "radius_mean REAL",
        "texture_mean REAL",
        "perimeter_mean REAL",
        "area_mean REAL",
        "smoothness_mean REAL",
        "compactness_mean REAL",
        "concavity_mean REAL",
        "concave_points_mean REAL",
        "symmetry_mean REAL",
        "fractal_dimension_mean REAL",
        # Add a JSON/text column to hold aggregated medical data if needed
        "medical_data TEXT"
    ]

    for col in medical_columns:
        add_column_if_missing(cursor, "patients", col)

    # Columns your app updates on patients (add if missing)
    patient_prediction_columns = [
        "rf_prediction TEXT",
        "rf_confidence REAL",
        "svm_prediction TEXT",
        "svm_confidence REAL",
        "consensus_result TEXT"
    ]
    for col in patient_prediction_columns:
        add_column_if_missing(cursor, "patients", col)

    # Expand predictions table with additional columns (add only if missing)
    prediction_columns = [
        "doctor_id INTEGER",
        "model TEXT",
        "confidence REAL",
        # Random forest specific prediction column requested (if you also want it here)
        "rf_prediction TEXT",
        "rf_confidence REAL"
    ]

    for col in prediction_columns:
        add_column_if_missing(cursor, "predictions", col)

def print_table_info(cursor, table):
    cursor.execute(f"PRAGMA table_info({table});")
    rows = cursor.fetchall()
    print(f"Schema for {table}:")
    for r in rows:
        # r format: (cid, name, type, notnull, dflt_value, pk)
        print(f"  - {r[1]} ({r[2]})")

def backup_db():
    # Create a simple backup before running migrations (safe-guard)
    try:
        backup_path = DB_FILE + ".bak"
        if os.path.exists(DB_FILE) and not os.path.exists(backup_path):
            shutil.copy2(DB_FILE, backup_path)
            print("Backup created at:", backup_path)
        elif not os.path.exists(DB_FILE):
            print("DB file does not exist yet; no backup created.")
        else:
            print("Backup already exists at:", backup_path)
    except Exception as e:
        print("Backup failed:", e)

def init_db():
    # Backup existing DB file (non-destructive)
    backup_db()

    conn = connect_db()
    cursor = conn.cursor()

    # Ensure base tables exist
    ensure_tables(cursor)

    # Run migrations to add missing columns safely
    migrate_schema(cursor)

    conn.commit()

    # Print resulting schemas for verification
    print_table_info(cursor, "patients")
    print_table_info(cursor, "predictions")

    conn.close()
    print("✅ Database initialized and migrated at:", DB_FILE)

if __name__ == "__main__":
    init_db()