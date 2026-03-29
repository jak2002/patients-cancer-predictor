from db import connect_db, get_db_path

print("doctor_helpers connecting to:", get_db_path())

def get_doctor_dashboard(doctor_id):
    conn = connect_db()
    # Make rows accessible by column name
    conn.row_factory = __import__("sqlite3").Row
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT p.*,
               pr.prediction_result AS result,
               pr.created_at AS timestamp,
               pr.model AS model,
               pr.confidence AS confidence
        FROM patients p
        LEFT JOIN predictions pr
          ON p.id = pr.patient_id
          AND pr.doctor_id = ?
        ORDER BY p.id DESC, pr.created_at DESC
    """, (doctor_id,)).fetchall()

    conn.close()

    # Convert sqlite3.Row objects to plain dicts so template dot-lookup works
    return [dict(r) for r in rows]