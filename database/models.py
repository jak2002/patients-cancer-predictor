from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db   # make sure db = SQLAlchemy(app) is initialized in app.py

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(128), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(64))              # e.g. Doctor, Admin
    hospital = db.Column(db.String(128))
    specialization = db.Column(db.String(128))   # e.g. Oncology
    department = db.Column(db.String(128))       # e.g. Radiology
    experience_years = db.Column(db.Integer)
    status = db.Column(db.String(32), default="Active")
    profile_pic = db.Column(db.String(256))      # filename of uploaded picture
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    password_hash = db.Column(db.String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)