# 
"""Flask web app that serves the lung cancer risk Random Forest model.

Developed with assistance from Claude (Anthropic), Nov 2026:
app routes, input validation, and the HTML/CSS front end.

Run from this folder:
    python predict_lung_cancer_app.py
then open http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lung_cancer_random_forest.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# LabelEncoder sorts classes alphabetically during training, so F -> 0 and M -> 1.
GENDER_CODES = {"F": 0, "M": 1}

# (form field name, dataset column name, question shown to the user)
# The dataset codes every yes/no answer as 1 = No, 2 = Yes.
YES_NO_FIELDS = [
    ("smoking", "SMOKING", "Do you smoke?"),
    ("yellow_fingers", "YELLOW_FINGERS", "Do you have yellow fingers?"),
    ("anxiety", "ANXIETY", "Do you experience anxiety?"),
    ("peer_pressure", "PEER_PRESSURE", "Do you feel peer pressure?"),
    ("chronic_disease", "CHRONIC DISEASE", "Do you have a chronic disease?"),
    ("fatigue", "FATIGUE", "Do you feel fatigued?"),
    ("allergy", "ALLERGY", "Do you have allergies?"),
    ("wheezing", "WHEEZING", "Do you wheeze?"),
    ("alcohol", "ALCOHOL CONSUMING", "Do you consume alcohol?"),
    ("coughing", "COUGHING", "Do you cough?"),
    ("shortness_of_breath", "SHORTNESS OF BREATH", "Do you get short of breath?"),
    ("swallowing_difficulty", "SWALLOWING DIFFICULTY", "Do you have difficulty swallowing?"),
    ("chest_pain", "CHEST PAIN", "Do you have chest pain?"),
]


def build_features(form):
    """Validate the submitted form and return a one-row DataFrame for the model."""
    gender = form.get("gender")
    if gender not in GENDER_CODES:
        raise ValueError("Select a gender.")

    try:
        age = int(form.get("age", ""))
    except ValueError:
        raise ValueError("Enter your age as a whole number.")
    if not 1 <= age <= 120:
        raise ValueError("Age must be between 1 and 120.")

    row = {"GENDER": GENDER_CODES[gender], "AGE": age}
    for field, column, question in YES_NO_FIELDS:
        value = form.get(field)
        if value not in ("1", "2"):
            raise ValueError(f"Answer yes or no: {question}")
        row[column] = int(value)

    # Use the same column order the model was trained with.
    exact_names = {name.strip(): name for name in model.feature_names_in_}
    features = pd.DataFrame([row]).rename(columns=exact_names)

    # Use the same column order the model was trained with.
    return features[list(model.feature_names_in_)]


def render_page(form=None, result=None, error=None):
    return render_template(
        "index.html",
        yes_no_fields=[(field, question) for field, _, question in YES_NO_FIELDS],
        form=form or {},
        result=result,
        error=error,
    )


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = build_features(request.form)
    except ValueError as exc:
        return render_page(form=request.form, error=str(exc))

    high_risk = int(model.predict(features)[0]) == 1

    result = {
        "high_risk": high_risk,
        "label": "Lung cancer risk: High" if high_risk else "Lung cancer risk: Low",
    }
    return render_page(form=request.form, result=result)


if __name__ == "__main__":
    app.run(debug=True)
