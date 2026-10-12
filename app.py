from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import re
import joblib
import numpy as np
import pandas as pd


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

CORS(app)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

USER_DATA_DIR = os.path.join(
    BASE_DIR,
    "user_data"
)

USER_DATA_FILE = os.path.join(
    USER_DATA_DIR,
    "user_health_data.csv"
)

os.makedirs(
    USER_DATA_DIR,
    exist_ok=True
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {

    "heart_disease":
        "heart_disease.joblib",

    "diabetes":
        "diabetes.joblib",

    "hypertension":
        "hypertension.joblib",

    "obesity":
        "obesity.joblib",

    "sleep":
        "sleep.joblib"
}


MODELS = {}


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    MODELS.clear()

    for disease, filename in MODEL_FILES.items():

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(path):

            print(
                f"Model not found: {path}"
            )

            continue

        try:

            MODELS[disease] = joblib.load(
                path
            )

            print(
                f"Loaded model: {disease}"
            )

        except Exception as error:

            print(
                f"Failed to load {disease}: {error}"
            )


load_models()


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(value):

    return str(value).strip().lower()


def normalize_gender(value):

    value = normalize_text(value)

    if value in [
        "male",
        "m"
    ]:
        return "Male"

    if value in [
        "female",
        "f"
    ]:
        return "Female"

    return value.title()


def normalize_yes_no(value):

    value = normalize_text(value)

    if value in [
        "yes",
        "y",
        "true",
        "1"
    ]:
        return "Yes"

    return "No"


def normalize_stress(value):

    value = normalize_text(value)

    if value == "low":
        return "Low"

    if value in [
        "medium",
        "moderate"
    ]:
        return "Moderate"

    if value == "high":
        return "High"

    try:

        number = float(value)

        if number <= 3:
            return "Low"

        if number <= 6:
            return "Moderate"

        return "High"

    except:

        return "Moderate"


def normalize_exercise(value):

    value = normalize_text(value)

    if value in [
        "low",
        "inactive",
        "low activity"
    ]:
        return "Low"

    if value in [
        "medium",
        "moderate",
        "moderately active"
    ]:
        return "Moderate"

    if value in [
        "high",
        "highly active"
    ]:
        return "High"

    return value.title()


# ============================================================
# BLOOD PRESSURE
# ============================================================

def parse_blood_pressure(value):

    text = str(value).strip()

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        text
    )

    if not match:

        raise ValueError(
            "Blood pressure must be in format like 120/80."
        )

    systolic = float(
        match.group(1)
    )

    diastolic = float(
        match.group(2)
    )

    if systolic <= 0 or diastolic <= 0:

        raise ValueError(
            "Blood pressure values must be greater than zero."
        )

    return systolic, diastolic


# ============================================================
# BMI
# ============================================================

def calculate_bmi(
    height_cm,
    weight
):

    height_m = (
        height_cm / 100
    )

    if height_m <= 0:

        raise ValueError(
            "Height must be greater than zero."
        )

    return weight / (
        height_m * height_m
    )


# ============================================================
# RISK LEVEL
# ============================================================

def binary_risk_level(
    probability
):

    if probability < 0.35:

        return "Low"

    if probability < 0.65:

        return "Moderate"

    return "High"


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_request(data):

    required_fields = [

        "age",
        "gender",
        "height",
        "weight",
        "blood_pressure",
        "glucose",
        "cholesterol",
        "sleep_hours",
        "stress_level",
        "exercise_level",
        "smoking",
        "alcohol"
    ]

    missing = [

        field
        for field in required_fields

        if field not in data
        or data[field] is None
        or data[field] == ""
    ]

    if missing:

        raise ValueError(
            "Missing fields: "
            + ", ".join(missing)
        )

    try:

        age = float(
            data["age"]
        )

        height = float(
            data["height"]
        )

        weight = float(
            data["weight"]
        )

        glucose = float(
            data["glucose"]
        )

        cholesterol = float(
            data["cholesterol"]
        )

        sleep_hours = float(
            data["sleep_hours"]
        )

    except ValueError:

        raise ValueError(
            "Age, height, weight, glucose, cholesterol and sleep hours must be numeric."
        )

    if age < 1 or age > 120:

        raise ValueError(
            "Age must be between 1 and 120."
        )

    if height <= 0:

        raise ValueError(
            "Height must be greater than zero."
        )

    if weight <= 0:

        raise ValueError(
            "Weight must be greater than zero."
        )

    if glucose < 0:

        raise ValueError(
            "Glucose cannot be negative."
        )

    if cholesterol < 0:

        raise ValueError(
            "Cholesterol cannot be negative."
        )

    if sleep_hours < 0 or sleep_hours > 24:

        raise ValueError(
            "Sleep hours must be between 0 and 24."
        )

    systolic, diastolic = parse_blood_pressure(
        data["blood_pressure"]
    )

    return {

        "age": age,

        "gender":
            normalize_gender(
                data["gender"]
            ),

        "height_cm": height,

        "weight": weight,

        "glucose": glucose,

        "cholesterol": cholesterol,

        "sleep_hours": sleep_hours,

        "stress_level":
            normalize_stress(
                data["stress_level"]
            ),

        "exercise_level":
            normalize_exercise(
                data["exercise_level"]
            ),

        "smoking":
            normalize_yes_no(
                data["smoking"]
            ),

        "alcohol":
            normalize_yes_no(
                data["alcohol"]
            ),

        "systolic_bp": systolic,

        "diastolic_bp": diastolic
    }


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(data):

    bmi = calculate_bmi(
        data["height_cm"],
        data["weight"]
    )

    return {

        "age":
            data["age"],

        "gender":
            data["gender"],

        "height":
            data["height_cm"],

        "height_cm":
            data["height_cm"],

        "weight":
            data["weight"],

        "bmi":
            bmi,

        "blood_pressure":
            data["systolic_bp"],

        "systolic_bp":
            data["systolic_bp"],

        "diastolic_bp":
            data["diastolic_bp"],

        "cholesterol":
            data["cholesterol"],

        "glucose":
            data["glucose"],

        "sleep_hours":
            data["sleep_hours"],

        "stress_level":
            data["stress_level"],

        "exercise_level":
            data["exercise_level"],

        "smoking":
            data["smoking"],

        "alcohol":
            data["alcohol"]
    }


# ============================================================
# GET PROBABILITY
# ============================================================

def get_probability(
    package,
    model_input
):

    model = package["model"]

    if not hasattr(
        model,
        "predict_proba"
    ):

        return None

    probabilities = model.predict_proba(
        model_input
    )

    classes = list(
        model.classes_
    )

    # Binary classification
    if len(classes) == 2:

        if 1 in classes:

            index = classes.index(1)

            return float(
                probabilities[0][index]
            )

        # If labels are strings
        if "1" in classes:

            index = classes.index("1")

            return float(
                probabilities[0][index]
            )

        return float(
            max(probabilities[0])
        )

    # Multiclass
    return float(
        max(probabilities[0])
    )


# ============================================================
# PREDICT ONE DISEASE
# ============================================================

def predict_disease(
    disease,
    input_data
):

    if disease not in MODELS:

        return {

            "prediction": None,

            "probability": None,

            "probability_type":
                "unavailable",

            "risk_level":
                "Unavailable",

            "error":
                "Model is not available."
        }

    package = MODELS[disease]

    model = package["model"]

    required_features = package[
        "features"
    ]

    # Check model requirements
    for feature in required_features:

        if feature not in input_data:

            return {

                "prediction": None,

                "probability": None,

                "probability_type":
                    "unavailable",

                "risk_level":
                    "Unavailable",

                "error":
                    (
                        f"Model requires feature "
                        f"'{feature}', but it is not "
                        f"available from the frontend."
                    )
            }

    model_input = pd.DataFrame(
        [
            {
                feature:
                    input_data[feature]

                for feature
                in required_features
            }
        ]
    )

    try:

        prediction = model.predict(
            model_input
        )[0]

        probability = get_probability(
            package,
            model_input
        )

        classes = package[
            "classes"
        ]

        # ----------------------------------------------------
        # BINARY
        # ----------------------------------------------------

        if len(classes) == 2:

            prediction_text = str(
                prediction
            )

            if prediction_text == "1":

                prediction_text = "Yes"

            elif prediction_text == "0":

                prediction_text = "No"

            if probability is not None:

                percentage = round(
                    probability * 100,
                    2
                )

                risk = binary_risk_level(
                    probability
                )

            else:

                percentage = None

                risk = (
                    "High"
                    if prediction_text == "Yes"
                    else "Low"
                )

        # ----------------------------------------------------
        # MULTICLASS
        # ----------------------------------------------------

        else:

            prediction_text = str(
                prediction
            )

            if probability is not None:

                percentage = round(
                    probability * 100,
                    2
                )

            else:

                percentage = None

            if disease == "obesity":

                if prediction_text in [
                    "Insufficient_Weight",
                    "Normal_Weight"
                ]:

                    risk = "Low"

                elif prediction_text in [
                    "Overweight_Level_I",
                    "Overweight_Level_II"
                ]:

                    risk = "Moderate"

                else:

                    risk = "High"

            elif disease == "sleep":

                if prediction_text == "No Disorder":

                    risk = "Low"

                elif prediction_text == "Insomnia":

                    risk = "Moderate"

                elif prediction_text == "Sleep Apnea":

                    risk = "High"

                else:

                    risk = "Moderate"

            else:

                risk = (
                    binary_risk_level(
                        probability
                    )
                    if probability is not None
                    else "Unavailable"
                )

        return {

            "prediction":
                prediction_text,

            "probability":
                percentage,

            "probability_type":
                (
                    "model_estimated"
                    if probability is not None
                    else "class_prediction"
                ),

            "risk_level":
                risk,

            "error":
                None
        }

    except Exception as error:

        return {

            "prediction": None,

            "probability": None,

            "probability_type":
                "unavailable",

            "risk_level":
                "Unavailable",

            "error":
                str(error)
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "healthy",
            "service":
                "Disease Prediction System API"
        }
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        if not request.is_json:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "Request must contain JSON data."
                }
            ), 400

        data = request.get_json()

        validated = validate_request(
            data
        )

        model_input = create_model_input(
            validated
        )

        predictions = {

            "heart_disease":
                predict_disease(
                    "heart_disease",
                    model_input
                ),

            "diabetes":
                predict_disease(
                    "diabetes",
                    model_input
                ),

            "hypertension":
                predict_disease(
                    "hypertension",
                    model_input
                ),

            "obesity":
                predict_disease(
                    "obesity",
                    model_input
                ),

            "sleep":
                predict_disease(
                    "sleep",
                    model_input
                )
        }

        bmi = model_input[
            "bmi"
        ]

        recommendations = []

        if bmi >= 30:

            recommendations.append(
                "Consider discussing weight and lifestyle management with a qualified healthcare professional."
            )

        if model_input[
            "sleep_hours"
        ] < 6:

            recommendations.append(
                "Consider maintaining a regular sleep schedule and discussing persistent sleep problems with a qualified healthcare professional."
            )

        if model_input[
            "exercise_level"
        ] == "Low":

            recommendations.append(
                "Consider gradually increasing regular physical activity as appropriate for you."
            )

        if model_input[
            "smoking"
        ] == "Yes":

            recommendations.append(
                "Avoiding tobacco use can support overall health."
            )

        if model_input[
            "alcohol"
        ] == "Yes":

            recommendations.append(
                "Consider limiting alcohol consumption."
            )

        high_risk = any(

            result.get(
                "risk_level"
            ) == "High"

            for result
            in predictions.values()
        )

        if high_risk:

            recommendations.append(
                "Some results are in a higher-risk category. Consider consulting a qualified healthcare professional for proper evaluation."
            )

        return jsonify(

            {
                "success": True,

                "bmi":
                    round(
                        bmi,
                        2
                    ),

                "predictions":
                    predictions,

                "recommendations":
                    recommendations,

                "disclaimer":
                    (
                        "This system is an educational "
                        "machine-learning project for "
                        "health risk screening. It is not "
                        "a medical diagnostic tool. "
                        "Predictions are model-based "
                        "estimates and should not replace "
                        "professional medical advice. "
                        "Consult a qualified healthcare "
                        "professional for proper evaluation."
                    )
            }
        )

    except ValueError as error:

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 400

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        return jsonify(
            {
                "success": False,
                "error":
                    "An unexpected server error occurred."
            }
        ), 500


# ============================================================
# SAVE USER DATA
# ============================================================

@app.route(
    "/api/save-user",
    methods=["POST"]
)
def save_user():

    try:

        if not request.is_json:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "JSON data required."
                }
            ), 400

        data = request.get_json()

        validated = validate_request(
            data
        )

        model_input = create_model_input(
            validated
        )

        record = {

            "age":
                model_input["age"],

            "gender":
                model_input["gender"],

            "height_cm":
                model_input["height_cm"],

            "weight":
                model_input["weight"],

            "bmi":
                model_input["bmi"],

            "blood_pressure":
                data["blood_pressure"],

            "glucose":
                model_input["glucose"],

            "cholesterol":
                model_input["cholesterol"],

            "sleep_hours":
                model_input["sleep_hours"],

            "stress_level":
                model_input["stress_level"],

            "exercise_level":
                model_input["exercise_level"],

            "smoking":
                model_input["smoking"],

            "alcohol":
                model_input["alcohol"]
        }

        new_df = pd.DataFrame(
            [record]
        )

        if os.path.exists(
            USER_DATA_FILE
        ):

            old_df = pd.read_csv(
                USER_DATA_FILE
            )

            final_df = pd.concat(
                [
                    old_df,
                    new_df
                ],
                ignore_index=True
            )

        else:

            final_df = new_df

        final_df.to_csv(
            USER_DATA_FILE,
            index=False
        )

        return jsonify(
            {
                "success": True,
                "message":
                    "User health data saved successfully."
            }
        )

    except ValueError as error:

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 400

    except Exception as error:

        print(
            f"Save user error: {error}"
        )

        return jsonify(
            {
                "success": False,
                "error":
                    "Unable to save user data."
            }
        ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "DISEASE PREDICTION SYSTEM API"
    )
    print("=" * 60)

    print(
        "Models loaded:",
        list(MODELS.keys())
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )