
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "laptop_performance_model.joblib"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scaler.joblib"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "label_encoder.joblib"
)


# Load trained ML files
model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

label_encoder = joblib.load(
    ENCODER_PATH
)


FEATURES = [

    "ram_gb",

    "cpu_usage_percent",

    "ram_usage_percent",

    "disk_usage_percent",

    "cpu_temperature",

    "gpu_usage_percent",

    "battery_health_percent",

    "storage_capacity_gb",

    "free_storage_gb",

    "laptop_age_years",

    "daily_usage_hours",

    "running_processes"

]


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error":
                "No input data received."
            }), 400


        values = []

        for feature in FEATURES:

            if feature not in data:

                return jsonify({
                    "error":
                    f"Missing input: {feature}"
                }), 400

            values.append(
                float(data[feature])
            )


        input_df = pd.DataFrame(
            [values],
            columns=FEATURES
        )


        # Apply same scaling
        scaled_input = scaler.transform(
            input_df
        )


        # Prediction
        prediction = model.predict(
            scaled_input
        )


        # Convert encoded value
        predicted_class = (
            label_encoder
            .inverse_transform(
                prediction
            )[0]
        )


        # Confidence
        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    scaled_input
                )[0]
            )

            confidence = round(
                float(
                    max(probabilities)
                ) * 100,
                2
            )


        return jsonify({

            "prediction":
                predicted_class,

            "confidence":
                confidence

        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
