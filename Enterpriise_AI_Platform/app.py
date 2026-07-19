import json
import os
from datetime import datetime
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from customer_analytics.neural_nets import NeuralNetworkEngine


def ensure_directory(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            return
        os.remove(path)
    os.makedirs(path, exist_ok=True)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERIALIZED_DIR = os.path.join(PROJECT_ROOT, "serialized_weights")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "analytical_reports")
MODEL_PATH = os.path.join(SERIALIZED_DIR, "customer_model.keras")
SCALER_PATH = os.path.join(SERIALIZED_DIR, "scaler.pkl")
TRAINING_REPORT_PATH = os.path.join(REPORTS_DIR, "training_results.json")
PREDICTION_REPORT_PATH = os.path.join(REPORTS_DIR, "prediction_results.json")

ensure_directory(SERIALIZED_DIR)
ensure_directory(REPORTS_DIR)

st.set_page_config(
    page_title="Enterprise AI Platform",
    layout="wide"
)

st.title("Customer Conversion Engine")
st.write("Neural Network Based Customer Purchase Prediction")

if "trained" not in st.session_state:
    st.session_state["trained"] = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

# ===============================
# Upload Dataset
# ===============================

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape :", df.shape)

    st.write("Columns")

    st.write(df.columns.tolist())

    # ===============================
    # Target Selection
    # ===============================

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # ===============================
    # Feature Selection
    # ===============================

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )

    if len(feature_columns) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    X = df[feature_columns]
    y = df[target]

    # ===============================
    # Convert Categorical Columns
    # ===============================

    X = pd.get_dummies(X)

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    # ===============================
    # Sidebar Options
    # ===============================

    st.sidebar.header("Model Configuration")

    model_type = st.sidebar.selectbox(
        "Model",
        ["MLP", "Perceptron"]
    )

    activation = st.sidebar.selectbox(
        "Activation Function",
        ["relu", "sigmoid", "tanh"]
    )

    optimizer = st.sidebar.selectbox(
        "Optimizer",
        ["adam", "sgd", "rmsprop"]
    )

    epochs = st.sidebar.slider(
        "Epochs",
        5,
        100,
        20
    )

    batch_size = st.sidebar.slider(
        "Batch Size",
        8,
        128,
        32
    )

    # ===============================
    # Train Button
    # ===============================

    if st.button("Train Model"):

        with st.spinner("Training Model..."):

            engine = NeuralNetworkEngine(
                model_type=model_type,
                activation=activation,
                optimizer=optimizer,
                epochs=epochs,
                batch_size=batch_size
            )

            engine.build_model(X.shape[1])
            history, accuracy = engine.train(X, y)

            engine.save_model(MODEL_PATH)
            joblib.dump(engine.scaler, SCALER_PATH)

            training_report = {
                "model_type": model_type,
                "activation": activation,
                "optimizer": optimizer,
                "epochs": epochs,
                "batch_size": batch_size,
                "accuracy": float(accuracy),
                "history": {
                    "accuracy": [float(value) for value in history.history.get("accuracy", [])],
                    "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
                    "loss": [float(value) for value in history.history.get("loss", [])],
                    "val_loss": [float(value) for value in history.history.get("val_loss", [])],
                },
                "saved_model_path": MODEL_PATH,
                "saved_scaler_path": SCALER_PATH,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

            with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as report_file:
                json.dump(training_report, report_file, indent=2)

        st.success("Training Completed")

        st.metric(
            "Model Accuracy",
            f"{accuracy*100:.2f}%"
        )

        st.session_state["trained"] = True

        # ===============================
        # Accuracy Plot
        # ===============================

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        ax1.plot(
            history.history["accuracy"],
            label="Train Accuracy"
        )

        ax1.plot(
            history.history["val_accuracy"],
            label="Validation Accuracy"
        )

        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        st.pyplot(fig1)

        # ===============================
        # Loss Plot
        # ===============================

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.plot(
            history.history["loss"],
            label="Train Loss"
        )

        ax2.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        st.pyplot(fig2)

    if st.session_state.get("trained", False):

        st.subheader("Predict Customer")

        user_input = []

        for col in X.columns:

            value = st.number_input(
                col,
                value=float(X[col].mean())
            )

            user_input.append(value)

        if st.button("Predict"):

            engine = NeuralNetworkEngine()
            engine.load_model(MODEL_PATH)
            engine.scaler = joblib.load(SCALER_PATH)

            sample = np.array(user_input).reshape(1, -1)

            result = engine.predict(sample)

            prediction_report = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "prediction": result,
                "features": [float(value) for value in user_input]
            }

            with open(PREDICTION_REPORT_PATH, "a", encoding="utf-8") as report_file:
                report_file.write(json.dumps(prediction_report) + "\n")

            if result == "Purchased":
                st.success("Prediction : Purchased")
            else:
                st.error("Prediction : Not Purchased")

        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button(
                    "Download Trained Model",
                    data=f,
                    file_name="customer_model.keras",
                    mime="application/octet-stream"
                )
