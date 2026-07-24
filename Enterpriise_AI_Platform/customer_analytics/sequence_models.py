import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


class SequenceModelEngine:

    def __init__(
        self,
        sequence_length=10,
        learning_rate=0.001,
        epochs=8,
        batch_size=64,
        dropout_rate=0.2,
        l1_penalty=0.0,
        l2_penalty=0.0,
    ):

        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.model = None
        self.history = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.isolation_forest = None

    def _regularizer(self):
        return regularizers.l1_l2(l1=self.l1_penalty, l2=self.l2_penalty)

        # --------------------------------------------------

    # CREATE TIME SEQUENCES
    # --------------------------------------------------

    def create_sequences(self, data, target):
        if sparse.issparse(data):
            data = data.toarray()
        else:
            data = np.asarray(data)

        target = np.asarray(target, dtype=np.float32).reshape(-1)

        if data.ndim != 2:
            raise ValueError("Sequence-model features must be a two-dimensional matrix.")
        if data.shape[0] != len(target):
            raise ValueError(
                "The feature matrix and target column have different row counts. "
                "Please preprocess the dataset again."
            )
        if data.shape[1] == 0:
            raise ValueError("No usable features are available for sequence-model training.")
        if not np.isfinite(target).all():
            raise ValueError(
                "The selected target column contains missing or non-finite values. "
                "Choose a complete numeric target or clean the data first."
            )

        # A batch-normalized recurrent model needs at least two training windows,
        # plus one held-out window for validation.
        if data.shape[0] < self.sequence_length + 3:
            raise ValueError(
                "Not enough samples to create sequences. "
                f"Need at least {self.sequence_length + 3} rows for a sequence "
                f"length of {self.sequence_length}."
            )

        X = []
        y = []

        for i in range(data.shape[0] - self.sequence_length):

            X.append(data[i : i + self.sequence_length])

            y.append(target[i + self.sequence_length])

        return (np.array(X), np.array(y))

        # --------------------------------------------------

    # TRAIN TEST SPLIT
    # --------------------------------------------------

    def split_data(self, X, y, test_size=0.2):

        if len(X) < 2:
            raise ValueError(
                "Not enough generated sequences to split into train and test sets."
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        return (self.X_train, self.X_test, self.y_train, self.y_test)

        # --------------------------------------------------

    # BUILD LSTM
    # --------------------------------------------------

    def build_lstm(self, input_shape):

        model = Sequential()

        model.add(Input(shape=input_shape))

        model.add(
            LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=self._regularizer(),
                recurrent_regularizer=self._regularizer(),
            )
        )

        model.add(BatchNormalization())

        model.add(Dropout(self.dropout_rate))

        model.add(
            LSTM(
                32,
                kernel_regularizer=self._regularizer(),
                recurrent_regularizer=self._regularizer(),
            )
        )

        model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, kernel_regularizer=self._regularizer()))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model

        return model

        # --------------------------------------------------

    # BUILD GRU
    # --------------------------------------------------

    def build_gru(self, input_shape):

        model = Sequential()

        model.add(Input(shape=input_shape))

        model.add(
            GRU(
                64,
                return_sequences=True,
                kernel_regularizer=self._regularizer(),
                recurrent_regularizer=self._regularizer(),
            )
        )

        model.add(BatchNormalization())

        model.add(Dropout(self.dropout_rate))

        model.add(
            GRU(
                32,
                kernel_regularizer=self._regularizer(),
                recurrent_regularizer=self._regularizer(),
            )
        )

        model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, kernel_regularizer=self._regularizer()))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model

        return model

        # --------------------------------------------------

    # TRAIN MODEL
    # --------------------------------------------------

    def train(self):
        """
        Train the selected sequence model.
        """

        if self.model is None:
            raise ValueError("Build a model before training.")

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        return self.history

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X=None):

        if self.model is None:
            raise ValueError("Model has not been trained.")

        if X is None:
            X = self.X_test

        predictions = self.model.predict(X, verbose=0)

        return predictions.flatten()

        # --------------------------------------------------

    # MODEL EVALUATION
    # --------------------------------------------------

    def evaluate(self):

        predictions = self.predict()

        mse = mean_squared_error(self.y_test, predictions)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(self.y_test, predictions)

        return {"MSE": mse, "RMSE": rmse, "MAE": mae}

        # --------------------------------------------------

    # FUTURE FORECAST
    # --------------------------------------------------

    def forecast(self, last_sequence, steps=10):

        sequence = np.array(last_sequence)

        forecasts = []

        for _ in range(steps):

            prediction = self.model.predict(
                sequence.reshape(1, sequence.shape[0], sequence.shape[1]), verbose=0
            )[0][0]

            forecasts.append(prediction)

            sequence = np.vstack([sequence[1:], [prediction] * sequence.shape[1]])

        return np.array(forecasts)

        # --------------------------------------------------

    # ANOMALY DETECTION
    # --------------------------------------------------

    def train_isolation_forest(self, X, contamination=0.05):

        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=42
        )

        self.isolation_forest.fit(X)

        return self.isolation_forest

    def detect_anomalies(self, X):

        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not trained.")

        predictions = self.isolation_forest.predict(X)

        anomaly_index = np.where(predictions == -1)[0]

        return anomaly_index

    def anomaly_scores(self, X):
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not trained.")
        return self.isolation_forest.decision_function(X)

        # --------------------------------------------------

    # COMPARE ACTUAL VS PREDICTED
    # --------------------------------------------------

    def compare_predictions(self):

        predictions = self.predict()

        return {"Actual": self.y_test, "Predicted": predictions}

        # --------------------------------------------------

    # TRAINING HISTORY PLOTS
    # --------------------------------------------------

    def plot_training_history(self):
        """
        Return training and validation loss plots.
        """

        if self.history is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.history.history["loss"], label="Training Loss")

        ax.plot(self.history.history["val_loss"], label="Validation Loss")

        ax.set_title("Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        return fig

        # --------------------------------------------------

    # ACTUAL VS PREDICTED
    # --------------------------------------------------

    def plot_predictions(self):

        predictions = self.predict()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.y_test, label="Actual")

        ax.plot(predictions, label="Predicted")

        ax.set_title("Actual vs Predicted")

        ax.set_xlabel("Samples")

        ax.set_ylabel("Target")

        ax.legend()

        return fig

        # --------------------------------------------------

    # ANOMALY VISUALIZATION
    # --------------------------------------------------

    def plot_anomalies(self, X):

        anomaly_idx = self.detect_anomalies(X)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(np.arange(len(X)), X[:, 0], label="Data")

        ax.scatter(anomaly_idx, X[anomaly_idx, 0], marker="x", s=80, label="Anomalies")

        ax.set_title("Isolation Forest Anomalies")

        ax.legend()

        return fig

        # --------------------------------------------------

    # SAVE MODEL
    # --------------------------------------------------

    def save_model(self, path="serialized_weights/sequence_model.keras"):

        if self.model is None:
            raise ValueError("No trained model available.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)

        return path

        # --------------------------------------------------

    # LOAD MODEL
    # --------------------------------------------------

    def load_saved_model(self, path="serialized_weights/sequence_model.keras"):

        self.model = load_model(path)

        return self.model

        # --------------------------------------------------

    # SAVE ISOLATION FOREST
    # --------------------------------------------------

    def save_isolation_forest(self, path="serialized_weights/isolation_forest.pkl"):

        if self.isolation_forest is None:
            raise ValueError("Isolation Forest has not been trained.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(self.isolation_forest, path)

        return path

        # --------------------------------------------------

    # LOAD ISOLATION FOREST
    # --------------------------------------------------

    def load_isolation_forest(self, path="serialized_weights/isolation_forest.pkl"):

        self.isolation_forest = joblib.load(path)

        return self.isolation_forest


# ==========================================================
# RUN SEQUENCE MODEL
# ==========================================================


def run_sequence_models(
    X,
    y,
    model_type="lstm",
    sequence_length=10,
    l1_penalty=0.0,
    l2_penalty=0.0,
    anomaly_contamination=0.05,
    max_samples=2500,
    max_components=32,
):
    """
    Unified entry point for Sequence Models.
    """

    engine = SequenceModelEngine(
        sequence_length=sequence_length, l1_penalty=l1_penalty, l2_penalty=l2_penalty
    )

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("Sequence-model features must be a two-dimensional matrix.")
    if X.shape[0] != len(y):
        raise ValueError(
            "The feature matrix and target column have different row counts. "
            "Please preprocess the dataset again."
        )
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")

    if sparse.issparse(X) and X.shape[1] > max_components:
        n_components = min(max_components, X.shape[1] - 1)
        if n_components >= 2:
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            X = reducer.fit_transform(X)

    if X.shape[0] > max_samples:
        X = X[:max_samples]
        y = np.asarray(y)[:max_samples]

    if sparse.issparse(X):
        X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    X_seq, y_seq = engine.create_sequences(X, y)

    engine.split_data(X_seq, y_seq)

    input_shape = (engine.X_train.shape[1], engine.X_train.shape[2])

    if model_type.lower() == "gru":

        engine.build_gru(input_shape)

    else:

        engine.build_lstm(input_shape)

    engine.train()

    metrics = engine.evaluate()

    prediction_plot = engine.plot_predictions()

    history_plot = engine.plot_training_history()

    sequence_vectors = X_seq.reshape(X_seq.shape[0], -1)
    engine.train_isolation_forest(sequence_vectors, contamination=anomaly_contamination)
    anomaly_idx = engine.detect_anomalies(sequence_vectors)
    anomaly_scores = engine.anomaly_scores(sequence_vectors)
    anomaly_plot = engine.plot_anomalies(sequence_vectors)

    return {
        "engine": engine,
        "metrics": metrics,
        "prediction_plot": prediction_plot,
        "history_plot": history_plot,
        "anomaly_indices": anomaly_idx,
        "anomaly_scores": anomaly_scores,
        "anomaly_plot": anomaly_plot,
    }

    # ==========================================================


# COMPARE LSTM AND GRU
# ==========================================================


def compare_models(X, y, sequence_length=10):

    lstm = run_sequence_models(X, y, "lstm", sequence_length)

    gru = run_sequence_models(X, y, "gru", sequence_length)

    results = {"LSTM": lstm["metrics"], "GRU": gru["metrics"]}

    if results["LSTM"]["RMSE"] < results["GRU"]["RMSE"]:

        best_model = "LSTM"

    else:

        best_model = "GRU"

    return {
        "results": results,
        "best_model": best_model,
        "lstm_engine": lstm["engine"],
        "gru_engine": gru["engine"],
    }

    # ==========================================================


# SUMMARY REPORT
# ==========================================================


def sequence_summary(comparison):

    report = {
        "Best Model": comparison["best_model"],
        "LSTM RMSE": comparison["results"]["LSTM"]["RMSE"],
        "GRU RMSE": comparison["results"]["GRU"]["RMSE"],
        "LSTM MAE": comparison["results"]["LSTM"]["MAE"],
        "GRU MAE": comparison["results"]["GRU"]["MAE"],
    }

    return report


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    np.random.seed(42)

    X = np.random.rand(500, 5)

    y = np.random.rand(500)

    comparison = compare_models(X, y)

    print("=" * 60)
    print("Sequence Models Module")
    print("=" * 60)

    print(sequence_summary(comparison))
