import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from core_pipeline.transformation_pipes import calculate_class_weights


class NeuralNetworkEngine:
    """
    Customer Conversion Engine
    """

    def __init__(
        self,
        activation="relu",
        optimizer="adam",
        learning_rate=0.001,
        epochs=30,
        batch_size=32,
        dropout_rate=0.3,
    ):

        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.epochs = epochs
        self.batch_size = batch_size

        self.dropout_rate = dropout_rate

        self.model = None

        self.history = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.class_weights = None

    # -----------------------------------------------------

    def split_data(self, X, y, test_size=0.2, random_state=42, max_samples=10000):
        y = pd.Series(y).reset_index(drop=True)
        valid_mask = y.notna().to_numpy()

        if sparse.issparse(X):
            X = X[valid_mask]
        else:
            X = np.asarray(X, dtype=np.float32)[valid_mask]

        y = y.loc[valid_mask].reset_index(drop=True)

        if y.empty:
            raise ValueError("Target column does not contain any valid values.")

        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(pd.factorize(y)[0], index=y.index)

        if len(y) > max_samples:
            sample_size = max_samples
            indices = np.arange(len(y))
            stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
            sampled_indices, _ = train_test_split(
                indices,
                train_size=sample_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify,
            )
            sampled_indices = np.sort(sampled_indices)
            if sparse.issparse(X):
                X = X[sampled_indices]
            else:
                X = X[sampled_indices]
            y = y.iloc[sampled_indices].reset_index(drop=True)

        if sparse.issparse(X):
            X = X.astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        y = y.astype(int).reset_index(drop=True)

        unique_classes = np.unique(y)

        if len(unique_classes) != 2:
            raise ValueError(
                "Neural Network module supports only binary classification. "
                f"Found classes: {unique_classes}"
            )

        counts = np.unique(y, return_counts=True)[1]
        if len(y) < 4 or np.min(counts) < 2:
            raise ValueError(
                "Neural networks need at least two rows for each binary target "
                "class so both the training and validation sets contain both classes."
            )

        # Stratification is safe after the class-count validation above and keeps
        # the validation metrics meaningful for imbalanced datasets.
        stratify = y

        # Stratified splitting needs at least one validation row per class. For
        # very small valid datasets, increase the validation fraction accordingly
        # instead of surfacing sklearn's low-level split error.
        effective_test_size = max(test_size, len(unique_classes) / len(y))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=effective_test_size,
            random_state=random_state,
            stratify=stratify,
        )

        self.class_weights = calculate_class_weights(self.y_train)

        return (self.X_train, self.X_test, self.y_train, self.y_test)

    # -----------------------------------------------------

    def get_optimizer(self):

        if self.optimizer.lower() == "sgd":

            return SGD(learning_rate=self.learning_rate)

        if self.optimizer.lower() == "rmsprop":

            return RMSprop(learning_rate=self.learning_rate)

        return Adam(learning_rate=self.learning_rate)

        # -----------------------------------------------------

    # PERCEPTRON MODEL
    # -----------------------------------------------------

    def build_perceptron(self, input_dim):
        """
        Build a single-layer Perceptron.
        """

        model = Sequential()

        model.add(Input(shape=(input_dim,)))

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=self.get_optimizer(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

        return model

    # -----------------------------------------------------
    # MULTI-LAYER PERCEPTRON
    # -----------------------------------------------------

    def build_mlp(self, input_dim, hidden_layers=[128, 64, 32]):
        """
        Build a configurable Multi-Layer Perceptron.
        """

        model = Sequential()

        model.add(Input(shape=(input_dim,)))

        # First hidden layer
        model.add(Dense(hidden_layers[0], activation=self.activation))

        model.add(Dropout(self.dropout_rate))

        # Remaining hidden layers
        for units in hidden_layers[1:]:

            model.add(Dense(units, activation=self.activation))

            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=self.get_optimizer(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

        return model

    # -----------------------------------------------------
    # MODEL SUMMARY
    # -----------------------------------------------------

    def summary(self):
        """
        Print model summary.
        """

        if self.model is None:
            print("No model has been built yet.")
            return

        self.model.summary()

    # -----------------------------------------------------
    # AVAILABLE ACTIVATIONS
    # -----------------------------------------------------

    @staticmethod
    def available_activations():

        return ["relu", "sigmoid", "tanh"]

    # -----------------------------------------------------
    # AVAILABLE OPTIMIZERS
    # -----------------------------------------------------

    @staticmethod
    def available_optimizers():

        return ["adam", "sgd", "rmsprop"]

        # -----------------------------------------------------

    # TRAIN MODEL
    # -----------------------------------------------------

    def train(self):
        """
        Train the neural network.
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
            class_weight=self.class_weights,
            callbacks=[early_stop],
            verbose=0,
        )

        return self.history

    # -----------------------------------------------------
    # EVALUATE MODEL
    # -----------------------------------------------------

    def evaluate(self):
        """
        Evaluate model performance.
        """

        if self.model is None:
            raise ValueError("Model has not been trained.")

        predictions = self.model.predict(self.X_test)

        predictions = (predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(self.y_test, predictions)

        report = classification_report(self.y_test, predictions, output_dict=True)

        matrix = confusion_matrix(self.y_test, predictions)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix,
        }

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    def predict(self, X):

        if self.model is None:
            raise ValueError("Model not available.")

        probabilities = self.model.predict(X)

        predictions = (probabilities > 0.5).astype(int)

        return predictions

    # -----------------------------------------------------
    # PREDICT PROBABILITY
    # -----------------------------------------------------

    def predict_probability(self, X):

        if self.model is None:
            raise ValueError("Model not available.")

        return self.model.predict(X)

    # -----------------------------------------------------
    # LOSS HISTORY
    # -----------------------------------------------------

    def get_loss_history(self):

        if self.history is None:
            return None

        return self.history.history["loss"]

    # -----------------------------------------------------
    # ACCURACY HISTORY
    # -----------------------------------------------------

    def get_accuracy_history(self):

        if self.history is None:
            return None

        return self.history.history["accuracy"]

    # -----------------------------------------------------
    # VALIDATION LOSS
    # -----------------------------------------------------

    def get_validation_loss(self):

        if self.history is None:
            return None

        return self.history.history["val_loss"]

    # -----------------------------------------------------
    # VALIDATION ACCURACY
    # -----------------------------------------------------

    def get_validation_accuracy(self):

        if self.history is None:
            return None

        return self.history.history["val_accuracy"]

        # -----------------------------------------------------

    # SAVE MODEL
    # -----------------------------------------------------

    def save_model(self, model_path="serialized_weights/neural_network.keras"):
        """
        Save trained model.
        """

        if self.model is None:
            raise ValueError("No trained model found.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)

        return model_path

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    def load_saved_model(self, model_path="serialized_weights/neural_network.keras"):
        """
        Load trained model.
        """

        self.model = load_model(model_path)

        return self.model

    # -----------------------------------------------------
    # PLOT TRAINING HISTORY
    # -----------------------------------------------------

    def plot_training_history(self):
        """
        Return matplotlib figures for Streamlit.
        """

        if self.history is None:
            return None, None

        # Accuracy Figure
        fig1, ax1 = plt.subplots(figsize=(7, 5))

        ax1.plot(self.history.history["accuracy"], label="Training Accuracy")

        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")

        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Loss Figure
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ax2.plot(self.history.history["loss"], label="Training Loss")

        ax2.plot(self.history.history["val_loss"], label="Validation Loss")

        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        return fig1, fig2


# ==========================================================
# MODULE ENTRY POINT
# ==========================================================


def run_neural_network_module(
    X,
    y,
    model_type="mlp",
    activation="relu",
    optimizer="adam",
    max_samples=8000,
    max_components=128,
):
    """
    Main interface used by app.py.
    """

    engine = NeuralNetworkEngine(
        activation=activation,
        optimizer=optimizer,
        epochs=12,
        batch_size=64,
    )

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("Neural-network features must be a two-dimensional matrix.")
    if X.shape[0] != len(y):
        raise ValueError(
            "The feature matrix and target column have different row counts. "
            "Please preprocess the dataset again."
        )
    if X.shape[1] == 0:
        raise ValueError("No usable features are available for neural-network training.")

    if sparse.issparse(X) and X.shape[1] > max_components:
        n_components = min(max_components, X.shape[1] - 1)
        if n_components >= 2:
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            X = reducer.fit_transform(X)

    engine.split_data(X, y, max_samples=max_samples)

    input_dim = X.shape[1]

    if model_type.lower() == "perceptron":

        engine.build_perceptron(input_dim)

    else:

        engine.build_mlp(input_dim)

    engine.train()

    results = engine.evaluate()

    accuracy_fig, loss_fig = engine.plot_training_history()

    return {
        "engine": engine,
        "results": results,
        "accuracy_figure": accuracy_fig,
        "loss_figure": loss_fig,
    }


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Customer Conversion Engine")
    print("Module A Loaded Successfully")
    print("=" * 60)
