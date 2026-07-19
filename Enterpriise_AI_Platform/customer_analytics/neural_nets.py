# customer_analytics/neural_nets.py

import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



def ensure_directory(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            return
        os.remove(path)
    os.makedirs(path, exist_ok=True)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERIALIZED_DIR = os.path.join(PROJECT_ROOT, "serialized_weights")
ensure_directory(SERIALIZED_DIR)
DEFAULT_MODEL_PATH = os.path.join(SERIALIZED_DIR, "customer_model.keras")
DEFAULT_SCALER_PATH = os.path.join(SERIALIZED_DIR, "scaler.pkl")


class NeuralNetworkEngine:

    def __init__(
        self,
        model_type="MLP",
        activation="relu",
        optimizer="adam",
        epochs=20,
        batch_size=32
    ):

        self.model_type = model_type
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.scaler = StandardScaler()

    ####################################################
    # Optimizer
    ####################################################

    def get_optimizer(self):

        if self.optimizer.lower() == "adam":
            return Adam()

        elif self.optimizer.lower() == "sgd":
            return SGD()

        elif self.optimizer.lower() == "rmsprop":
            return RMSprop()

        else:
            raise ValueError("Invalid Optimizer")

    ####################################################
    # Perceptron
    ####################################################

    def build_perceptron(self, input_dim):

        model = Sequential([
            Dense(
                1,
                activation="sigmoid",
                input_shape=(input_dim,)
            )
        ])

        model.compile(
            optimizer=self.get_optimizer(),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model

    ####################################################
    # Multi Layer Perceptron
    ####################################################

    def build_mlp(self, input_dim):

        model = Sequential([

            Dense(
                64,
                activation=self.activation,
                input_shape=(input_dim,)
            ),

            Dense(
                32,
                activation=self.activation
            ),

            Dense(
                16,
                activation=self.activation
            ),

            Dense(
                1,
                activation="sigmoid"
            )

        ])

        model.compile(

            optimizer=self.get_optimizer(),

            loss="binary_crossentropy",

            metrics=["accuracy"]

        )

        self.model = model

    ####################################################
    # Build Model
    ####################################################

    def build_model(self, input_dim):

        if self.model_type.lower() == "perceptron":

            self.build_perceptron(input_dim)

        else:

            self.build_mlp(input_dim)

    ####################################################
    # Train Model
    ####################################################

    def train(self, X, y):

        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        history = self.model.fit(

            X_train,

            y_train,

            validation_data=(X_test, y_test),

            epochs=self.epochs,

            batch_size=self.batch_size,

            verbose=1

        )

        prediction = self.model.predict(X_test)

        prediction = (prediction > 0.5).astype(int)

        accuracy = accuracy_score(

            y_test,

            prediction

        )

        return history, accuracy

    ####################################################
    # Predict New Customer
    ####################################################

    def predict(self, sample):

        if self.model is None:
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.load_model(DEFAULT_MODEL_PATH)
            else:
                raise Exception("No trained model found.")

        sample = self.scaler.transform(sample)

        prediction = self.model.predict(sample, verbose=0)

        if prediction[0][0] >= 0.5:
            return "Purchased"
        else:
            return "Not Purchased"

    ####################################################
    # Evaluate
    ####################################################

    def evaluate(self, X_test, y_test):

        X_test = self.scaler.transform(X_test)

        loss, accuracy = self.model.evaluate(

            X_test,

            y_test,

            verbose=0

        )

        return loss, accuracy

    ####################################################
    # Save Model
    ####################################################

    def save_model(self, path=DEFAULT_MODEL_PATH):
        self.model.save(path)

    ####################################################
    # Load Model
    ####################################################

    def load_model(self, path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(path)
