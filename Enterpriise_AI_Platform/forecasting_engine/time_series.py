import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

from prophet import Prophet

warnings.filterwarnings("ignore")


class TimeSeriesEngine:

    def __init__(self):

        self.series = None

        self.frequency = None

        self.arima_model = None

        self.prophet_model = None

        self.arima_forecast = None

        self.prophet_forecast = None

        self.decomposition = None

        self.metrics = {}

    # --------------------------------------------------
    # LOAD SERIES
    # --------------------------------------------------

    def load_series(self, dataframe, date_column, target_column, frequency="D"):

        if date_column not in dataframe.columns:
            raise ValueError(
                f"Date column '{date_column}' was not found in the dataframe."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found in the dataframe."
            )

        df = dataframe.copy()

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, target_column])
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce").fillna(0)

        # Aggregate duplicate date timestamps by frequency before setting index
        grouped = df.groupby(pd.Grouper(key=date_column, freq=frequency))[
            target_column
        ].sum()

        # This app forecasts aggregated activity (for example, daily revenue or
        # order value). An interval without a row means no recorded activity,
        # not a repeat of the previous interval's total. Forward-filling here
        # created artificial plateaus and could make ARIMA and Prophet diverge
        # for the wrong reason.
        self.series = grouped.asfreq(frequency, fill_value=0.0).astype(float)

        if self.series.dropna().empty:
            raise ValueError("Time series data is empty after date aggregation.")
        if len(self.series) < 8:
            raise ValueError(
                "Forecasting needs at least 8 aggregated time periods. "
                "Choose a coarser frequency or provide more dated records."
            )
        if self.series.nunique() < 2:
            raise ValueError(
                "The selected target has no variation after aggregation, so it "
                "cannot be forecast reliably."
            )

        self.frequency = frequency

        return self.series

    # --------------------------------------------------
    # TREND + SEASONALITY
    # --------------------------------------------------

    def decompose_series(self, period=None):
        if len(self.series) < 4:
            self.decomposition = None
            return None

        if period is None:
            if self.frequency == "D":
                period = 7
            elif self.frequency == "W":
                period = 4
            elif self.frequency == "M":
                period = 12
            else:
                period = 7

        if len(self.series) < 2 * period:
            period = max(2, len(self.series) // 2)

        try:
            self.decomposition = seasonal_decompose(
                self.series, model="additive", period=period
            )
        except Exception:
            self.decomposition = None

        return self.decomposition

    # --------------------------------------------------
    # ROLLING MEAN
    # --------------------------------------------------

    def rolling_statistics(self, window=30):

        window = min(window, max(1, len(self.series)))

        rolling_mean = self.series.rolling(window).mean()

        rolling_std = self.series.rolling(window).std()

        return {"Rolling Mean": rolling_mean, "Rolling Std": rolling_std}

    # --------------------------------------------------
    # AUTOCORRELATION
    # --------------------------------------------------

    def autocorrelation(self, lags=40):

        lags = min(lags, max(1, len(self.series) - 1))

        values = acf(self.series, nlags=lags)

        return values

    def validation_table(self, periods=30, order=(5, 1, 0)):
        if len(self.series) < 4:
            return pd.DataFrame(columns=["Model", "RMSE", "MAE", "MAPE"])

        periods = min(periods, max(1, len(self.series) // 3))
        if (len(self.series) - periods) < 4:
            periods = max(1, len(self.series) - 4)

        train_series = self.series.iloc[:-periods]
        test_series = self.series.iloc[-periods:]

        arima_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        prophet_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

        p, d, q = order
        if len(train_series) <= (p + d):
            p = max(1, len(train_series) - d - 1)
            order = (max(1, p), d, q)

        try:
            arima_model = ARIMA(train_series, order=order).fit()
            arima_pred = arima_model.forecast(periods)
            arima_pred_values = self._clamp_to_series_scale(arima_pred.values)
            arima_metrics = self.evaluate_forecast(
                test_series.values, arima_pred_values
            )
        except Exception:
            pass

        try:
            prophet_train = pd.DataFrame(
                {"ds": train_series.index, "y": train_series.values}
            )
            prophet_model = Prophet()
            prophet_model.fit(prophet_train)
            future = prophet_model.make_future_dataframe(
                periods=periods, freq=self.frequency
            )
            prophet_forecast = prophet_model.predict(future)
            prophet_pred = self._clamp_to_series_scale(
                prophet_forecast["yhat"].tail(periods).values
            )
            prophet_metrics = self.evaluate_forecast(test_series.values, prophet_pred)
        except Exception:
            pass

        return pd.DataFrame(
            [
                {"Model": "ARIMA", **arima_metrics},
                {"Model": "Prophet", **prophet_metrics},
            ]
        )

    # --------------------------------------------------
    # BUILD ARIMA
    # --------------------------------------------------

    def build_arima(self, order=(5, 1, 0)):

        p, d, q = order
        if len(self.series) <= (p + d):
            p = max(1, len(self.series) - d - 1)
            order = (max(1, p), d, q)

        try:
            self.arima_model = ARIMA(self.series, order=order).fit()
        except Exception:
            self.arima_model = ARIMA(self.series, order=(1, 1, 0)).fit()

        return self.arima_model

    # --------------------------------------------------
    # ARIMA FORECAST
    # --------------------------------------------------

    def forecast_arima(self, periods=30):

        if self.arima_model is None:

            raise ValueError("Train ARIMA first.")

        self.arima_forecast = self.arima_model.forecast(periods)

        return self.arima_forecast

    # --------------------------------------------------
    # BUILD PROPHET
    # --------------------------------------------------

    def build_prophet(self):

        df = pd.DataFrame({"ds": self.series.index, "y": self.series.values})

        self.prophet_model = Prophet()

        self.prophet_model.fit(df)

        return self.prophet_model

    # --------------------------------------------------
    # PROPHET FORECAST
    # --------------------------------------------------

    def forecast_prophet(self, periods=30):

        if self.prophet_model is None:

            raise ValueError("Train Prophet first.")

        future = self.prophet_model.make_future_dataframe(
            periods=periods, freq=self.frequency
        )

        forecast = self.prophet_model.predict(future)

        self.prophet_forecast = forecast

        return forecast

        # --------------------------------------------------

    # EVALUATION METRICS
    # --------------------------------------------------

    def evaluate_forecast(self, actual, predicted):

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        rmse = np.sqrt(mean_squared_error(actual, predicted))

        mae = mean_absolute_error(actual, predicted)

        non_zero_mask = np.abs(actual) > 1e-8
        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (actual[non_zero_mask] - predicted[non_zero_mask])
                        / actual[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

        return metrics

    def _clamp_to_series_scale(self, values):
        values = np.asarray(values, dtype=float)
        series_values = np.asarray(self.series.values, dtype=float)
        series_std = float(np.nanstd(series_values))
        if np.isnan(series_std) or series_std == 0:
            series_std = 1.0
        lower = float(np.nanmin(series_values) - 3 * series_std)
        upper = float(np.nanmax(series_values) + 3 * series_std)
        return np.clip(values, lower, upper)

    # --------------------------------------------------
    # COMPARE FORECASTS
    # --------------------------------------------------

    def compare_models(self, periods=30):

        if self.arima_model is None:
            self.build_arima()

        if self.prophet_model is None:
            self.build_prophet()

        arima = self.forecast_arima(periods)

        prophet = self.forecast_prophet(periods)

        # Compare values at the same timestamps. Using a positional tail can
        # silently compare different calendar dates for weekly/monthly series.
        forecast_index = arima.index
        prophet_by_date = prophet.set_index("ds")["yhat"]
        prophet_values = prophet_by_date.reindex(forecast_index).to_numpy()
        if np.isnan(prophet_values).any():
            raise ValueError(
                "Prophet and ARIMA produced different forecast timestamps. "
                "Please select a supported daily, weekly, or monthly frequency."
            )

        arima_values = self._clamp_to_series_scale(arima.values)
        prophet_values = self._clamp_to_series_scale(prophet_values)
        absolute_difference = np.abs(arima_values - prophet_values)
        series_scale = max(float(np.nanstd(self.series.values)), 1e-8)

        comparison = pd.DataFrame(
            {
                "ARIMA": arima_values,
                "Prophet": prophet_values,
                "Absolute difference": absolute_difference,
                "Difference (series std)": absolute_difference / series_scale,
            },
            index=forecast_index,
        )

        return comparison

    # --------------------------------------------------
    # TREND PLOT
    # --------------------------------------------------

    def plot_trend(self):

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(self.series, label="Original Series")

        ax.set_title("Time Series")

        ax.set_xlabel("Date")

        ax.set_ylabel("Value")

        ax.legend()

        return fig

    # --------------------------------------------------
    # DECOMPOSITION PLOT
    # --------------------------------------------------

    def plot_decomposition(self):

        if self.decomposition is None:

            self.decompose_series()

        if self.decomposition is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5,
                0.5,
                "Not enough data points for seasonal decomposition",
                ha="center",
                va="center",
            )
            return fig

        fig = self.decomposition.plot()

        fig.set_size_inches(12, 8)

        return fig

    # --------------------------------------------------
    # ROLLING STATISTICS PLOT
    # --------------------------------------------------

    def plot_rolling_statistics(self, window=30):

        stats = self.rolling_statistics(window)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(self.series, label="Original")

        ax.plot(stats["Rolling Mean"], label="Rolling Mean")

        ax.plot(stats["Rolling Std"], label="Rolling Std")

        ax.legend()

        ax.set_title("Rolling Statistics")

        return fig

    def plot_autocorrelation(self, lags=40):
        values = self.autocorrelation(lags=lags)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(len(values)), values)
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

        return fig

    # --------------------------------------------------
    # ARIMA FORECAST PLOT
    # --------------------------------------------------

    def plot_arima_forecast(self):

        if self.arima_forecast is None:

            self.forecast_arima()

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(self.series, label="Historical")

        clamped_forecast = self._clamp_to_series_scale(self.arima_forecast.values)

        ax.plot(
            self.arima_forecast.index,
            clamped_forecast,
            label="ARIMA Forecast",
        )

        ax.legend()

        return fig

    # --------------------------------------------------
    # PROPHET FORECAST PLOT
    # --------------------------------------------------

    def plot_prophet_forecast(self):

        if self.prophet_forecast is None:

            self.forecast_prophet()

        future_rows = self.prophet_forecast[
            self.prophet_forecast["ds"] > self.series.index.max()
        ].copy()
        if "yhat" in future_rows.columns:
            future_rows["yhat"] = self._clamp_to_series_scale(future_rows["yhat"].values)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.series.index, self.series.values, label="Historical")
        ax.plot(future_rows["ds"], future_rows["yhat"], label="Prophet Forecast")
        ax.set_title("Prophet forecast")
        ax.legend()

        return fig

    def plot_forecast_comparison(self, periods=30):
        comparison = self.compare_models(periods=periods)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.series.index, self.series.values, label="Historical")
        ax.plot(comparison.index, comparison["ARIMA"], label="ARIMA Forecast")
        ax.plot(comparison.index, comparison["Prophet"], label="Prophet Forecast")
        ax.set_title("ARIMA vs Prophet (same scale)")
        ax.legend()
        return fig

    # --------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------

    def save_models(self, directory="serialized_weights"):

        os.makedirs(directory, exist_ok=True)

        if self.arima_model is not None:

            joblib.dump(self.arima_model, os.path.join(directory, "arima.pkl"))

        if self.prophet_model is not None:

            joblib.dump(self.prophet_model, os.path.join(directory, "prophet.pkl"))

    # --------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------

    def load_models(self, directory="serialized_weights"):

        arima_path = os.path.join(directory, "arima.pkl")

        prophet_path = os.path.join(directory, "prophet.pkl")

        if os.path.exists(arima_path):

            self.arima_model = joblib.load(arima_path)

        if os.path.exists(prophet_path):

            self.prophet_model = joblib.load(prophet_path)


# ==================================================
# RUN TIME SERIES ENGINE
# ==================================================


def run_time_series_engine(
    dataframe, date_column, target_column, frequency="D", forecast_periods=30
):

    engine = TimeSeriesEngine()

    engine.load_series(dataframe, date_column, target_column, frequency)

    engine.decompose_series()

    engine.build_arima()

    engine.build_prophet()

    comparison = engine.compare_models(forecast_periods)
    agreement = pd.DataFrame(
        [
            {
                "Mean absolute difference": comparison["Absolute difference"].mean(),
                "Max absolute difference": comparison["Absolute difference"].max(),
                "Mean difference (series std)": comparison[
                    "Difference (series std)"
                ].mean(),
            }
        ]
    )
    validation = engine.validation_table(
        periods=min(forecast_periods, max(1, len(engine.series) // 4))
    )

    return {
        "engine": engine,
        "comparison": comparison,
        "forecast_agreement": agreement,
        "comparison_plot": engine.plot_forecast_comparison(forecast_periods),
        "trend_plot": engine.plot_trend(),
        "decomposition_plot": engine.plot_decomposition(),
        "rolling_plot": engine.plot_rolling_statistics(),
        "autocorrelation_plot": engine.plot_autocorrelation(),
        "arima_plot": engine.plot_arima_forecast(),
        "prophet_plot": engine.plot_prophet_forecast(),
        "validation_table": validation,
    }


# ==================================================
# MODULE TEST
# ==================================================

if __name__ == "__main__":

    dates = pd.date_range("2022-01-01", periods=365)

    values = np.random.randint(100, 300, 365)

    sample = pd.DataFrame({"Date": dates, "Sales": values})

    results = run_time_series_engine(sample, "Date", "Sales")

    print("=" * 60)
    print("Time Series Forecasting Module")
    print("=" * 60)

    print(results["comparison"].head())
