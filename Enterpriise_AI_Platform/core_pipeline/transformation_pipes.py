import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None


# ==========================================================
# DATA LOADING
# ==========================================================

def load_olist_data(
    customers_path,
    orders_path,
    order_items_path,
    products_path,
    sellers_path,
    payments_path,
    reviews_path
):
    """
    Load and merge all Olist datasets.
    """

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(order_items_path)
    products = pd.read_csv(products_path)
    sellers = pd.read_csv(sellers_path)
    payments = pd.read_csv(payments_path)
    reviews = pd.read_csv(reviews_path)

    df = orders.merge(customers, on="customer_id", how="left")

    df = df.merge(items, on="order_id", how="left")

    df = df.merge(products, on="product_id", how="left")

    df = df.merge(sellers, on="seller_id", how="left")

    df = df.merge(payments, on="order_id", how="left")

    df = df.merge(
        reviews[["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left"
    )

    return df


# ==========================================================
# DATE PROCESSING
# ==========================================================

def process_dates(df):
    """
    Convert timestamp columns into datetime objects.
    """

    date_columns = [

        "order_purchase_timestamp",

        "order_approved_at",

        "order_delivered_customer_date",

        "order_estimated_delivery_date"

    ]

    for col in date_columns:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def feature_engineering(df):
    """
    Generate engineered features.
    """

    df = process_dates(df)

    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        -
        df["order_purchase_timestamp"]
    ).dt.days

    df["delivery_delay"] = (
        df["order_delivered_customer_date"]
        -
        df["order_estimated_delivery_date"]
    ).dt.days

    df["total_order_value"] = (
        df["price"].fillna(0)
        +
        df["freight_value"].fillna(0)
    )

    median_order_value = df["total_order_value"].median()
    df["high_value_order"] = (
        df["total_order_value"] > median_order_value
    ).astype(int)

    df["purchase_year"] = (
        df["order_purchase_timestamp"]
        .dt.year
    )

    df["purchase_month"] = (
        df["order_purchase_timestamp"]
        .dt.month
    )

    df["purchase_day"] = (
        df["order_purchase_timestamp"]
        .dt.day
    )

    df = create_lag_features(
        df,
        column="payment_value",
        lags=[1, 7]
    )

    return df


# ==========================================================
# FEATURE TABLE
# ==========================================================

def create_feature_table(df):
    """
    Select features shared across modules.
    """

    columns = [

        "customer_city",

        "customer_state",

        "seller_city",

        "seller_state",

        "product_category_name",

        "payment_type",

        "payment_installments",

        "payment_value",

        "price",

        "freight_value",

        "delivery_days",

        "delivery_delay",

        "review_score",

        "total_order_value",

        "payment_value_lag_1",

        "payment_value_lag_7",

        "high_value_order",

        "purchase_year",

        "purchase_month",

        "purchase_day"

    ]

    return df[columns].copy()

# ==========================================================
# PREPROCESSING
# ==========================================================

def get_feature_lists():
    """
    Returns numerical and categorical feature lists.
    """

    numerical_features = [
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "delivery_days",
        "delivery_delay",
        "review_score",
        "total_order_value",
        "payment_value_lag_1",
        "payment_value_lag_7",
        "high_value_order",
        "purchase_year",
        "purchase_month",
        "purchase_day"
    ]

    categorical_features = [
        "customer_city",
        "customer_state",
        "seller_city",
        "seller_state",
        "product_category_name",
        "payment_type"
    ]

    return numerical_features, categorical_features


# ==========================================================
# COLUMN TRANSFORMER
# ==========================================================

def build_preprocessor():
    """
    Build a reusable preprocessing pipeline.
    """

    numerical_features, categorical_features = get_feature_lists()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# ==========================================================
# TRANSFORM DATA
# ==========================================================

def preprocess_data(df):
    """
    Complete preprocessing routine.
    """

    feature_df = create_feature_table(df)

    preprocessor = build_preprocessor()

    X = preprocessor.fit_transform(feature_df)

    return X, feature_df, preprocessor


# ==========================================================
# DENSE MATRIX
# ==========================================================

def to_dense(X):
    """
    Convert sparse matrices into dense arrays.
    """

    if hasattr(X, "toarray"):
        return X.toarray()

    return X


# ==========================================================
# DATA SUMMARY
# ==========================================================

def dataset_summary(df):
    """
    Quick statistics for dashboard.
    """

    summary = {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Missing Values": int(df.isna().sum().sum()),

        "Duplicate Rows": int(df.duplicated().sum())

    }

    return summary

# ==========================================================
# LAG FEATURE ENGINEERING
# ==========================================================

def create_lag_features(
    df,
    column="payment_value",
    lags=[1, 7]
):
    """
    Create lag features for forecasting and sequence models.
    """

    if column not in df.columns:
        return df

    df = df.sort_values("order_purchase_timestamp")

    for lag in lags:

        df[f"{column}_lag_{lag}"] = (
            df[column]
            .shift(lag)
        )

    return df


# ==========================================================
# TARGET ENCODING
# ==========================================================

def apply_target_encoding(
    df,
    categorical_columns,
    target_column
):
    """
    Target Encoding using category_encoders.
    """

    if TargetEncoder is None:

        print(
            "category_encoders not installed. Skipping Target Encoding."
        )

        return df, None

    encoder = TargetEncoder(
        cols=categorical_columns
    )

    df[categorical_columns] = encoder.fit_transform(
        df[categorical_columns],
        df[target_column]
    )

    return df, encoder


# ==========================================================
# SMOTE
# ==========================================================

def apply_smote(
    X,
    y,
    random_state=42
):
    """
    Balance classes using SMOTE safely.
    """
    try:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 2:
            return X, y

        k_neighbors = min(5, np.min(counts) - 1)
        if k_neighbors < 1:
            return X, y

        smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors
        )

        X_resampled, y_resampled = smote.fit_resample(
            X,
            y
        )

        return X_resampled, y_resampled
    except Exception:
        return X, y


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

def calculate_class_weights(y):
    """
    Compute balanced class weights.
    """

    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))


# ==========================================================
# PIPELINE SUMMARY
# ==========================================================

def preprocessing_report(
    original_df,
    processed_matrix,
    class_weights=None,
    smote_shape=None,
    target_encoder_fitted=False
):
    """
    Generate preprocessing statistics.
    """

    report = {

        "Original Rows": len(original_df),

        "Original Columns": len(original_df.columns),

        "Processed Shape": processed_matrix.shape,

        "Missing Values Before":
            int(original_df.isna().sum().sum()),

        "Missing Values After":
            0,
        "Target Encoder Fitted":
            target_encoder_fitted,
        "Class Weights":
            class_weights,
        "SMOTE Shape":
            smote_shape

    }

    return report

# ==========================================================
# MAIN DATA PREPARATION FUNCTION
# ==========================================================

def prepare_dataset(
    customers_path,
    orders_path,
    order_items_path,
    products_path,
    sellers_path,
    payments_path,
    reviews_path
):
    """
    Main preprocessing pipeline used by all project modules.
    """

    df = load_olist_data(
        customers_path,
        orders_path,
        order_items_path,
        products_path,
        sellers_path,
        payments_path,
        reviews_path
    )

    df = feature_engineering(df)

    feature_df = create_feature_table(df)

    X, processed_df, preprocessor = preprocess_data(df)

    target_encoded_df, target_encoder = apply_target_encoding(
        feature_df.copy(),
        categorical_columns=[
            "customer_city",
            "customer_state",
            "seller_city",
            "seller_state",
            "product_category_name",
            "payment_type"
        ],
        target_column="high_value_order"
    )

    y = feature_df["high_value_order"].fillna(0).astype(int)

    class_weights = calculate_class_weights(y)

    smote_input = feature_df.select_dtypes(include=np.number).fillna(0)
    smote_X, smote_y = apply_smote(
        smote_input,
        y
    )

    diagnostics_table, diagnostics_plot = compare_pipeline_diagnostics(
        feature_df,
        X,
        target_column="high_value_order"
    )

    report = preprocessing_report(
        feature_df,
        X,
        class_weights=class_weights,
        smote_shape=(smote_X.shape, smote_y.shape),
        target_encoder_fitted=target_encoder is not None
    )

    return {
        "raw_dataframe": df,
        "feature_dataframe": feature_df,
        "target_encoded_feature_dataframe": target_encoded_df,
        "processed_features": X,
        "preprocessor": preprocessor,
        "report": report,
        "pipeline_diagnostics_table": diagnostics_table,
        "pipeline_diagnostics_plot": diagnostics_plot
    }


def compare_pipeline_diagnostics(
    feature_df,
    processed_matrix,
    target_column="high_value_order"
):
    y = feature_df[target_column].fillna(0).astype(int)

    if len(np.unique(y)) < 2:
        diagnostics = pd.DataFrame({
            "Pipeline": ["Raw numeric baseline", "Transformed pipeline"],
            "Accuracy": [np.nan, np.nan],
            "F1 Score": [np.nan, np.nan]
        })
        return diagnostics, plot_pipeline_diagnostics(diagnostics)

    numeric_baseline = feature_df.select_dtypes(include=np.number).copy()
    numeric_baseline = numeric_baseline.drop(columns=[target_column], errors="ignore")

    baseline_imputer = SimpleImputer(strategy="median")
    X_baseline = baseline_imputer.fit_transform(numeric_baseline)

    (
        X_base_train,
        X_base_test,
        y_train,
        y_test
    ) = train_test_split(
        X_baseline,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    (
        X_proc_train,
        X_proc_test,
        _,
        _
    ) = train_test_split(
        processed_matrix,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_base_train, y_train)
    baseline_pred = baseline_model.predict(X_base_test)

    transformed_model = LogisticRegression(max_iter=1000)
    transformed_model.fit(X_proc_train, y_train)
    transformed_pred = transformed_model.predict(X_proc_test)

    diagnostics = pd.DataFrame({
        "Pipeline": ["Raw numeric baseline", "Transformed pipeline"],
        "Accuracy": [
            accuracy_score(y_test, baseline_pred),
            accuracy_score(y_test, transformed_pred)
        ],
        "F1 Score": [
            f1_score(y_test, baseline_pred),
            f1_score(y_test, transformed_pred)
        ]
    })

    return diagnostics, plot_pipeline_diagnostics(diagnostics)


def plot_pipeline_diagnostics(diagnostics_table):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(diagnostics_table))
    width = 0.35

    ax.bar(
        x - width / 2,
        diagnostics_table["Accuracy"],
        width=width,
        label="Accuracy"
    )
    ax.bar(
        x + width / 2,
        diagnostics_table["F1 Score"],
        width=width,
        label="F1 Score"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(diagnostics_table["Pipeline"], rotation=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline diagnostics: before vs after transformation")
    ax.legend()

    return fig


# ==========================================================
# BEFORE VS AFTER COMPARISON
# ==========================================================

def compare_before_after(
    original_df,
    processed_matrix
):
    """
    Generate comparison statistics before and after preprocessing.
    """

    comparison = {

        "Original Samples":
            len(original_df),

        "Original Features":
            len(original_df.columns),

        "Processed Samples":
            processed_matrix.shape[0],

        "Processed Features":
            processed_matrix.shape[1],

        "Missing Before":
            int(original_df.isna().sum().sum()),

        "Missing After":
            0

    }

    return pd.DataFrame(
        comparison.items(),
        columns=["Metric", "Value"]
    )


# ==========================================================
# FEATURE NAMES
# ==========================================================

def get_feature_names(preprocessor):
    """
    Return transformed feature names.
    """

    try:

        return preprocessor.get_feature_names_out()

    except Exception:

        return []


# ==========================================================
# PIPELINE INFORMATION
# ==========================================================

def pipeline_information():
    """
    Display pipeline components.
    """

    return {

        "Missing Value Handling":
            "SimpleImputer",

        "Scaling":
            "StandardScaler",

        "Categorical Encoding":
            "OneHotEncoder",

        "Feature Engineering":
            "Lag Features + Date Features",

        "Target Encoding":
            "Supported",

        "SMOTE":
            "Supported",

        "Class Weights":
            "Supported",

        "Framework":
            "Scikit-Learn Pipeline"

    }


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Scalable Data Transformation Pipes")
    print("Enterprise AI Platform")
    print("=" * 60)
    print("Module loaded successfully.")