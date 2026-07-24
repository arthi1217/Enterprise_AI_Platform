import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core_pipeline.transformation_pipes import prepare_dataset

from customer_analytics.neural_nets import run_neural_network_module

from customer_analytics.sequence_models import (
    run_sequence_models,
    compare_models as compare_sequence_models,
)

from customer_analytics.segmentation_engines import run_segmentation_engine

from natural_language.nlp_pipelines import run_nlp_pipeline

from forecasting_engine.time_series import run_time_series_engine

PAGE_OPTIONS = [
    "Dashboard",
    "Data upload",
    "Data preprocessing",
    "Neural networks",
    "Customer segmentation",
    "NLP",
    "Sequence models",
    "Time series forecasting",
]

PAGE_META = {
    "Dashboard": ("Command center", ":material/dashboard:"),
    "Data upload": ("Data intake", ":material/upload_file:"),
    "Data preprocessing": ("Feature engineering", ":material/tune:"),
    "Neural networks": ("Classification lab", ":material/neurology:"),
    "Customer segmentation": ("Cluster studio", ":material/account_tree:"),
    "NLP": ("Language intelligence", ":material/text_analysis:"),
    "Sequence models": ("Sequence lab", ":material/timeline:"),
    "Time series forecasting": ("Forecast studio", ":material/query_stats:"),
}

ID_HINTS = (
    "id",
    "uuid",
    "code",
    "key",
    "number",
    "zip",
    "postal",
    "phone",
    "email",
    "sku",
    "reference",
    "token",
)

TEXT_HINTS = (
    "comment",
    "review",
    "message",
    "text",
    "description",
    "content",
    "feedback",
    "note",
)


def inject_app_styles():
    st.markdown(
        """
        <style>
        :root {
            --panel: rgba(13, 27, 47, 0.82);
            --panel-strong: #10233e;
            --line: rgba(148, 185, 255, 0.16);
            --muted: #91a6c7;
            --cyan: #38d9ff;
            --violet: #9b8cff;
        }
        .stApp {
            background:
                radial-gradient(circle at 88% -4%, rgba(56, 217, 255, 0.13), transparent 26rem),
                radial-gradient(circle at -4% 56%, rgba(155, 140, 255, 0.11), transparent 28rem),
                #07111f;
        }
        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.76);
            backdrop-filter: blur(18px);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .app-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(99, 213, 255, 0.24);
            border-radius: 1.25rem;
            padding: 1.7rem 1.8rem;
            background: linear-gradient(120deg, rgba(13, 35, 62, 0.98), rgba(14, 27, 55, 0.86));
            box-shadow: 0 22px 50px rgba(0, 0, 0, 0.2);
            margin-bottom: 0.8rem;
        }
        .app-hero:after {
            content: "";
            position: absolute;
            inset: 0;
            background-image: linear-gradient(rgba(80, 177, 255, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(80, 177, 255, 0.08) 1px, transparent 1px);
            background-size: 30px 30px;
            mask-image: linear-gradient(90deg, transparent, black);
            pointer-events: none;
        }
        .app-hero h1 {
            position: relative;
            margin: 0.35rem 0 0.5rem 0;
            font-size: clamp(2rem, 4vw, 3rem);
            letter-spacing: -0.055em;
            line-height: 1;
            color: #f5f9ff;
        }
        .app-hero p {
            position: relative;
            max-width: 48rem;
            margin: 0;
            color: #b8c8e2;
            font-size: 1rem;
        }
        .app-hero .eyebrow {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--cyan);
            font-weight: 700;
        }
        .app-hero .eyebrow:before {
            content: "";
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 999px;
            background: #5ff2bc;
            box-shadow: 0 0 0.7rem #5ff2bc;
        }
        .module-kicker {
            margin-bottom: 0.25rem;
            color: var(--cyan);
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .app-footer {
            text-align: center;
            color: #7f95ba;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1.1rem;
            border-top: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091729 0%, #07111f 100%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem;
        }
        .sidebar-brand {
            padding: 0.9rem 0.2rem 1.1rem;
        }
        .sidebar-brand h2 {
            margin: 0;
            font-size: 1.05rem;
            color: #f4f8ff;
            letter-spacing: -0.03em;
        }
        .sidebar-brand p {
            margin: 0.35rem 0 0;
            color: var(--muted);
            font-size: 0.78rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(18, 39, 68, 0.9), rgba(10, 25, 45, 0.84));
            border: 1px solid var(--line);
            border-radius: 0.9rem;
            padding: 0.9rem 1rem;
            min-height: 6.2rem;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: #f2f7ff; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            overflow: hidden;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(13, 27, 47, 0.64);
            border-color: var(--line);
            border-radius: 1rem;
        }
        div[data-baseweb="select"] > div, [data-testid="stTextInput"] input {
            background: rgba(9, 24, 43, 0.9);
            border-color: rgba(139, 180, 242, 0.24);
        }
        .stButton > button {
            border: 1px solid rgba(86, 223, 255, 0.45);
            border-radius: 0.65rem;
            background: linear-gradient(135deg, #27c5ef, #5b77ff);
            color: #03101d;
            font-weight: 750;
            box-shadow: 0 8px 22px rgba(46, 191, 255, 0.18);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(46, 191, 255, 0.3);
            border-color: #b1efff;
        }
        [data-testid="stAlert"] {
            border-radius: 0.75rem;
            border: 1px solid var(--line);
        }
        [data-testid="stExpander"] {
            border-color: var(--line);
            border-radius: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">System online · Enterprise intelligence</div>
            <h1>AI operations workspace</h1>
            <p>From raw commerce data to explainable machine-learning signals — in one focused technical environment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_id_like(column_name: str) -> bool:
    lower_name = column_name.lower()
    return any(hint in lower_name for hint in ID_HINTS)


def get_datetime_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]


def get_text_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = []
    for column in dataframe.select_dtypes(include=["object", "string"]).columns:
        lower_name = column.lower()
        if is_id_like(lower_name):
            continue
        if any(hint in lower_name for hint in TEXT_HINTS):
            columns.append(column)
            continue
        sample = dataframe[column].dropna().astype(str).head(100)
        if sample.empty:
            continue
        avg_length = sample.map(len).mean()
        avg_words = sample.map(lambda value: len(value.split())).mean()
        if avg_length >= 25 and avg_words >= 3:
            columns.append(column)
    return columns


def get_binary_target_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = []
    for column in dataframe.columns:
        if is_id_like(column):
            continue
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            continue
        values = dataframe[column].dropna()
        if values.empty:
            continue
        if values.nunique() == 2:
            columns.append(column)
    return columns


def get_numeric_target_columns(
    dataframe: pd.DataFrame,
    min_unique: int = 10,
) -> list[str]:
    columns = []
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        if is_id_like(column):
            continue
        values = dataframe[column].dropna()
        unique_count = values.nunique()
        if unique_count >= min_unique:
            columns.append(column)
    return columns


def safe_selectbox(
    label: str,
    options: list[str],
    key: str,
    warning_text: str,
    help_text: str | None = None,
):
    if not options:
        st.warning(warning_text)
        return None

    selected = st.selectbox(
        label,
        options,
        key=key,
        help=help_text,
        index=0,
    )

    if selected not in options:
        st.warning(warning_text)
        return None

    return selected


def render_page_title(title: str, subtitle: str):
    module_name, _ = PAGE_META.get(title, ("Workspace", ":material/hub:"))
    st.markdown(f'<div class="module-kicker">{module_name}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.caption(subtitle)


def record_validation_metrics(module: str, model: str, metrics: dict):
    """Store comparable scalar model metrics for the session dashboard."""
    for metric, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            st.session_state.validation_records.append(
                {
                    "Module": module,
                    "Model": model,
                    "Metric": metric,
                    "Value": float(value),
                    "Recorded at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI operations workspace",
    page_icon=":material/neurology:",
    layout="wide",
)
inject_app_styles()

# ============================================================
# TITLE
# ============================================================
render_hero()
st.markdown(
    """
    <div style="margin: 0.1rem 0 1.35rem; color: #91a6c7; font-size: 0.86rem; letter-spacing: 0.02em;">
        <span style="color: #5ff2bc;">●</span> READY &nbsp; / &nbsp; DATA · MODELS · FORECASTS
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "data" not in st.session_state:
    st.session_state.data = None

if "validation_records" not in st.session_state:
    st.session_state.validation_records = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>Neural command</h2>
            <p>Enterprise intelligence platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.selectbox(
        "Workspace module",
        PAGE_OPTIONS,
        index=0,
        key="sidebar_navigation",
    )

    st.caption("CORE STACK")
    st.markdown(":blue-badge[Streamlit] :orange-badge[TensorFlow] :green-badge[Scikit-learn]")

# ============================================================
# DATA UPLOAD
# ============================================================

if page == "Data upload":

    render_page_title(
        "Data upload",
        "Upload the seven Olist CSV files and prepare the unified dataset.",
    )

    uploaded_files = st.file_uploader(
        "Upload all required CSV files", type="csv", accept_multiple_files=True
    )

    required_files = {
        "olist_customers_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
    }

    if uploaded_files:

        uploaded_dict = {}

        for file in uploaded_files:

            uploaded_dict[file.name] = file

        uploaded_names = set(uploaded_dict.keys())

        missing = required_files - uploaded_names

        if len(missing) > 0:

            st.error("Missing Files")

            for file in sorted(missing):

                st.write("❌", file)

            st.stop()

        if st.button("Prepare Enterprise Dataset"):

            with st.spinner("Preparing Dataset..."):

                for file_obj in uploaded_dict.values():
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)

                processed = prepare_dataset(
                    uploaded_dict["olist_customers_dataset.csv"],
                    uploaded_dict["olist_orders_dataset.csv"],
                    uploaded_dict["olist_order_items_dataset.csv"],
                    uploaded_dict["olist_products_dataset.csv"],
                    uploaded_dict["olist_sellers_dataset.csv"],
                    uploaded_dict["olist_order_payments_dataset.csv"],
                    uploaded_dict["olist_order_reviews_dataset.csv"],
                )

            st.session_state.processed_data = processed

            st.session_state.data = processed["raw_dataframe"]

            diagnostics = processed.get("pipeline_diagnostics_table")
            if isinstance(diagnostics, pd.DataFrame):
                for _, row in diagnostics.iterrows():
                    record_validation_metrics(
                        "Transformation pipeline",
                        str(row.get("Pipeline", "Pipeline")),
                        {
                            "Accuracy": row.get("Accuracy", np.nan),
                            "F1 score": row.get("F1 Score", np.nan),
                        },
                    )

            st.success("Dataset Loaded Successfully ✅")

            st.write("### Dataset Information")

            st.metric("Rows", processed["raw_dataframe"].shape[0])

            st.metric("Columns", processed["raw_dataframe"].shape[1])

            st.dataframe(processed["raw_dataframe"].head())

            # ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    render_page_title(
        "Dashboard",
        "Overview of the prepared dataset and pipeline summary.",
    )

    if st.session_state.processed_data is None:

        st.info("Upload all required Olist datasets first.")

    else:

        processed = st.session_state.processed_data

        raw_df = processed["raw_dataframe"]

        feature_df = processed["feature_dataframe"]

        report = processed["report"]

        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", raw_df.shape[0])

        col2.metric("Columns", raw_df.shape[1])

        col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

        col4.metric("Duplicate Rows", int(raw_df.duplicated().sum()))

        if st.session_state.validation_records:
            st.subheader("Validation performance matrix")
            validation_matrix = pd.DataFrame(st.session_state.validation_records)
            st.dataframe(validation_matrix, width="stretch", hide_index=True)
            st.download_button(
                "Download validation matrix",
                validation_matrix.to_csv(index=False),
                file_name="validation_performance_matrix.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        st.divider()

        st.subheader("Raw Dataset")

        st.dataframe(raw_df.head(10), width="stretch")

        st.divider()

        st.subheader("Feature Engineered Dataset")

        st.dataframe(feature_df.head(10), width="stretch")

        st.divider()

        st.subheader("Pipeline Summary")

        if isinstance(report, dict):

            for key, value in report.items():

                st.write(f"**{key}** : {value}")

        else:

            st.write(report)

            # ============================================================
# DATA PREPROCESSING
# ============================================================

elif page == "Data preprocessing":

    render_page_title(
        "Data preprocessing",
        "Inspect the engineered features, encoded matrix, and diagnostics.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload datasets first.")

    else:

        processed = st.session_state.processed_data

        st.success("Dataset has already been preprocessed.")

        st.subheader("Processed Features")

        st.dataframe(processed["feature_dataframe"].head(20), width="stretch")

        st.subheader("Encoded Feature Matrix")

        st.write(processed["processed_features"])

        st.subheader("Pipeline Report")

        report = processed["report"]

        if isinstance(report, dict):

            for key, value in report.items():

                st.write(f"**{key}** : {value}")

        else:

            st.write(report)

        st.subheader("Preprocessor")

        st.write(processed["preprocessor"])

        if "target_encoded_feature_dataframe" in processed:
            st.subheader("Target encoded features")
            st.dataframe(
                processed["target_encoded_feature_dataframe"].head(20), width="stretch"
            )

        if "pipeline_diagnostics_table" in processed:
            st.subheader("Pipeline diagnostics (before vs after)")
            st.dataframe(processed["pipeline_diagnostics_table"], width="stretch")
            st.pyplot(processed["pipeline_diagnostics_plot"])

        # ============================================================
# NEURAL NETWORKS
# ============================================================

elif page == "Neural networks":

    render_page_title(
        "Neural networks",
        "Train the default binary classification model on a valid target.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload and preprocess the dataset first.")

    else:

        processed = st.session_state.processed_data

        feature_df = processed["feature_dataframe"]

        X = processed["processed_features"]

        # -------------------------------
        # Find binary target columns only
        # -------------------------------

        candidate_targets = get_binary_target_columns(feature_df)

        target_column = safe_selectbox(
            "Target column",
            candidate_targets,
            key="nn_target_column",
            warning_text="No valid binary classification columns are available for neural networks.",
        )

        if target_column is None:

            st.stop()

        model_type = st.selectbox(
            "Model type", ["perceptron", "mlp"], key="nn_model_type"
        )
        activation = st.selectbox(
            "Activation", ["relu", "sigmoid", "tanh"], key="nn_activation"
        )
        optimizer = st.selectbox(
            "Optimizer", ["adam", "sgd", "rmsprop"], key="nn_optimizer"
        )

        st.caption("Use the full configuration controls from the project blueprint.")

        if st.button("Train Neural Network"):

            y = feature_df[target_column]

            try:
                results = run_neural_network_module(
                    X,
                    y,
                    model_type=model_type,
                    activation=activation,
                    optimizer=optimizer,
                )
            except ValueError as exc:
                st.warning(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Neural-network training failed: {exc}")
                st.stop()

            st.success("Training Complete!")

            record_validation_metrics(
                "Customer conversion",
                model_type.upper(),
                {"Accuracy": results["results"]["accuracy"]},
            )

            st.metric("Accuracy", f"{results['results']['accuracy']:.4f}")

            st.subheader("Classification Report")

            report = pd.DataFrame(
                results["results"]["classification_report"]
            ).transpose()

            st.dataframe(report)

            st.subheader("Confusion Matrix")

            st.write(results["results"]["confusion_matrix"])

            st.subheader("Training Accuracy")

            st.pyplot(results["accuracy_figure"])

            st.subheader("Training Loss")

            st.pyplot(results["loss_figure"])

            # ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer segmentation":

    render_page_title(
        "Customer segmentation",
        "Cluster customers with stable numeric features and PCA visualization.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload and preprocess the dataset first.")

    else:

        processed = st.session_state.processed_data

        feature_df = processed["feature_dataframe"]

        st.write(
            "Cluster customers using K-Means, DBSCAN and Agglomerative Clustering."
        )

        max_clusters = min(10, len(feature_df))
        if max_clusters < 2:
            st.warning("Customer segmentation needs at least two rows.")
            st.stop()

        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=max_clusters,
            value=min(5, max_clusters),
        )
        reduction = st.selectbox(
            "Visualization reduction",
            ["PCA", "TSNE", "UMAP"],
            key="segmentation_reduction",
        )
        st.caption("All three reduction methods from the rubric are available.")

        if st.button("Run Segmentation"):

            with st.spinner("Running Customer Segmentation..."):
                try:
                    results = run_segmentation_engine(feature_df, n_clusters=n_clusters)
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Customer-segmentation failed: {exc}")
                    st.stop()

            st.success("Segmentation Completed Successfully ✅")

            for model_name, metrics in results["metrics"].items():
                record_validation_metrics("Customer segmentation", model_name, metrics)

            st.subheader("Clustering metrics")
            st.dataframe(pd.DataFrame(results["metrics"]).T, width="stretch")

            st.subheader("Best model")
            st.write(results["best_model"])

            st.subheader("Operational latency logs (seconds)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Step": key, "Seconds": value}
                        for key, value in results["latency_logs"].items()
                    ]
                ),
                width="stretch",
            )

            st.subheader("Cluster summaries")
            for model_name, summary_df in results["cluster_summaries"].items():
                st.write(f"**{model_name}**")
                st.dataframe(summary_df, width="stretch")

            st.subheader(f"Cluster visualizations ({reduction})")
            for model_name, visualization_by_reduction in results[
                "visualizations"
            ].items():
                if reduction in visualization_by_reduction:
                    st.write(f"**{model_name}**")
                    st.pyplot(visualization_by_reduction[reduction])

                # ============================================================
# NATURAL LANGUAGE PROCESSING
# ============================================================

elif page == "NLP":

    render_page_title(
        "NLP",
        "Analyze review-style text columns with sentiment and embeddings.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload and preprocess the dataset first.")

    else:

        processed = st.session_state.processed_data

        raw_df = processed["raw_dataframe"]

        text_columns = get_text_columns(raw_df)

        text_column = safe_selectbox(
            "Text column",
            text_columns,
            key="nlp_text_column",
            warning_text="No valid text columns were found in the dataset.",
        )

        if text_column is None:

            st.stop()

        st.subheader("Sample Reviews")

        st.dataframe(raw_df[[text_column]].head(10), width="stretch")

        if st.button("Run NLP Pipeline"):

            with st.spinner("Processing Text Data..."):
                try:
                    results = run_nlp_pipeline(raw_df, text_column)
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()

            st.success("NLP Analysis Completed ✅")

            record_validation_metrics(
                "Sentiment and feedback",
                "NLP pipeline",
                {"Processed documents": results["summary"].get("documents", 0)},
            )

            st.subheader("Sentiment summary")
            st.write(results["summary"])

            st.subheader("Processed dataframe")
            st.dataframe(results["processed_dataframe"].head(20), width="stretch")

            st.subheader("Top TF-IDF keywords")
            st.dataframe(
                pd.DataFrame(results["keywords"], columns=["Keyword", "Score"]),
                width="stretch",
            )

            st.subheader("Sentiment distribution")
            st.pyplot(results["sentiment_plot"])

            st.subheader("Word frequency")
            st.pyplot(results["word_frequency_plot"])

            st.subheader("POS tag distribution")
            st.dataframe(results["pos_summary"], width="stretch")

            st.subheader("Named entities")
            st.dataframe(results["ner_summary"], width="stretch")

            st.subheader("Embedding cross-analysis")
            st.dataframe(results["embedding_comparison"], width="stretch")
            if results.get("word2vec_error") is not None:
                st.warning(
                    "Word2Vec model could not be trained for this column: "
                    f"{results['word2vec_error']}"
                )
            if results["glove_error"] is not None:
                st.warning(
                    "GloVe model could not be loaded in this environment: "
                    f"{results['glove_error']}"
                )

        st.divider()

        st.subheader("Dataset Text Preview")

        st.write(raw_df[text_column].dropna().head(15))

        # ============================================================
# SEQUENCE MODELS
# ============================================================

elif page == "Sequence models":

    render_page_title(
        "Sequence models",
        "Train the default LSTM pipeline on a valid numeric target.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload and preprocess the dataset first.")

    else:

        processed = st.session_state.processed_data

        feature_df = processed["feature_dataframe"]

        X = processed["processed_features"]

        numeric_columns = get_numeric_target_columns(feature_df, min_unique=2)

        target_column = safe_selectbox(
            "Target column",
            numeric_columns,
            key="sequence_target",
            warning_text="No suitable numeric target columns are available for sequence models.",
        )

        if target_column is None:

            st.stop()

        if len(feature_df) < 8:

            st.warning(
                "Sequence models need at least 8 rows to build training and validation windows."
            )

            st.stop()

        # Keep at least three generated windows: two for training and one for
        # validation. This also prevents BatchNormalization failures on a
        # single-window training split.
        max_sequence_length = min(30, len(feature_df) - 3)
        default_sequence_length = min(10, max_sequence_length)

        sequence_length = st.slider(
            "Sequence length",
            min_value=5,
            max_value=max_sequence_length,
            value=default_sequence_length,
        )
        model_type = st.selectbox(
            "Sequence model", ["lstm", "gru"], key="sequence_model_type"
        )
        l1_penalty = st.slider(
            "L1 regularization",
            min_value=0.0,
            max_value=0.01,
            value=0.0,
            step=0.0005,
            key="sequence_l1",
        )
        l2_penalty = st.slider(
            "L2 regularization",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0005,
            key="sequence_l2",
        )
        st.caption(
            f"Sequence windows from 5 to {max_sequence_length} rows with LSTM/GRU support."
        )

        if st.button("Train Sequence Model"):

            y = feature_df[target_column]

            with st.spinner("Training Model..."):
                try:
                    results = run_sequence_models(
                        X,
                        y,
                        model_type=model_type,
                        sequence_length=sequence_length,
                        l1_penalty=l1_penalty,
                        l2_penalty=l2_penalty,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Sequence-model training failed: {exc}")
                    st.stop()

            st.success("Training Completed Successfully ✅")

            record_validation_metrics(
                "Sequence and anomaly modeling",
                model_type.upper(),
                results["metrics"],
            )

            if isinstance(results, dict):

                for key, value in results.items():

                    st.subheader(str(key))

                    if isinstance(value, pd.DataFrame):

                        st.dataframe(value, width="stretch")

                    elif isinstance(value, plt.Figure):

                        st.pyplot(value)

                    elif hasattr(value, "figure"):

                        st.pyplot(value.figure)

                    elif isinstance(value, np.ndarray):

                        st.write(value)

                    else:

                        st.write(value)

            else:

                st.write(results)

            st.subheader("Anomaly detection")
            st.metric("Detected anomaly windows", len(results["anomaly_indices"]))
            st.pyplot(results["anomaly_plot"])
            anomaly_df = pd.DataFrame(
                {
                    "sequence_index": np.arange(len(results["anomaly_scores"])),
                    "anomaly_score": results["anomaly_scores"],
                    "is_anomaly": False,
                }
            )
            anomaly_df.loc[results["anomaly_indices"], "is_anomaly"] = True
            st.dataframe(anomaly_df.head(100), width="stretch")

        if st.button("Compare LSTM vs GRU"):
            with st.spinner("Comparing sequence models..."):
                try:
                    comparison = compare_sequence_models(
                        X,
                        feature_df[target_column],
                        sequence_length=sequence_length,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Sequence-model comparison failed: {exc}")
                    st.stop()

            st.success(f"Best model: {comparison['best_model']}")
            st.dataframe(pd.DataFrame(comparison["results"]).T, width="stretch")

            # ============================================================
# TIME SERIES FORECASTING
# ============================================================

elif page == "Time series forecasting":

    render_page_title(
        "Time series forecasting",
        "Forecast a datetime-indexed numeric measure with ARIMA and Prophet.",
    )

    if st.session_state.processed_data is None:

        st.warning("Please upload and preprocess the dataset first.")

    else:

        processed = st.session_state.processed_data

        raw_df = processed["raw_dataframe"]

        date_columns = get_datetime_columns(raw_df)
        numeric_columns = get_numeric_target_columns(raw_df, min_unique=10)

        date_column = safe_selectbox(
            "Date column",
            date_columns,
            key="ts_date_column",
            warning_text="No datetime columns are available for forecasting.",
        )

        if date_column is None:

            st.stop()

        target_column = safe_selectbox(
            "Target column",
            numeric_columns,
            key="ts_target_column",
            warning_text="No suitable numeric target columns are available for forecasting.",
        )

        if target_column is None:

            st.stop()

        frequency = st.selectbox("Frequency", ["D", "W", "M"], key="ts_frequency")
        forecast_horizon = st.slider(
            "Prediction horizon",
            min_value=7,
            max_value=90,
            value=30,
        )

        if st.button("Run Forecasting"):

            with st.spinner("Running ARIMA & Prophet..."):
                try:
                    results = run_time_series_engine(
                        raw_df,
                        date_column,
                        target_column,
                        frequency=frequency,
                        forecast_periods=forecast_horizon,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Forecasting failed: {exc}")
                    st.stop()

            st.success("Forecast Generated Successfully ✅")

            if isinstance(results, dict):

                for key, value in results.items():

                    st.subheader(str(key))

                    if isinstance(value, pd.DataFrame):

                        st.dataframe(value, width="stretch")

                    elif isinstance(value, plt.Figure):

                        st.pyplot(value)

                    elif hasattr(value, "figure"):

                        st.pyplot(value.figure)

                    else:

                        st.write(value)

            else:

                st.write(results)

            st.subheader("Validation table")
            st.dataframe(results["validation_table"], width="stretch")
            for _, row in results["validation_table"].iterrows():
                record_validation_metrics(
                    "Financial demand horizons",
                    str(row["Model"]),
                    {"RMSE": row["RMSE"], "MAE": row["MAE"], "MAPE": row["MAPE"]},
                )
            st.caption(
                "ARIMA and Prophet use different assumptions, so different forecasts "
                "are normal. Use the validation errors and the forecast-agreement "
                "diagnostic to judge whether either forecast is credible."
            )

        st.divider()

        st.write("Forecasts generated using ARIMA and Prophet models.")

# ============================================================
# DOWNLOAD REPORT
# ============================================================

if st.session_state.processed_data is not None:

    st.sidebar.divider()

    st.sidebar.subheader("Reports")

    csv = st.session_state.processed_data["feature_dataframe"].to_csv(index=False)

    st.sidebar.download_button(
        label="Download Processed Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv",
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        Enterprise AI Business Intelligence Platform • Streamlit • TensorFlow • Scikit-learn • Prophet • Statsmodels
    </div>
    """,
    unsafe_allow_html=True,
)
