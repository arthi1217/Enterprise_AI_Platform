# Enterprise AI Business Intelligence & Analytics Platform

Unified Streamlit application implementing:

- Customer conversion engine (Perceptron/MLP)
- Sentiment and feedback analytics (NLP + embeddings + NER)
- Sequence and anomaly modeling (LSTM/GRU + Isolation Forest)
- Customer segmentation (K-Means, DBSCAN, Agglomerative + PCA/t-SNE/UMAP)
- Scalable feature transformation pipeline (Lag features, Target Encoding, SMOTE, ColumnTransformer)
- Time-series forecasting (ARIMA + Prophet + validation table)

## Project structure

```text
Enterpriise_AI_Platform/
├── app.py
├── requirements.txt
├── README.md
├── core_pipeline/
├── customer_analytics/
├── natural_language/
├── forecasting_engine/
├── static_assets/
├── serialized_weights/
└── analytical_reports/
```

## Setup guide

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the dashboard:

```bash
streamlit run app.py
```

For architecture, validation, operational guidance, and a rubric-to-feature map, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Running with Olist data

From **Data Upload** page, upload these files together:

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`

Then click **Prepare Enterprise Dataset**.

## Module capability summary

- **Neural Networks:** perceptron/MLP routing, activation and optimizer controls, training telemetry.
- **NLP:** tokenization, stopword removal, stemming, lemmatization, POS tags, sentiment classification, NER, TF-IDF/Word2Vec/GloVe comparison table.
- **Sequence Models:** LSTM/GRU with Dropout, BatchNorm, L1/L2 penalties, anomaly scoring and visualization.
- **Segmentation:** K-Means, DBSCAN, Agglomerative, PCA/t-SNE/UMAP visualization, latency logs.
- **Pipeline:** Lag features, Target Encoding, SMOTE balancing, class weights, ColumnTransformer, before/after diagnostics.
- **Time Series:** ARIMA/Prophet comparison, decomposition, rolling statistics, autocorrelation, configurable prediction horizon, validation metrics.

## Submission notes

Before final zipping:

- remove `.venv`, `env`, and `__pycache__`
- include screenshots/reports under `analytical_reports/`
- include any architecture diagrams and performance evidence required by evaluation rubric
- download the **Validation performance matrix** from the Dashboard after running the modules and place it under `analytical_reports/`
- package the repository as one ZIP file only after verifying the dashboard and removing local caches
