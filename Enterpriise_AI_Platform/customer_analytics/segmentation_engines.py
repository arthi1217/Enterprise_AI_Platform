import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except Exception:
    umap = None


class SegmentationEngine:

    def __init__(self):

        self.scaler = StandardScaler()

        self.data = None

        self.scaled_data = None

        self.models = {}

        self.cluster_labels = {}

        self.reduced_data = {}
        self.reduction_indices = {}

        self.metrics = {}

        # --------------------------------------------------

    # PREPARE DATA
    # --------------------------------------------------

    def prepare_data(self, dataframe):

        self.data = dataframe.copy()

        numeric = self.data.select_dtypes(include=np.number)

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(axis=1, how="all")

        if numeric.shape[1] == 0:
            raise ValueError(
                "Customer segmentation requires at least one numeric column."
            )

        imputed = SimpleImputer(strategy="median").fit_transform(numeric)

        self.scaled_data = self.scaler.fit_transform(imputed)

        return self.scaled_data

        # --------------------------------------------------

    # KMEANS
    # --------------------------------------------------

    def run_kmeans(self, n_clusters=5):

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        labels = model.fit_predict(self.scaled_data)

        self.models["KMeans"] = model

        self.cluster_labels["KMeans"] = labels

        return labels

        # --------------------------------------------------

    # DBSCAN
    # --------------------------------------------------

    def run_dbscan(self, eps=0.5, min_samples=5):

        model = DBSCAN(eps=eps, min_samples=min_samples)

        labels = model.fit_predict(self.scaled_data)

        self.models["DBSCAN"] = model

        self.cluster_labels["DBSCAN"] = labels

        return labels

        # --------------------------------------------------

    # AGGLOMERATIVE
    # --------------------------------------------------

    def run_agglomerative(self, n_clusters=5):

        model = AgglomerativeClustering(n_clusters=n_clusters)

        labels = model.fit_predict(self.scaled_data)

        self.models["Agglomerative"] = model

        self.cluster_labels["Agglomerative"] = labels

        return labels

        # --------------------------------------------------

    # PCA
    # --------------------------------------------------

    def apply_pca(self, components=2):

        pca = PCA(n_components=components, random_state=42)

        reduced = pca.fit_transform(self.scaled_data)

        self.reduced_data["PCA"] = reduced
        self.reduction_indices["PCA"] = np.arange(self.scaled_data.shape[0])

        return reduced

        # --------------------------------------------------

    # TSNE
    # --------------------------------------------------

    def apply_tsne(self, components=2):
        n_samples = self.scaled_data.shape[0]
        if n_samples <= 2:
            raise ValueError("t-SNE requires at least 3 rows.")

        perplexity = min(30, n_samples - 1)
        max_samples = 1000
        if n_samples > max_samples:
            sample_idx = np.random.RandomState(42).choice(
                n_samples, size=max_samples, replace=False
            )
            tsne_input = self.scaled_data[sample_idx]
        else:
            sample_idx = np.arange(n_samples)
            tsne_input = self.scaled_data

        tsne = TSNE(
            n_components=components, random_state=42, init="pca", perplexity=perplexity
        )

        reduced = tsne.fit_transform(tsne_input)

        self.reduced_data["TSNE"] = reduced
        self.reduction_indices["TSNE"] = sample_idx

        return reduced

        # --------------------------------------------------

    # UMAP
    # --------------------------------------------------

    def apply_umap(self, components=2):

        if umap is None:
            raise ImportError(
                "UMAP is unavailable. Install umap-learn or use PCA/TSNE instead."
            )

        reducer = umap.UMAP(n_components=components, random_state=42)

        max_samples = 2000
        n_samples = self.scaled_data.shape[0]
        if n_samples > max_samples:
            sample_idx = np.random.RandomState(42).choice(
                n_samples, size=max_samples, replace=False
            )
            umap_input = self.scaled_data[sample_idx]
        else:
            sample_idx = np.arange(n_samples)
            umap_input = self.scaled_data

        reduced = reducer.fit_transform(umap_input)

        self.reduced_data["UMAP"] = reduced
        self.reduction_indices["UMAP"] = sample_idx

        return reduced

        # --------------------------------------------------

    # CLUSTER EVALUATION
    # --------------------------------------------------

    def evaluate_clustering(self, model_name):
        """
        Evaluate clustering performance using multiple metrics.
        """

        if model_name not in self.cluster_labels:
            raise ValueError(f"{model_name} has not been executed.")

        labels = self.cluster_labels[model_name]

        # Ignore noise labels for DBSCAN
        unique_labels = np.unique(labels)

        if len(unique_labels) <= 1:
            metrics = {
                "Silhouette Score": None,
                "Davies-Bouldin Index": None,
                "Calinski-Harabasz Index": None,
            }

            self.metrics[model_name] = metrics

            return metrics

        valid_mask = labels != -1

        X = self.scaled_data[valid_mask]
        y = labels[valid_mask]

        if len(np.unique(y)) <= 1:

            metrics = {
                "Silhouette Score": None,
                "Davies-Bouldin Index": None,
                "Calinski-Harabasz Index": None,
            }

        else:

            metrics = {
                "Silhouette Score": silhouette_score(X, y),
                "Davies-Bouldin Index": davies_bouldin_score(X, y),
                "Calinski-Harabasz Index": calinski_harabasz_score(X, y),
            }

        self.metrics[model_name] = metrics

        return metrics

    # --------------------------------------------------
    # EVALUATE ALL MODELS
    # --------------------------------------------------

    def evaluate_all_models(self):

        results = {}

        for model_name in self.cluster_labels.keys():

            results[model_name] = self.evaluate_clustering(model_name)

        return results

        # --------------------------------------------------

    # ELBOW METHOD
    # --------------------------------------------------

    def elbow_method(self, max_clusters=10):

        inertia = []

        cluster_range = range(2, max_clusters + 1)

        for k in cluster_range:

            model = KMeans(n_clusters=k, random_state=42, n_init=10)

            model.fit(self.scaled_data)

            inertia.append(model.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(list(cluster_range), inertia, marker="o")

        ax.set_title("Elbow Method")

        ax.set_xlabel("Number of Clusters")

        ax.set_ylabel("Inertia")

        return fig

        # --------------------------------------------------

    # BEST MODEL
    # --------------------------------------------------

    def best_model(self):

        if len(self.metrics) == 0:

            self.evaluate_all_models()

        best = None

        best_score = -999999

        for model_name, metric in self.metrics.items():

            score = metric["Silhouette Score"]

            if score is None:
                continue

            if score > best_score:

                best_score = score

                best = model_name

        return {"Best Model": best, "Silhouette Score": best_score}

        # --------------------------------------------------

    # CLUSTER SUMMARY
    # --------------------------------------------------

    def cluster_summary(self, model_name):

        if model_name not in self.cluster_labels:

            raise ValueError("Model not found.")

        labels = self.cluster_labels[model_name]

        summary = pd.DataFrame({"Cluster": labels})

        summary = summary.value_counts().reset_index()

        summary.columns = ["Cluster", "Count"]

        return summary

        # --------------------------------------------------

    # CLUSTER VISUALIZATION
    # --------------------------------------------------

    def plot_clusters(self, model_name, reduction="PCA"):
        """
        Visualize clusters using PCA, TSNE or UMAP.
        """

        if model_name not in self.cluster_labels:
            raise ValueError(f"{model_name} has not been executed.")

        reduction = reduction.upper()

        if reduction == "PCA":

            if "PCA" not in self.reduced_data:
                self.apply_pca()

            reduced = self.reduced_data["PCA"]

        elif reduction == "TSNE":

            if "TSNE" not in self.reduced_data:
                self.apply_tsne()

            reduced = self.reduced_data["TSNE"]

        elif reduction == "UMAP":

            if "UMAP" not in self.reduced_data:
                self.apply_umap()

            reduced = self.reduced_data["UMAP"]

        else:

            raise ValueError("Reduction must be PCA, TSNE or UMAP.")

        labels = self.cluster_labels[model_name]
        reduction_idx = self.reduction_indices.get(reduction, np.arange(len(labels)))
        reduced_labels = labels[reduction_idx]

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1], c=reduced_labels, cmap="tab10"
        )

        ax.set_title(f"{model_name} Clusters ({reduction})")

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        plt.colorbar(scatter, ax=ax)

        return fig

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    def save_model(self, model_name, path="serialized_weights"):

        if model_name not in self.models:
            raise ValueError("Model not available.")

        os.makedirs(path, exist_ok=True)

        filename = os.path.join(path, f"{model_name.lower()}.pkl")

        joblib.dump(self.models[model_name], filename)

        return filename

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    def load_model(self, filepath):

        return joblib.load(filepath)


# ==================================================
# RUN SEGMENTATION ENGINE
# ==================================================


def run_segmentation_engine(dataframe, n_clusters=5):

    engine = SegmentationEngine()
    latency_logs = {}

    start = time.perf_counter()

    max_rows = 5000
    working_df = dataframe
    if len(working_df) > max_rows:
        working_df = working_df.sample(n=max_rows, random_state=42)

    engine.prepare_data(working_df)

    if engine.scaled_data.shape[0] < 2:
        raise ValueError("Customer segmentation needs at least two rows.")

    if n_clusters > engine.scaled_data.shape[0]:
        raise ValueError(
            "Number of clusters cannot exceed the number of available rows."
        )

    latency_logs["prepare_data_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    engine.run_kmeans(n_clusters)
    latency_logs["kmeans_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    engine.run_dbscan()
    latency_logs["dbscan_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    engine.run_agglomerative(n_clusters)
    latency_logs["agglomerative_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    engine.apply_pca()
    latency_logs["pca_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    try:
        engine.apply_tsne()
        latency_logs["tsne_seconds"] = time.perf_counter() - start
    except Exception:
        latency_logs["tsne_seconds"] = None

    start = time.perf_counter()
    try:
        engine.apply_umap()
        latency_logs["umap_seconds"] = time.perf_counter() - start
    except Exception:
        latency_logs["umap_seconds"] = None

    metrics = engine.evaluate_all_models()

    best = engine.best_model()

    visualizations = {}
    for model_name in engine.cluster_labels:
        visualizations[model_name] = {}
        for reduction in ["PCA", "TSNE", "UMAP"]:
            if reduction == "PCA":
                visualizations[model_name][reduction] = engine.plot_clusters(
                    model_name, reduction=reduction
                )
            elif reduction in engine.reduced_data:
                visualizations[model_name][reduction] = engine.plot_clusters(
                    model_name, reduction=reduction
                )

    cluster_summaries = {
        model_name: engine.cluster_summary(model_name)
        for model_name in engine.cluster_labels
    }

    return {
        "engine": engine,
        "metrics": metrics,
        "best_model": best,
        "latency_logs": latency_logs,
        "cluster_summaries": cluster_summaries,
        "visualizations": visualizations,
    }


# ==================================================
# COMPARE ALL CLUSTERING MODELS
# ==================================================


def compare_clustering_models(dataframe):

    results = run_segmentation_engine(dataframe)

    return {"Best Model": results["best_model"], "Metrics": results["metrics"]}


# ==================================================
# MODULE TEST
# ==================================================

if __name__ == "__main__":

    np.random.seed(42)

    sample = pd.DataFrame(
        np.random.rand(300, 6), columns=["A", "B", "C", "D", "E", "F"]
    )

    results = run_segmentation_engine(sample)

    print("=" * 60)
    print("Segmentation Engine")
    print("=" * 60)

    print("\nBest Model:\n")

    print(results["best_model"])

    print("\nMetrics:\n")

    for model, metric in results["metrics"].items():

        print(model)

        print(metric)

        print("-" * 40)
