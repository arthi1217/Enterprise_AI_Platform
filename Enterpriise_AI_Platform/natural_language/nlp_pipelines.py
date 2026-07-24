import re
import string
import numpy as np
import pandas as pd
from collections import Counter

import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

from gensim.models import Word2Vec
from gensim.downloader import load as gensim_load

import matplotlib.pyplot as plt

# Download required resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    pass

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")


class NLPPipeline:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))

        self.stemmer = PorterStemmer()

        self.lemmatizer = WordNetLemmatizer()

        self.tfidf_vectorizer = None

        self.word2vec_model = None

        self.glove_model = None

        # -------------------------------------------------------

    # CLEAN TEXT
    # -------------------------------------------------------

    def clean_text(self, text):

        if pd.isna(text):
            return ""

        text = str(text).lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"www\S+", "", text)

        text = re.sub(r"\d+", "", text)

        text = text.translate(str.maketrans("", "", string.punctuation))

        text = re.sub(r"\s+", " ", text)

        return text.strip()

        # -------------------------------------------------------

    # TOKENIZATION
    # -------------------------------------------------------

    def tokenize(self, text):

        text = self.clean_text(text)

        return word_tokenize(text)

    # -------------------------------------------------------
    # STOPWORD REMOVAL
    # -------------------------------------------------------

    def remove_stopwords(self, tokens):

        return [word for word in tokens if word not in self.stop_words]

        # -------------------------------------------------------

    # STEMMING
    # -------------------------------------------------------

    def stem_words(self, tokens):

        return [self.stemmer.stem(word) for word in tokens]

        # -------------------------------------------------------

    # LEMMATIZATION
    # -------------------------------------------------------

    def lemmatize_words(self, tokens):

        return [self.lemmatizer.lemmatize(word) for word in tokens]

        # -------------------------------------------------------

    # COMPLETE PREPROCESSING
    # -------------------------------------------------------

    def preprocess(self, text):

        cleaned = self.clean_text(text)

        tokens = self.tokenize(cleaned)

        tokens = self.remove_stopwords(tokens)

        lemmas = self.lemmatize_words(tokens)

        return lemmas

        # -------------------------------------------------------

    # TF-IDF VECTORIZATION
    # -------------------------------------------------------

    def fit_tfidf(self, texts, max_features=5000):

        cleaned = [" ".join(self.preprocess(text)) for text in texts]

        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

        matrix = self.tfidf_vectorizer.fit_transform(cleaned)

        return matrix

    def transform_tfidf(self, texts):

        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF model has not been fitted.")

        cleaned = [" ".join(self.preprocess(text)) for text in texts]

        return self.tfidf_vectorizer.transform(cleaned)

    def get_feature_names(self):

        if self.tfidf_vectorizer is None:
            return []

        return self.tfidf_vectorizer.get_feature_names_out()

        # -------------------------------------------------------

    # WORD2VEC
    # -------------------------------------------------------

    def train_word2vec(self, texts, vector_size=100, window=5, min_count=2):

        sentences = [self.preprocess(text) for text in texts]

        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
        )

        return self.word2vec_model

    def get_word_vector(self, word):

        if self.word2vec_model is None:
            return None

        if word not in self.word2vec_model.wv:
            return None

        return self.word2vec_model.wv[word]

    def most_similar_words(self, word, topn=10):

        if self.word2vec_model is None:
            return []

        if word not in self.word2vec_model.wv:
            return []

        return self.word2vec_model.wv.most_similar(word, topn=topn)

        # -------------------------------------------------------

    # GLOVE EMBEDDINGS
    # -------------------------------------------------------

    def load_glove_model(self):

        if self.glove_model is None:

            self.glove_model = gensim_load("glove-wiki-gigaword-100")

        return self.glove_model

    def glove_vector(self, word):

        self.load_glove_model()

        if word not in self.glove_model:
            return None

        return self.glove_model[word]

    def glove_similarity(self, word1, word2):

        self.load_glove_model()

        if word1 not in self.glove_model:
            return None

        if word2 not in self.glove_model:
            return None

        return self.glove_model.similarity(word1, word2)

        # -------------------------------------------------------

    # PART OF SPEECH TAGGING
    # -------------------------------------------------------

    def pos_tags(self, text):

        tokens = self.preprocess(text)

        return pos_tag(tokens)

        # -------------------------------------------------------

    # NAMED ENTITY RECOGNITION
    # -------------------------------------------------------

    def named_entities(self, text):

        document = nlp(text)

        entities = []

        for entity in document.ents:

            entities.append({"text": entity.text, "label": entity.label_})

        return entities

        # -------------------------------------------------------

    # COSINE SIMILARITY
    # -------------------------------------------------------

    def cosine_similarity_matrix(self, texts):

        matrix = self.fit_tfidf(texts)

        return cosine_similarity(matrix)

        # -------------------------------------------------------

    # SENTIMENT ANALYSIS
    # -------------------------------------------------------

    def sentiment_analysis(self, text):
        """
        Perform sentiment analysis using TextBlob.
        """

        analysis = TextBlob(str(text))

        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity,
        }

    # -------------------------------------------------------
    # ANALYZE A DATAFRAME COLUMN
    # -------------------------------------------------------

    def analyze_dataframe(self, dataframe, text_column):
        """
        Apply preprocessing and sentiment analysis to a dataframe.
        """

        df = dataframe.copy()

        df["clean_text"] = df[text_column].astype(str).apply(self.clean_text)

        df["tokens"] = df[text_column].astype(str).apply(self.preprocess)

        df["stemmed_tokens"] = df["tokens"].apply(self.stem_words)

        sentiments = df[text_column].astype(str).apply(self.sentiment_analysis)

        df["sentiment"] = sentiments.apply(lambda x: x["sentiment"])

        df["polarity"] = sentiments.apply(lambda x: x["polarity"])

        df["subjectivity"] = sentiments.apply(lambda x: x["subjectivity"])

        return df

    def _document_embedding_word2vec(self, tokens):
        if self.word2vec_model is None:
            return None
        vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def _document_embedding_glove(self, tokens):
        self.load_glove_model()
        vectors = [
            self.glove_model[token] for token in tokens if token in self.glove_model
        ]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

        # -------------------------------------------------------

    # WORD FREQUENCY
    # -------------------------------------------------------

    def word_frequency(self, texts, top_n=20):

        frequency = {}

        for text in texts:

            words = self.preprocess(text)

            for word in words:

                frequency[word] = frequency.get(word, 0) + 1

        frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

        return frequency[:top_n]

        # -------------------------------------------------------

    # TOP TF-IDF KEYWORDS
    # -------------------------------------------------------

    def top_keywords(self, texts, top_n=20):

        matrix = self.fit_tfidf(texts)

        scores = np.asarray(matrix.mean(axis=0)).flatten()

        features = self.get_feature_names()

        ranking = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)

        return ranking[:top_n]

        # -------------------------------------------------------

    # SENTIMENT DISTRIBUTION
    # -------------------------------------------------------

    def plot_sentiment_distribution(self, dataframe):

        counts = dataframe["sentiment"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(counts.index, counts.values)

        ax.set_title("Sentiment Distribution")

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")

        return fig

        # -------------------------------------------------------

    # WORD FREQUENCY PLOT
    # -------------------------------------------------------

    def plot_word_frequency(self, texts, top_n=15):

        words = self.word_frequency(texts, top_n)

        labels = [x[0] for x in words]
        values = [x[1] for x in words]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(labels, values)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title("Top Words")

        return fig

        # -------------------------------------------------------

    # NLP SUMMARY
    # -------------------------------------------------------

    def summary(self, dataframe, text_column):

        processed = self.analyze_dataframe(dataframe, text_column)

        return {
            "documents": len(processed),
            "average_polarity": processed["polarity"].mean(),
            "average_subjectivity": processed["subjectivity"].mean(),
            "sentiment_counts": processed["sentiment"].value_counts().to_dict(),
        }


import os
import joblib

# ==========================================================
# SAVE TF-IDF MODEL
# ==========================================================


def save_tfidf_model(pipeline, path="serialized_weights/tfidf_vectorizer.pkl"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(pipeline.tfidf_vectorizer, path)

    return path


# ==========================================================
# LOAD TF-IDF MODEL
# ==========================================================


def load_tfidf_model(pipeline, path="serialized_weights/tfidf_vectorizer.pkl"):

    pipeline.tfidf_vectorizer = joblib.load(path)

    return pipeline


# ==========================================================
# SAVE WORD2VEC MODEL
# ==========================================================


def save_word2vec_model(pipeline, path="serialized_weights/word2vec.model"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    pipeline.word2vec_model.save(path)

    return path


# ==========================================================
# LOAD WORD2VEC MODEL
# ==========================================================


def load_word2vec_model(pipeline, path="serialized_weights/word2vec.model"):

    pipeline.word2vec_model = Word2Vec.load(path)

    return pipeline


# ==========================================================
# EXPORT DATAFRAME
# ==========================================================


def export_dataframe(dataframe, path="analytical_reports/nlp_results.csv"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    dataframe.to_csv(path, index=False)

    return path


# ==========================================================
# RUN NLP PIPELINE
# ==========================================================


def run_nlp_pipeline(dataframe, text_column):

    if text_column not in dataframe.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the dataframe.")

    pipeline = NLPPipeline()

    processed = pipeline.analyze_dataframe(dataframe, text_column)
    processed[text_column] = processed[text_column].fillna("").astype(str)
    processed["tokens"] = processed["tokens"].apply(
        lambda tokens: [str(token) for token in tokens] if isinstance(tokens, list) else []
    )

    if processed["tokens"].map(len).sum() == 0:
        raise ValueError(
            "Selected text column does not contain enough language content for NLP analysis."
        )

    try:
        pipeline.fit_tfidf(processed[text_column])
    except ValueError as error:
        raise ValueError(
            "Unable to compute TF-IDF for this text column. Please choose a richer text field."
        ) from error

    word2vec_error = None
    try:
        pipeline.train_word2vec(processed[text_column])
    except RuntimeError as error:
        word2vec_error = str(error)
        pipeline.word2vec_model = None

    tfidf_matrix = pipeline.transform_tfidf(processed[text_column])

    sentiment_fig = pipeline.plot_sentiment_distribution(processed)

    frequency_fig = pipeline.plot_word_frequency(processed[text_column])

    keywords = pipeline.top_keywords(processed[text_column])

    sample_size = min(len(processed), 250)
    sampled_text = processed[text_column].head(sample_size).astype(str).tolist()
    sampled_tokens = processed["tokens"].head(sample_size).tolist()

    pos_counter = Counter()
    for tokens in sampled_tokens:
        for _, tag in pos_tag(tokens):
            pos_counter[tag] += 1
    pos_summary = pd.DataFrame(
        pos_counter.most_common(20), columns=["POS Tag", "Count"]
    )

    ner_records = []
    for text in sampled_text:
        for entity in pipeline.named_entities(text):
            ner_records.append({"Entity": entity["text"], "Label": entity["label"]})
    if ner_records:
        ner_df = pd.DataFrame(ner_records)
        ner_summary = (
            ner_df.groupby(["Entity", "Label"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(25)
        )
    else:
        ner_summary = pd.DataFrame(columns=["Entity", "Label", "Count"])

    word2vec_embeddings = []
    glove_embeddings = []
    word2vec_coverage = []
    glove_coverage = []

    glove_error = None
    try:
        pipeline.load_glove_model()
    except (ValueError, OSError, RuntimeError) as error:
        glove_error = str(error)

    for tokens in sampled_tokens:
        word2vec_vector = pipeline._document_embedding_word2vec(tokens)
        word2vec_embeddings.append(word2vec_vector)
        token_count = max(len(tokens), 1)
        if pipeline.word2vec_model is None:
            word2vec_coverage.append(0.0)
        else:
            word2vec_coverage.append(
                sum(1 for token in tokens if pipeline.get_word_vector(token) is not None)
                / token_count
            )

        if glove_error is None:
            glove_vector = pipeline._document_embedding_glove(tokens)
            glove_embeddings.append(glove_vector)
            glove_coverage.append(
                sum(1 for token in tokens if pipeline.glove_vector(token) is not None)
                / token_count
            )
        else:
            glove_embeddings.append(None)
            glove_coverage.append(0.0)

    tfidf_sample = tfidf_matrix[:sample_size]
    tfidf_similarity = cosine_similarity(tfidf_sample)
    upper_idx = np.triu_indices_from(tfidf_similarity, k=1)
    tfidf_upper = tfidf_similarity[upper_idx]

    def _embedding_similarity(embeddings):
        valid = [embedding for embedding in embeddings if embedding is not None]
        if len(valid) < 2:
            return np.nan, np.nan
        matrix = np.vstack(valid)
        similarity = cosine_similarity(matrix)
        upper = similarity[np.triu_indices_from(similarity, k=1)]
        return float(np.mean(upper)), upper

    w2v_avg_similarity, w2v_upper = _embedding_similarity(word2vec_embeddings)
    glove_avg_similarity, glove_upper = _embedding_similarity(glove_embeddings)

    if isinstance(w2v_upper, np.ndarray) and len(w2v_upper) == len(tfidf_upper):
        w2v_corr = float(np.corrcoef(tfidf_upper, w2v_upper)[0, 1])
    else:
        w2v_corr = np.nan

    if isinstance(glove_upper, np.ndarray) and len(glove_upper) == len(tfidf_upper):
        glove_corr = float(np.corrcoef(tfidf_upper, glove_upper)[0, 1])
    else:
        glove_corr = np.nan

    tfidf_vocab_size = len(pipeline.get_feature_names())
    tfidf_denominator = tfidf_sample.shape[0] * tfidf_sample.shape[1]
    tfidf_density = (
        tfidf_sample.nnz / tfidf_denominator if tfidf_denominator > 0 else np.nan
    )

    embedding_comparison = pd.DataFrame(
        [
            {
                "Embedding": "TF-IDF",
                "Vector Size": tfidf_vocab_size,
                "Avg Pairwise Cosine": (
                    float(np.mean(tfidf_upper)) if len(tfidf_upper) > 0 else np.nan
                ),
                "Token Coverage": 1.0,
                "Similarity Correlation vs TF-IDF": 1.0,
                "Sparsity/Density": tfidf_density,
            },
            {
                "Embedding": "Word2Vec",
                "Vector Size": (
                    pipeline.word2vec_model.vector_size
                    if pipeline.word2vec_model is not None
                    else np.nan
                ),
                "Avg Pairwise Cosine": w2v_avg_similarity,
                "Token Coverage": float(np.mean(word2vec_coverage)),
                "Similarity Correlation vs TF-IDF": w2v_corr,
                "Sparsity/Density": np.nan,
            },
            {
                "Embedding": "GloVe",
                "Vector Size": 100 if glove_error is None else np.nan,
                "Avg Pairwise Cosine": glove_avg_similarity,
                "Token Coverage": float(np.mean(glove_coverage)),
                "Similarity Correlation vs TF-IDF": glove_corr,
                "Sparsity/Density": np.nan,
            },
        ]
    )

    summary = pipeline.summary(dataframe, text_column)

    return {
        "pipeline": pipeline,
        "processed_dataframe": processed,
        "summary": summary,
        "keywords": keywords,
        "sentiment_plot": sentiment_fig,
        "word_frequency_plot": frequency_fig,
        "pos_summary": pos_summary,
        "ner_summary": ner_summary,
        "embedding_comparison": embedding_comparison,
        "word2vec_error": word2vec_error,
        "glove_error": glove_error,
    }

    # ==========================================================


# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "text": [
                "Artificial Intelligence is amazing.",
                "This project is difficult.",
                "The weather is nice today.",
                "I absolutely love machine learning.",
                "The service was terrible.",
            ]
        }
    )

    results = run_nlp_pipeline(sample, "text")

    print("=" * 60)
    print("Natural Language Processing Module")
    print("=" * 60)

    print(results["summary"])

    print("\nTop Keywords\n")

    for word, score in results["keywords"]:

        print(word, ":", round(score, 4))
