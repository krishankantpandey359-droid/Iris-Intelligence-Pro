
"""
IRIS INTELLIGENCE PRO
Premium Machine Learning Dashboard
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_data, FEATURES, TARGET
from batch_utils import upload_signature, validate_batch_csv
from train import train_and_evaluate


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Iris Intelligence Pro",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PREMIUM DARK THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background: #0b1020;
    color: #edf0ff;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid #27314b;
}

[data-testid="stSidebar"] * {
    color: #e6eaff;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f5ff !important;
}

p, label, span {
    color: #c4cce2;
}

.hero {
    background: linear-gradient(120deg, #171e3c, #28205a 55%, #172a48);
    border: 1px solid #414078;
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 2.5rem;
    margin: 0;
    color: white !important;
}

.hero p {
    color: #c6c9ed;
    margin-top: 8px;
    font-size: 1rem;
}

.metric-card {
    background: linear-gradient(145deg, #171f36, #121a2d);
    border: 1px solid #2d3a5b;
    border-radius: 16px;
    padding: 19px;
    min-height: 110px;
}

.metric-label {
    color: #9ba9ca;
    font-size: 0.85rem;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 8px;
}

.section-card {
    background: #131c31;
    border: 1px solid #293654;
    padding: 20px;
    border-radius: 16px;
}

div.stButton > button {
    background: linear-gradient(100deg, #7057ff, #a052ed);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}

div.stButton > button:hover {
    border: 1px solid #c2b7ff;
    color: white !important;
}

.stDownloadButton button {
    background: #222d49;
    color: white;
    border: 1px solid #465579;
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293654;
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: #151e34;
    border: 1px solid #2e3c5d;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stMetric"] label {
    color: #a9b5d3;
}

div[data-testid="stMetricValue"] {
    color: white;
}

hr {
    border-color: #293654;
}

.stTabs [data-baseweb="tab"] {
    color: #b8c4e4;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================

COLORS = {
    "background": "#0b1020",
    "surface": "#131c31",
    "purple": "#9475ff",
    "blue": "#45a6ff",
    "pink": "#ef83c5",
    "green": "#43d6a1",
    "text": "#edf0ff",
    "muted": "#9ba9ca"
}

CHART_TEMPLATE = "plotly_dark"


def style_chart(fig, height=420):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color=COLORS["text"]
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig


def metric_card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# LOAD DATA AND MODEL
# ==========================================================

@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    model_path = ROOT / "models" / "best_iris_classifier.joblib"

    if not model_path.exists():
        train_and_evaluate()

    if not model_path.exists():
        raise FileNotFoundError("Trained model is unavailable.")

    return joblib.load(model_path)


@st.cache_data
def get_results():
    path = ROOT / "outputs" / "model_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def get_metadata():
    path = ROOT / "outputs" / "run_metadata.json"

    if path.exists():
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    return {}


try:
    df = get_data()
except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()


if not set(FEATURES + [TARGET]).issubset(df.columns):
    st.error("Dataset is missing required columns.")
    st.stop()

if df[FEATURES].isnull().any().any() or df[TARGET].isnull().any():
    st.error("Dataset contains missing values. Please validate the source data.")
    st.stop()

results = get_results()
metadata = get_metadata()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🌸 IRIS PRO")
    st.caption("MACHINE LEARNING INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Data Explorer",
            "Predict Studio",
            "Batch Prediction",
            "Model Intelligence",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### DATASET STATUS")

    st.success("Dataset loaded")

    st.write(f"Records: {len(df)}")
    st.write(f"Features: {len(FEATURES)}")
    st.write(f"Species: {df[TARGET].nunique()}")

    st.divider()

    st.caption("Built with Python • Scikit-learn • Streamlit • Plotly")


# ==========================================================
# HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🌸 Iris Intelligence Pro</h1>
    <p>Advanced Machine Learning Analytics & Prediction Platform</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# EXECUTIVE OVERVIEW
# ==========================================================

if page == "Executive Overview":

    st.subheader("Executive Dashboard")
    st.caption("A complete overview of the Iris classification dataset.")

    total_records = len(df)
    species_count = df[TARGET].nunique()
    feature_count = len(FEATURES)
    missing_values = int(df.isnull().sum().sum())

    columns = st.columns(4)

    cards = [
        ("TOTAL RECORDS", total_records, "Dataset observations"),
        ("FEATURES", feature_count, "Numerical variables"),
        ("SPECIES", species_count, "Target classes"),
        ("MISSING VALUES", missing_values, "Data quality check")
    ]

    for col, card in zip(columns, cards):
        with col:
            metric_card(*card)

    st.write("")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Species Distribution")

        counts = (
            df[TARGET]
            .value_counts()
            .rename_axis("Species")
            .reset_index(name="Records")
        )

        fig = px.pie(
            counts,
            names="Species",
            values="Records",
            hole=0.60,
            color_discrete_sequence=[
                COLORS["purple"],
                COLORS["blue"],
                COLORS["pink"]
            ]
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(style_chart(fig), width="stretch")

    with right:
        st.markdown("### Dataset Summary")

        summary = df[FEATURES].describe().T.reset_index()
        summary = summary.rename(columns={"index": "Feature"})

        st.dataframe(
            summary.round(3),
            width="stretch",
            hide_index=True
        )

        st.markdown("### Data Quality")

        if missing_values == 0:
            st.success("No missing values detected.")
        else:
            st.warning(f"{missing_values} missing values detected.")

    st.divider()

    st.markdown("### Model Snapshot")

    if metadata:

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Selected Model",
            metadata.get("selected_model", "N/A")
        )

        m2.metric(
            "Test Accuracy",
            f"{metadata.get('test_accuracy', 0):.2%}"
        )

        m3.metric(
            "Macro F1 Score",
            f"{metadata.get('macro_f1', 0):.2%}"
        )

    else:
        st.info("Model evaluation results are not available yet.")

    st.markdown("### Recent Dataset Records")
    st.dataframe(df.head(10), width="stretch", hide_index=True)


# ==========================================================
# DATA EXPLORER
# ==========================================================

elif page == "Data Explorer":

    st.subheader("Interactive Data Explorer")
    st.caption("Explore distributions, feature relationships and statistical patterns.")

    tab1, tab2, tab3 = st.tabs([
        "Feature Analysis",
        "Relationships",
        "Correlation Analysis"
    ])

    with tab1:

        feature = st.selectbox(
            "Choose a measurement",
            FEATURES
        )

        chart = st.selectbox(
            "Chart type",
            ["Histogram", "Box Plot", "Violin Plot"]
        )

        if chart == "Histogram":

            fig = px.histogram(
                df,
                x=feature,
                color=TARGET,
                marginal="box",
                barmode="overlay",
                opacity=0.75
            )

        elif chart == "Box Plot":

            fig = px.box(
                df,
                x=TARGET,
                y=feature,
                color=TARGET,
                points="all"
            )

        else:

            fig = px.violin(
                df,
                x=TARGET,
                y=feature,
                color=TARGET,
                box=True,
                points="all"
            )

        st.plotly_chart(style_chart(fig), width="stretch")

    with tab2:

        x_feature = st.selectbox(
            "Horizontal feature",
            FEATURES,
            index=2
        )

        y_feature = st.selectbox(
            "Vertical feature",
            FEATURES,
            index=3
        )

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=TARGET,
            symbol=TARGET,
            hover_data=FEATURES,
            size_max=12
        )

        st.plotly_chart(style_chart(fig), width="stretch")

    with tab3:

        corr = df[FEATURES].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(style_chart(fig), width="stretch")

        st.markdown("### Correlation Values")
        st.dataframe(corr.round(3), width="stretch")


# ==========================================================
# SINGLE PREDICTION STUDIO
# ==========================================================

elif page == "Predict Studio":

    st.subheader("Intelligent Flower Prediction")
    st.caption("Enter flower measurements to predict the species.")

    presets = {
        "Custom Values": [5.1, 3.5, 1.4, 0.2],
        "Sample A": [5.1, 3.5, 1.4, 0.2],
        "Sample B": [6.0, 2.9, 4.5, 1.5],
        "Sample C": [6.5, 3.0, 5.5, 2.0]
    }

    preset = st.selectbox("Choose a sample", list(presets))
    default_values = presets[preset]

    with st.form("single_prediction"):

        left, right = st.columns(2)

        with left:

            sl = st.number_input(
                "Sepal Length (cm)",
                0.1, 10.0, float(default_values[0]), 0.1
            )

            sw = st.number_input(
                "Sepal Width (cm)",
                0.1, 10.0, float(default_values[1]), 0.1
            )

        with right:

            pl = st.number_input(
                "Petal Length (cm)",
                0.1, 10.0, float(default_values[2]), 0.1
            )

            pw = st.number_input(
                "Petal Width (cm)",
                0.1, 10.0, float(default_values[3]), 0.1
            )

        predict_button = st.form_submit_button(
            "✨ Predict Species",
            type="primary",
            width="stretch"
        )

    if predict_button:

        try:

            model = get_model()

            sample = pd.DataFrame(
                [[sl, sw, pl, pw]],
                columns=FEATURES
            )

            prediction = model.predict(sample)[0]

            probabilities = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]

            st.divider()

            st.success(f"Predicted Species: {str(prediction).title()}")

            if probabilities is not None:

                confidence = float(np.max(probabilities)) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                probability_df = pd.DataFrame({
                    "Species": model.classes_,
                    "Probability": probabilities
                })

                fig = px.bar(
                    probability_df,
                    x="Species",
                    y="Probability",
                    color="Species",
                    text=probability_df["Probability"].map(
                        lambda x: f"{x:.1%}"
                    ),
                    range_y=[0, 1]
                )

                st.plotly_chart(style_chart(fig), width="stretch")

            history_record = {
                "Sepal Length": sl,
                "Sepal Width": sw,
                "Petal Length": pl,
                "Petal Width": pw,
                "Predicted Species": str(prediction)
            }

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append(history_record)

        except Exception as error:
            st.error(f"Prediction error: {error}")

    if st.session_state.get("history"):

        st.divider()
        st.markdown("### Prediction History")

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Prediction History",
            history_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ==========================================================
# BATCH PREDICTION
# ==========================================================

elif page == "Batch Prediction":

    st.subheader("Batch Prediction Centre")
    st.caption("Upload a CSV containing the four required measurement columns.")

    st.info(
        "Required columns: " + ", ".join(FEATURES)
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        current_signature = upload_signature(uploaded_file)
        if st.session_state.get("batch_upload_signature") != current_signature:
            st.session_state.pop("batch_results", None)
            st.session_state["batch_upload_signature"] = current_signature

        try:
            batch_input = validate_batch_csv(uploaded_file.getvalue())
            batch_df = pd.read_csv(uploaded_file)

            if True:

                    st.markdown("### Uploaded Data Preview")
                    st.dataframe(
                        batch_input.head(10),
                        width="stretch",
                        hide_index=True
                    )

                    if st.button(
                        "Run Batch Predictions",
                        type="primary"
                    ):

                        model = get_model()

                        predictions = model.predict(batch_input)

                        output_df = batch_input.copy()
                        output_df["Predicted Species"] = predictions

                        if hasattr(model, "predict_proba"):

                            probabilities = model.predict_proba(
                                batch_input
                            )

                            output_df["Confidence"] = (
                                probabilities.max(axis=1) * 100
                            ).round(2)

                        st.session_state.batch_results = output_df

                        st.success(
                            f"Successfully processed {len(output_df)} records."
                        )

                if "batch_results" in st.session_state:

                    st.markdown("### Batch Prediction Results")

                    output_df = st.session_state.batch_results

                    st.dataframe(
                        output_df,
                        width="stretch",
                        hide_index=True
                    )

                    distribution = (
                        output_df["Predicted Species"]
                        .value_counts()
                        .rename_axis("Species")
                        .reset_index(name="Predictions")
                    )

                    fig = px.bar(
                        distribution,
                        x="Species",
                        y="Predictions",
                        color="Species"
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        width="stretch"
                    )

                    st.download_button(
                        "Download Batch Results",
                        output_df.to_csv(index=False).encode("utf-8"),
                        "iris_batch_predictions.csv",
                        "text/csv"
                    )

        except Exception as error:
            st.error(f"Unable to process file: {error}")


# ==========================================================
# MODEL INTELLIGENCE
# ==========================================================

elif page == "Model Intelligence":

    st.subheader("Model Intelligence Centre")
    st.caption("Compare classifiers and inspect the selected model.")

    if results.empty:

        st.warning("Model results are unavailable.")

        if st.button("Train Models", type="primary"):

            with st.spinner("Training five machine learning models..."):
                train_and_evaluate()

            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    else:

        st.markdown("### Model Comparison")

        st.dataframe(
            results,
            width="stretch",
            hide_index=True
        )

        if "cv_accuracy" in results.columns:

            fig = px.bar(
                results.sort_values(
                    "cv_accuracy",
                    ascending=False
                ),
                x="model",
                y="cv_accuracy",
                color="model",
                text=results.sort_values(
                    "cv_accuracy",
                    ascending=False
                )["cv_accuracy"].map(lambda x: f"{x:.2%}")
            )

            st.plotly_chart(style_chart(fig), width="stretch")

        st.divider()

        if metadata:

            st.markdown("### Selected Model Evaluation")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Test Accuracy",
                f"{metadata.get('test_accuracy', 0):.2%}"
            )

            c2.metric(
                "Precision",
                f"{metadata.get('macro_precision', 0):.2%}"
            )

            c3.metric(
                "Recall",
                f"{metadata.get('macro_recall', 0):.2%}"
            )

            c4.metric(
                "Macro F1",
                f"{metadata.get('macro_f1', 0):.2%}"
            )

            if metadata.get("macro_roc_auc_ovr") is not None:

                st.metric(
                    "Macro ROC-AUC (One-vs-Rest)",
                    f"{metadata['macro_roc_auc_ovr']:.4f}"
                )

        # Confusion matrix
        confusion_path = ROOT / "outputs" / "confusion_matrix.csv"

        if confusion_path.exists():

            st.markdown("### Confusion Matrix")

            confusion = pd.read_csv(
                confusion_path,
                index_col=0
            )

            fig = px.imshow(
                confusion,
                text_auto=True,
                color_continuous_scale="Purples",
                labels={
                    "x": "Predicted",
                    "y": "Actual",
                    "color": "Count"
                }
            )

            st.plotly_chart(style_chart(fig), width="stretch")

        # ROC curves
        roc_path = ROOT / "outputs" / "roc_curve_data.csv"

        if roc_path.exists():

            st.markdown("### ROC Curve Analysis")

            roc_df = pd.read_csv(roc_path)

            fig = px.line(
                roc_df,
                x="false_positive_rate",
                y="true_positive_rate",
                color="class",
                title="One-vs-Rest ROC Curves"
            )

            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="#8290ad")
            )

            st.plotly_chart(style_chart(fig), width="stretch")

        # Feature importance
        importance_path = ROOT / "outputs" / "feature_importance.csv"

        if importance_path.exists():

            st.markdown("### Feature Importance")

            importance_df = pd.read_csv(importance_path)

            fig = px.bar(
                importance_df.sort_values("importance_mean"),
                x="importance_mean",
                y="feature",
                orientation="h",
                error_x="importance_std",
                color="importance_mean",
                color_continuous_scale="Purples"
            )

            st.plotly_chart(style_chart(fig), width="stretch")

        # Classification report
        report_path = ROOT / "outputs" / "classification_report.txt"

        if report_path.exists():

            with st.expander("View Detailed Classification Report"):

                st.code(
                    report_path.read_text(encoding="utf-8"),
                    language="text"
                )

        # Model metadata
        with st.expander("View Training Metadata"):

            st.json(metadata)


# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.subheader("About Iris Intelligence Pro")

    st.markdown("""
    ### Project Objective

    This application demonstrates a complete machine learning
    classification workflow using the Iris flower dataset.

    ### Dataset Features

    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    ### Target Classes

    - Setosa
    - Versicolor
    - Virginica

    ### Machine Learning Algorithms

    1. Logistic Regression
    2. K-Nearest Neighbors
    3. Support Vector Machine
    4. Random Forest
    5. Gradient Boosting

    ### Technical Stack

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib

    ### Evaluation Methodology

    - Stratified train-test split
    - Five-fold stratified cross-validation
    - GridSearchCV hyperparameter optimization
    - Held-out test evaluation
    - Classification report
    - Confusion matrix
    - ROC-AUC analysis
    - Permutation feature importance

    ### Important Note

    Cross-validation is used for model selection.
    The held-out test set is used for final evaluation.
    Reported performance depends on the dataset and split.
    """)

    st.divider()

    st.markdown("### Project Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch",
        hide_index=True
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8290ad; padding:12px;">
        IRIS INTELLIGENCE PRO · MACHINE LEARNING ANALYTICS PLATFORM
        <br>
        Python · Scikit-learn · Streamlit · Plotly
    </div>
    """,
    unsafe_allow_html=True
)
