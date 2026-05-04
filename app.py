from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"

FEATURE_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

FEATURE_LABELS = {
    "ph": "pH Level",
    "Hardness": "Hardness",
    "Solids": "Total Solids",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Conductivity",
    "Organic_carbon": "Organic Carbon",
    "Trihalomethanes": "Trihalomethanes",
    "Turbidity": "Turbidity",
}

FEATURE_HELP = {
    "ph": "Acidity or alkalinity of the water sample.",
    "Hardness": "Concentration of minerals such as calcium and magnesium.",
    "Solids": "Amount of dissolved solids in the sample.",
    "Chloramines": "Disinfectant-related chemical measurement.",
    "Sulfate": "Sulfate concentration in the water sample.",
    "Conductivity": "Ability of the water to conduct electrical current.",
    "Organic_carbon": "Amount of organic carbon present in the sample.",
    "Trihalomethanes": "Chemical compounds that may form during disinfection.",
    "Turbidity": "Cloudiness or haziness of the water sample.",
}

PRESETS = {
    "Custom values": {
        "ph": 7.0,
        "Hardness": 196.0,
        "Solids": 22000.0,
        "Chloramines": 7.0,
        "Sulfate": 333.0,
        "Conductivity": 426.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 66.0,
        "Turbidity": 4.0,
    },
    "Sample potable": {
        "ph": 7.30,
        "Hardness": 143.43,
        "Solids": 46718.56,
        "Chloramines": 4.77,
        "Sulfate": 252.47,
        "Conductivity": 446.84,
        "Organic_carbon": 12.58,
        "Trihalomethanes": 60.65,
        "Turbidity": 3.16,
    },
    "Sample not potable": {
        "ph": 8.32,
        "Hardness": 214.37,
        "Solids": 22018.42,
        "Chloramines": 8.05,
        "Sulfate": 356.89,
        "Conductivity": 363.27,
        "Organic_carbon": 18.44,
        "Trihalomethanes": 100.34,
        "Turbidity": 4.63,
    },
}

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main .block-container {
            max-width: 1150px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef7f6 100%);
        }

        .hero {
            padding: 2.4rem 2.2rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(45, 212, 191, 0.32), transparent 32%),
                radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.20), transparent 36%),
                linear-gradient(135deg, #0f172a 0%, #164e63 52%, #0f766e 100%);
            color: white;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.24);
            margin-bottom: 1.3rem;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.42rem 0.72rem;
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            background: rgba(255,255,255,0.11);
            color: rgba(255,255,255,0.88);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.1rem, 4vw, 3.65rem);
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: -0.065em;
            margin: 0 0 0.85rem 0;
        }

        .hero-subtitle {
            max-width: 760px;
            color: rgba(255,255,255,0.82);
            font-size: 1.02rem;
            line-height: 1.7;
            margin: 0;
        }

        .card {
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 26px;
            padding: 1.35rem;
            box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .section-label {
            color: #0f766e;
            font-size: 0.76rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 0.4rem;
        }

        .muted {
            color: #64748b;
            font-size: 0.94rem;
            line-height: 1.65;
        }

        .result-card {
            margin-top: 1rem;
            border-radius: 28px;
            padding: 1.75rem 1.35rem;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.10);
        }

        .potable {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            color: #065f46;
            border-color: #86efac;
        }

        .not-potable {
            background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
            color: #991b1b;
            border-color: #fecaca;
        }

        .result-small {
            font-size: 0.82rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            opacity: 0.72;
        }

        .result-large {
            font-size: clamp(2.2rem, 4vw, 3.35rem);
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.055em;
            margin-top: 0.35rem;
        }

        .feature-pill {
            display: inline-block;
            margin: 0.24rem 0.18rem;
            padding: 0.48rem 0.72rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            border: 1px solid rgba(148, 163, 184, 0.24);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .footer {
            color: #94a3b8;
            font-size: 0.82rem;
            text-align: center;
            margin-top: 1.2rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 900;
        }

        .stButton > button {
            border-radius: 16px;
            min-height: 3.15rem;
            font-weight: 900;
            letter-spacing: -0.01em;
        }

        .stNumberInput input {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning Final Project</div>
        <div class="hero-title">Water Potability Prediction</div>
        <p class="hero-subtitle">
            Enter water quality measurements and generate a prediction using the deployed machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Input Measurements")
    st.caption("Choose a sample or enter your own water-quality values.")

    sample_choice = st.selectbox(
        "Input preset",
        ["Custom values", "Sample potable", "Sample not potable"],
    )

    def default_value(feature: str) -> float:
        return float(PRESETS[sample_choice][feature])

    with st.expander("Basic water indicators", expanded=True):
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=default_value("ph"), step=0.1, help=FEATURE_HELP["ph"])
        hardness = st.number_input("Hardness", min_value=0.0, value=default_value("Hardness"), step=1.0, help=FEATURE_HELP["Hardness"])
        solids = st.number_input("Total Solids", min_value=0.0, value=default_value("Solids"), step=100.0, help=FEATURE_HELP["Solids"])

    with st.expander("Chemical measurements", expanded=True):
        chloramines = st.number_input("Chloramines", min_value=0.0, value=default_value("Chloramines"), step=0.1, help=FEATURE_HELP["Chloramines"])
        sulfate = st.number_input("Sulfate", min_value=0.0, value=default_value("Sulfate"), step=1.0, help=FEATURE_HELP["Sulfate"])
        conductivity = st.number_input("Conductivity", min_value=0.0, value=default_value("Conductivity"), step=1.0, help=FEATURE_HELP["Conductivity"])

    with st.expander("Additional indicators", expanded=False):
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=default_value("Organic_carbon"), step=0.1, help=FEATURE_HELP["Organic_carbon"])
        trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=default_value("Trihalomethanes"), step=0.1, help=FEATURE_HELP["Trihalomethanes"])
        turbidity = st.number_input("Turbidity", min_value=0.0, value=default_value("Turbidity"), step=0.1, help=FEATURE_HELP["Turbidity"])

inputs = {
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity,
}

input_df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
preview_df = input_df.rename(columns=FEATURE_LABELS)

main_col, side_col = st.columns([1.45, 0.85], gap="large")

with main_col:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Prediction</div>
            <div class="section-title">Analyze Water Sample</div>
            <div class="muted">Review the entered measurements, then run the model to display the classification result and probability score.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    predict_clicked = st.button("Predict Potability", type="primary", use_container_width=True)

    if predict_clicked:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        potable_probability = float(probabilities[1])
        not_potable_probability = float(probabilities[0])

        if prediction == 1:
            st.markdown(
                """
                <div class="result-card potable">
                    <div class="result-small">Model Prediction</div>
                    <div class="result-large">POTABLE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card not-potable">
                    <div class="result-small">Model Prediction</div>
                    <div class="result-large">NOT POTABLE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("Potable Probability", f"{potable_probability:.2%}")
        prob_col2.metric("Not Potable Probability", f"{not_potable_probability:.2%}")
        st.progress(potable_probability, text="Potability probability score")

        if 0.45 <= potable_probability <= 0.55:
            st.info("The model probability is close to the decision boundary, so the prediction is less confident.")
        elif prediction == 1:
            st.success("The entered values are closer to patterns labeled as potable in the training dataset.")
        else:
            st.warning("The entered values are closer to patterns labeled as not potable in the training dataset.")
    else:
        st.markdown(
            """
            <div class="card">
                <strong>Ready for prediction.</strong><br>
                <span class="muted">Click the button above after entering the sample values.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with side_col:
    st.markdown(
        """
        <div class="card">
            <div class="section-label">Input Guide</div>
            <div class="section-title">Key Measurements</div>
            <div class="muted">The model evaluates all measurements together. These are the most useful values to focus on when testing the app:</div>
            <div style="margin-top:0.85rem;">
                <span class="feature-pill">pH</span>
                <span class="feature-pill">Sulfate</span>
                <span class="feature-pill">Hardness</span>
                <span class="feature-pill">Chloramines</span>
                <span class="feature-pill">Solids</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Water Potability Prediction System · PTF03 Machine Learning Final Project</div>',
    unsafe_allow_html=True,
)
