import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import build_X  # noqa: E402

# Model response_minutes (minutes) pe train hua hai, log pe nahi
LOG_TARGET = False
RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}

MODEL_PATH = ROOT / "data/processed/xgb_model.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Who Waits Longer for Help?", layout="wide")


@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model(str(MODEL_PATH))
    return m


@st.cache_data
def load_lookup():
    return pd.read_csv(ROOT / "app/zip_lookup.csv", dtype={"zipcode": str})


model = load_model()
lk = load_lookup()

st.title("Who Waits Longer for Help?")
st.caption("NYC EMS response-time inequality: decision-support demo (2022 to 2023 data)")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Scenario maps", "Explainability", "Limitations"])

with tab1:
    zips = sorted(lk["zipcode"].unique())
    default_zip = zips.index("10001") if "10001" in zips else 0

    c1, c2 = st.columns(2)
    with c1:
        zipcode = st.selectbox("ZIP code", zips, index=default_zip)
        severity = st.slider("Incident severity level (1 = most critical)", 1, 8, 4)
    with c2:
        hour = st.slider("Hour of day", 0, 23, 18)
        day = st.selectbox("Day of week", DAYS, index=2)
        month = st.slider("Month", 1, 12, 6)

    row = lk[lk["zipcode"] == zipcode].iloc[0].to_dict()
    dow = DAYS.index(day)
    row.update(
        initial_severity_level_code=severity,
        hour=hour,
        dayofweek=dow,
        month=month,
        is_weekend=int(dow >= 5),
        is_night=int(hour >= 22 or hour < 5),
        is_rush_hour=int(hour in RUSH_HOURS),
    )
    X = build_X(pd.DataFrame([row]))
    X = X.reindex(columns=model.get_booster().feature_names, fill_value=0.0)

    pred_min = float(model.predict(X)[0])      # model ka output minutes me hai
    if LOG_TARGET:
        pred_min = float(np.expm1(pred_min))

    st.metric("Predicted response time", f"{pred_min:.1f} min", f"{pred_min * 60:.0f} seconds")
    st.write(f"Borough: **{str(row['borough']).title()}**, "
             f"Median income: **${row['median_hh_income']:,.0f}**")
    st.info("Ye ek statistical estimate hai, kisi individual call ki guarantee nahi.")

with tab2:
    for name, cap in [
        ("choropleth_scenarios.png", "Median response time by ZIP, scenario-wise"),
        ("choropleth_differences.png", "Scenario minus overall"),
        ("choropleth_income_vs_delay.png", "Income vs response delay"),
    ]:
        p = ROOT / "reports" / name
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)

with tab3:
    p = ROOT / "reports/shap_summary.png"
    if p.exists():
        st.image(str(p), caption="SHAP feature importance", use_container_width=True)
    st.write("Features jitne upar, model ke prediction par utna zyada asar.")

with tab4:
    st.markdown(
        """
- **Association, not causation.** Ye data dikhata hai ki kya saath-saath hota hai, kyun hota hai wo nahi.
- **ZIP-level analysis.** Area ke averages individual logon ke baare mein nahi batate (ecological fallacy).
- **Demographic features.** Model pct_black / pct_hispanic use karta hai; isse unfair behaviour ka risk hai, isliye fairness audit zaroori hai.
- **Crime data use nahi hua.** Raat ke pattern ko crime ka effect nahi kaha ja sakta.
- **Sirf 2022 to 2023.** Naye saalon par performance alag ho sakti hai.
        """
    )