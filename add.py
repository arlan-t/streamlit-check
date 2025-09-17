
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telecom Churn — Data Story", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    # Try to coerce common churn column to boolean
    target = None
    for c in df.columns:
        if c.lower() in ("churn","is_churn","churned","churn_flag","exited"):
            target = c
            break
    def coerce_bool_series(s):
        if s.dtype == object:
            mapped = s.astype(str).str.strip().str.lower().map({
                'yes': True, 'y': True, 'true': True, '1': True, 't': True,
                'no': False, 'n': False, 'false': False, '0': False, 'f': False
            })
            if mapped.notna().mean() > 0.8:
                return mapped
        return s
    if target is not None:
        df[target] = coerce_bool_series(df[target])
    return df, target

df, target = load_data()

st.title("📉 Telecom Churn — Data Storyline")
st.write("Interactive exploration of the churn dataset with KPIs, charts, correlations, and insights.")

# Sidebar
st.sidebar.header("Settings")
show_raw = st.sidebar.checkbox("Show raw data sample", value=False)
sample_n = st.sidebar.slider("Rows to sample", min_value=5, max_value=100, value=20, step=5)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    st.metric("Numeric cols", len(num_cols))
with col4:
    if target is not None and (pd.api.types.is_bool_dtype(df[target]) or pd.api.types.is_numeric_dtype(df[target])):
        rate = float(pd.Series(df[target]).mean())
        st.metric("Churn rate", f"{rate*100:.1f}%")
    else:
        st.metric("Churn rate", "N/A")

if show_raw:
    st.dataframe(df.sample(sample_n, random_state=42))

# Churn distribution
if target is not None:
    st.subheader("Churn Distribution")
    counts = df[target].value_counts(dropna=False).sort_index()
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title("Churn distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    st.pyplot(fig)

# Correlation heatmap
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) >= 2:
    st.subheader("Correlation Heatmap (Numeric Features)")
    corr = df[num_cols].corr(numeric_only=True)
    fig = plt.figure()
    plt.imshow(corr, interpolation='nearest', aspect='auto')
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation heatmap (Pearson)")
    plt.colorbar()
    st.pyplot(fig)

# Churn by categorical features (top by deviation)
if target is not None and (pd.api.types.is_bool_dtype(df[target]) or pd.api.types.is_numeric_dtype(df[target])):
    st.subheader("Churn Rate by Category (Top Signals)")
    global_rate = df[target].mean()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target and df[c].nunique(dropna=False) <= 12]
    scores = []
    for c in cat_cols:
        try:
            rates = df.groupby(c, dropna=False)[target].mean()
            weight = df[c].value_counts(normalize=True, dropna=False).reindex(rates.index).fillna(0)
            score = float((weight * (rates - global_rate).abs()).sum())
            scores.append((c, score))
        except Exception:
            pass
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:4]
    for c, _ in scores:
        grp = df.groupby(c, dropna=False)[target].mean().sort_values(ascending=False)
        fig = plt.figure()
        grp.plot(kind="bar")
        plt.title(f"Churn rate by {c}")
        plt.ylabel("Churn rate")
        plt.xlabel(c)
        st.pyplot(fig)

# Scatter explorers for common telecom fields (if present)
st.subheader("Exploratory Plots")
pairs = [("tenure","MonthlyCharges"), ("tenure","TotalCharges"), ("SeniorCitizen","MonthlyCharges")]
for x, y in pairs:
    if x in df.columns and y in df.columns:
        fig = plt.figure()
        plt.scatter(df[x], df[y], s=10, alpha=0.6)
        plt.title(f"{y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot(fig)

st.markdown("""
### Key Insight Prompts
- Which categories show the largest deviation from the global churn rate?
- Are higher monthly charges associated with higher churn?
- Does churn concentrate in early-tenure customers?
""")
