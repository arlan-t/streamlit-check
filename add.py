import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import sklearn

st.set_page_config(page_title="Telecom Churn — EDA & t-SNE", layout="wide")
sns.set_theme(style="ticks")

# -----------------------------
# Data loading & preprocessing
# -----------------------------
@st.cache_data
def load_data(path: str = "telecom_churn.csv"):
    df = pd.read_csv(path)

    # Normalize churn to boolean if possible
    if "Churn" in df.columns:
        if df["Churn"].dtype == bool:
            pass
        else:
            df["Churn"] = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": True, "true": True, "1": True, "no": False, "false": False, "0": False})
            )
    return df

df = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
show_raw = st.sidebar.checkbox("Show raw sample", value=False)
sample_n = st.sidebar.slider("Sample rows to show", 5, 100, 20, 5)

heavy_plots = st.sidebar.checkbox("Enable heavy plots (pairplot, t-SNE)", value=False)

st.sidebar.subheader("t-SNE Parameters")
tsne_seed = st.sidebar.number_input("Random seed", value=17, step=1)
tsne_iter = st.sidebar.number_input("Max iterations (n_iter)", value=500, step=100)
desired_perplexity = st.sidebar.number_input("Desired perplexity", value=30.0, step=5.0, format="%.1f")

# -----------------------------
# Header & KPIs
# -----------------------------
st.title("📊 Telecom Churn — Data Storyline & Visual EDA")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Rows", df.shape[0])
with k2:
    st.metric("Columns", df.shape[1])
with k3:
    n_numeric = df.select_dtypes(include=[np.number]).shape[1]
    st.metric("Numeric features", n_numeric)
with k4:
    if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
        st.metric("Churn rate", f"{df['Churn'].mean()*100:.1f}%")
    else:
        st.metric("Churn rate", "N/A")

if show_raw:
    st.dataframe(df.sample(sample_n, random_state=42))

# -----------------------------
# Churn distribution
# -----------------------------
if "Churn" in df.columns:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# Histograms & Density
# -----------------------------
st.subheader("Histograms & Density (Analytical Distributions)")
features = [c for c in ["Total day minutes", "Total intl calls"] if c in df.columns]
if features:
    c1, c2 = st.columns(2)
    with c1:
        fig, axes = plt.subplots(1, len(features), figsize=(10, 4))
        if len(features) == 1:
            axes = [axes]
        df[features].hist(ax=axes)
        fig.suptitle("Histograms", y=1.02)
        st.pyplot(fig)

    with c2:
        fig, axes = plt.subplots(1, len(features), figsize=(10, 4))
        if len(features) == 1:
            axes = [axes]
        df[features].plot(kind="density", subplots=True, layout=(1, len(features)), sharex=False, ax=axes)
        fig.suptitle("Kernel Density Estimates", y=1.02)
        st.pyplot(fig)

# -----------------------------
# Boxplot & Violin for "Total intl calls"
# -----------------------------
if "Total intl calls" in df.columns:
    st.subheader("Outliers & Distribution Shape — Total intl calls")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    sns.boxplot(y=df["Total intl calls"], ax=axes[0])
    axes[0].set_title("Boxplot")
    sns.violinplot(y=df["Total intl calls"], ax=axes[1])
    axes[1].set_title("Violin")
    st.pyplot(fig)

# -----------------------------
# Correlation Heatmap (drop charges to reduce redundancy)
# -----------------------------
st.subheader("Correlation Heatmap (Numeric Features)")
redundant = {"Total day charge", "Total eve charge", "Total night charge", "Total intl charge"}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in redundant]
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Pearson correlation")
    st.pyplot(fig)

# -----------------------------
# Jointplots and LM plots (if columns exist)
# -----------------------------
if all(c in df.columns for c in ["Total day minutes", "Total night minutes"]):
    st.subheader("Bivariate Relationships")
    st.caption("Day vs Night minutes — scatter and KDE")
    g1 = sns.jointplot(x="Total day minutes", y="Total night minutes", data=df, kind="scatter")
    st.pyplot(g1.fig)
    g2 = sns.jointplot(x="Total day minutes", y="Total night minutes", data=df, kind="kde", color="g")
    st.pyplot(g2.fig)

    if "Churn" in df.columns:
        g3 = sns.lmplot(
            x="Total day minutes", y="Total night minutes",
            data=df, hue="Churn", fit_reg=False, aspect=1.2
        )
        st.pyplot(g3.fig)

# -----------------------------
# Boxplots by Churn (top 12 numerics)
# -----------------------------
if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
    st.subheader("Distribution by Churn (Boxplots)")
    # Treat Customer service calls as numeric if present
    extra = ["Customer service calls"] if "Customer service calls" in df.columns else []
    numerical_with_calls = [c for c in num_cols] + extra
    numerical_with_calls = list(dict.fromkeys(numerical_with_calls))  # deduplicate

    if numerical_with_calls:
        take = numerical_with_calls[:12]
        rows = int(np.ceil(len(take) / 4))
        fig, axes = plt.subplots(nrows=rows, ncols=4, figsize=(16, 4*rows))
        axes = np.array(axes).reshape(rows, 4)
        for idx, feat in enumerate(take):
            ax = axes[idx // 4, idx % 4]
            sns.boxplot(x="Churn", y=feat, data=df, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel(feat)
        # hide empty axes
        for j in range(len(take), rows*4):
            axes[j // 4, j % 4].axis("off")
        fig.tight_layout()
        st.pyplot(fig)

# -----------------------------
# Counts by key categoricals
# -----------------------------
st.subheader("Customer Service & Plans — Counts")
row = st.columns(2)
with row[0]:
    if "Customer service calls" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Customer service calls", data=df, ax=ax)
        ax.set_title("Customer service calls — count")
        st.pyplot(fig)
with row[1]:
    if "Customer service calls" in df.columns and "Churn" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Customer service calls", hue="Churn", data=df, ax=ax)
        ax.set_title("Customer service calls × Churn")
        st.pyplot(fig)

row2 = st.columns(2)
with row2[0]:
    if all(c in df.columns for c in ["International plan", "Churn"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="International plan", hue="Churn", data=df, ax=ax)
        ax.set_title("International plan × Churn")
        st.pyplot(fig)
with row2[1]:
    if all(c in df.columns for c in ["Voice mail plan", "Churn"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Voice mail plan", hue="Churn", data=df, ax=ax)
        ax.set_title("Voice mail plan × Churn")
        st.pyplot(fig)

# -----------------------------
# State-wise churn rates (top 10)
# -----------------------------
if all(c in df.columns for c in ["State", "Churn"]) and pd.api.types.is_bool_dtype(df["Churn"]):
    st.subheader("Top States by Churn Rate")
    state_churn = df.groupby("State")["Churn"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    state_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn rate")
    st.pyplot(fig)

# -----------------------------
# Heavy plots: Pairplot & t-SNE
# -----------------------------
if heavy_plots:
    # Pairplot (can be slow)
    st.subheader("Pairplot (selected numeric features)")
    try:
        few = [c for c in num_cols if df[c].nunique() > 5][:6]  # pick up to 6 informative numerics
        if few:
            g = sns.pairplot(df[few], corner=True)
            st.pyplot(g.fig)
        else:
            st.info("Not enough numeric features for a meaningful pairplot.")
    except Exception as e:
        st.warning(f"Pairplot skipped due to: {type(e).__name__}: {e}")

    # -------------------------
    # t-SNE with robust fallbacks
    # -------------------------
    st.subheader("t-SNE Representation of Customers")
    st.markdown("""
**t-distributed Stochastic Neighbor Embedding (t-SNE)** projects a high-dimensional feature space onto 2D while
trying to preserve neighborhoods:
- Points close in the original space stay close on the plane.
- Points far apart remain separated.

It’s useful to **visualize clusters & local structure**.  
**Caveats:** it’s computationally heavy; results vary with the random seed; and the plot is **exploratory**, not proof.

We’ll drop `State` and `Churn`, encode `International plan` and `Voice mail plan` as 0/1, and standardize features.
    """)

    # Prepare X
    if all(c in df.columns for c in ["International plan", "Voice mail plan"]):
        X = df.drop(columns=[c for c in ["Churn", "State"] if c in df.columns], errors="ignore").copy()
        # encode binary Yes/No
        for c in ["International plan", "Voice mail plan"]:
            if X[c].dtype == object:
                X[c] = X[c].map({"Yes": 1, "No": 0})
        # keep numerics only
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        n_samples = X.shape[0]
        # make perplexity valid and reasonable
        max_valid = max(2.0, (n_samples - 1) / 3.0)  # TSNE requires perplexity < n_samples
        perplexity = float(min(max(desired_perplexity, 5.0), max_valid))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build TSNE with version-safe args
        def build_tsne(perp: float, seed: int, n_iter: int):
            """
            Construct TSNE robustly across sklearn versions.
            Try with (random_state, n_iter, perplexity); if TypeError, fallback.
            """
            try:
                return TSNE(random_state=seed, n_iter=int(n_iter), perplexity=float(perp), init="pca")
            except TypeError:
                try:
                    return TSNE(random_state=seed, perplexity=float(perp), init="pca")
                except TypeError:
                    return TSNE(perplexity=float(perp))

        tsne = build_tsne(perplexity, tsne_seed, tsne_iter)

        # Fit, with automatic perplexity downgrade if needed
        fit_error = None
        for _ in range(3):
            try:
                tsne_repr = tsne.fit_transform(X_scaled)
                fit_error = None
                break
            except ValueError as ve:
                # Often: "perplexity must be less than n_samples"
                fit_error = ve
                perplexity = max(5.0, perplexity / 2.0)
                tsne = build_tsne(perplexity, tsne_seed, tsne_iter)
            except Exception as e:
                fit_error = e
                break

        st.caption(f"scikit-learn version: {sklearn.__version__} | final perplexity used: {perplexity:.1f}")

        if fit_error is None:
            # 1) Base t-SNE
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=0.5)
            ax.set_title("t-SNE representation (all customers)")
            st.pyplot(fig)

            # 2) Colored by churn
            if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = df["Churn"].map({False: "blue", True: "orange"})
                ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=colors, alpha=0.5)
                ax.set_title("t-SNE colored by Churn (Blue = Loyal, Orange = Churned)")
                st.pyplot(fig)

            # 3) Colored by International/Voice mail plan
            if all(c in df.columns for c in ["International plan", "Voice mail plan"]):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                for i, name in enumerate(["International plan", "Voice mail plan"]):
                    col = df[name].map({"Yes": "orange", "No": "blue"})
                    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=col, alpha=0.5)
                    axes[i].set_title(name)
                st.pyplot(fig)

                st.markdown("""
**Takeaway (typical in this dataset):**
- Churned customers (orange) often concentrate in specific regions.
- Clusters frequently align with **International plan = Yes**, and sometimes **no Voice mail plan**.
Remember: confirm any hypothesis from t-SNE with targeted analysis.
                """)
        else:
            st.error(f"t-SNE failed: {type(fit_error).__name__}: {fit_error}")
    else:
        st.info("t-SNE section requires 'International plan' and 'Voice mail plan' columns.")
