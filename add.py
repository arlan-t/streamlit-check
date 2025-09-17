import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ===== Page & style =====
st.set_page_config(page_title="Telecom Churn — Compact EDA", layout="wide")
sns.set_theme(style="ticks")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# ===== Utils =====
def plot(fig):
    """Render and close matplotlib figure, always contained to column width."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def two_cols():
    return st.columns(2, gap="small")

# ===== Data =====
@st.cache_data
def load_data(path: str = "telecom_churn.csv"):
    df = pd.read_csv(path)
    # normalize Churn to bool
    if "Churn" in df.columns and not pd.api.types.is_bool_dtype(df["Churn"]):
        df["Churn"] = (
            df["Churn"].astype(str).str.strip().str.lower()
            .map({"yes": True, "true": True, "1": True,
                  "no": False, "false": False, "0": False})
        )
    return df

df = load_data()

# ===== Sidebar =====
st.sidebar.header("Controls")
show_raw = st.sidebar.checkbox("Show raw sample", value=False)
sample_n = st.sidebar.slider("Sample rows to show", 5, 100, 20, 5)
show_heavy = st.sidebar.checkbox("Show heavy plots (pairplot & t-SNE)", value=False)

st.sidebar.subheader("t-SNE params")
tsne_seed = st.sidebar.number_input("Random seed", value=17, step=1)
tsne_iter = st.sidebar.number_input("Iterations (n_iter)", value=500, step=100)
desired_perp = st.sidebar.number_input("Desired perplexity", value=30.0, step=5.0, format="%.1f")

# ===== Header & KPIs =====
st.title("📊 Telecom Churn — Interactive Dashboard")

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Rows", df.shape[0])
with k2: st.metric("Columns", df.shape[1])
with k3: st.metric("Numeric cols", df.select_dtypes(include=[np.number]).shape[1])
with k4:
    if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
        st.metric("Churn rate", f"{df['Churn'].mean()*100:.1f}%")
    else:
        st.metric("Churn rate", "N/A")

if show_raw:
    st.dataframe(df.sample(sample_n, random_state=42))

# ===== Section: Churn & Calls =====
st.subheader("Churn & Service Interactions")
c1, c2 = two_cols()

with c1:
    if "Churn" in df.columns:
        fig = px.histogram(df, x="Churn", color="Churn",
                           title="Churn distribution")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if "Customer service calls" in df.columns:
        fig = px.histogram(df, x="Customer service calls", color="Churn" if "Churn" in df else None,
                           title="Customer service calls — count")
        st.plotly_chart(fig, use_container_width=True)

# ===== Section: Distributions (Hist & KDE) =====
st.subheader("Key Feature Distributions")
features = [c for c in ["Total day minutes", "Total intl calls"] if c in df.columns]

if features:
    c1, c2 = two_cols()
    with c1:
        fig = px.histogram(df, x=features[0], nbins=30, color="Churn" if "Churn" in df else None,
                           marginal="box", opacity=0.7, title=f"Histogram — {features[0]}")
        st.plotly_chart(fig, use_container_width=True)

    if len(features) > 1:
        with c2:
            fig = px.histogram(df, x=features[1], nbins=30, color="Churn" if "Churn" in df else None,
                               marginal="violin", opacity=0.7, title=f"Histogram — {features[1]}")
            st.plotly_chart(fig, use_container_width=True)

# ===== Section: Outliers & Shape (Box/Violin) =====
if "Total intl calls" in df.columns:
    st.subheader("Outliers & Shape — Total intl calls")
    c1, c2 = two_cols()
    with c1:
        fig = px.box(df, y="Total intl calls", color="Churn" if "Churn" in df else None,
                     points="all", title="Boxplot")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(df, y="Total intl calls", color="Churn" if "Churn" in df else None,
                        box=True, points="all", title="Violin")
        st.plotly_chart(fig, use_container_width=True)

# ===== Section: Correlations =====
st.subheader("Correlation (numeric) — charges dropped to reduce redundancy")
redundant = {"Total day charge", "Total eve charge", "Total night charge", "Total intl charge"}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in redundant]
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 5))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", cbar=True, square=True)
    ax.set_title("Correlation Heatmap")
    plot(fig)

# ===== Section: Bivariate (Day vs Night minutes) =====
if all(c in df.columns for c in ["Total day minutes", "Total night minutes"]):
    st.subheader("Bivariate Relationships — Day vs Night minutes")
    c1, c2 = two_cols()
    with c1:
        fig = px.scatter(df, x="Total day minutes", y="Total night minutes",
                         color="Churn" if "Churn" in df else None,
                         opacity=0.6, title="Scatter — Day vs Night")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(constrained_layout=True)
        sns.kdeplot(
            x="Total day minutes", y="Total night minutes",
            data=df, fill=True, levels=30, thresh=0.05, ax=ax
        )
        ax.set_title("2D KDE — Day vs Night")
        plot(fig)

# ===== Section: Boxplots by Churn =====
if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
    st.subheader("Distribution by Churn")
    f1 = "Total day minutes" if "Total day minutes" in df.columns else num_cols[0]
    f2 = "Customer service calls" if "Customer service calls" in df.columns else num_cols[min(1, len(num_cols)-1)]
    c1, c2 = two_cols()
    with c1:
        fig = px.box(df, x="Churn", y=f1, color="Churn",
                     title=f"{f1} by Churn", points="all")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x="Churn", y=f2, color="Churn",
                     title=f"{f2} by Churn", points="all")
        st.plotly_chart(fig, use_container_width=True)

# ===== Section: Plans vs Churn =====
if all(c in df.columns for c in ["International plan", "Voice mail plan", "Churn"]):
    st.subheader("Plans × Churn")
    c1, c2 = two_cols()
    with c1:
        fig = px.histogram(df, x="International plan", color="Churn",
                           barmode="group", text_auto=True,
                           title="International plan × Churn")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Voice mail plan", color="Churn",
                           barmode="group", text_auto=True,
                           title="Voice mail plan × Churn")
        st.plotly_chart(fig, use_container_width=True)

# ===== Section: State churn (top) =====
if all(c in df.columns for c in ["State", "Churn"]):
    st.subheader("Top States by Churn rate")
    top_states = df.groupby("State")["Churn"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(top_states, x=top_states.index, y=top_states.values,
                 labels={"x": "State", "y": "Churn rate"},
                 title="Top States by Churn rate")
    st.plotly_chart(fig, use_container_width=True)

# ===== Heavy plots: Pairplot & t-SNE =====
if show_heavy:
    st.subheader("Heavy Plots")

    # --- Pairplot (compact in expander)
    with st.expander("Pairplot (compact)"):
        few = [c for c in num_cols if df[c].nunique() > 5][:6]
        if len(few) >= 2:
            g = sns.pairplot(df[few], corner=True, height=1.5)
            st.pyplot(g.fig, use_container_width=True)
            plt.close(g.fig)
        else:
            st.info("Not enough numeric features for a meaningful pairplot.")

    # --- t-SNE explanation
    st.markdown("""
**t-distributed Stochastic Neighbor Embedding (t-SNE)** projects a high-dimensional feature space onto 2D
while trying to preserve neighborhoods (close points stay close; far points remain far).
It’s exploratory, computationally heavy, and sensitive to random seed — use it to generate hypotheses, not proofs.
""")

    def build_tsne(seed, n_iter, perp):
        """Construct TSNE robustly across sklearn versions."""
        try:
            return TSNE(random_state=int(seed), n_iter=int(n_iter), perplexity=float(perp), init="pca")
        except TypeError:
            try:
                return TSNE(random_state=int(seed), perplexity=float(perp))
            except TypeError:
                return TSNE(perplexity=float(perp))

    # Prep data
    X = df.drop(columns=[c for c in ["Churn", "State"] if c in df.columns], errors="ignore").copy()
    for c in ["International plan", "Voice mail plan"]:
        if c in X.columns and X[c].dtype == object:
            X[c] = X[c].map({"Yes": 1, "No": 0})
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    n_samples = X.shape[0]
    max_valid = max(2.0, (n_samples - 1) / 3.0)
    perp = float(min(max(desired_perp, 5.0), max_valid))

    X_scaled = StandardScaler().fit_transform(X)
    tsne = build_tsne(tsne_seed, tsne_iter, perp)
    tsne_repr = tsne.fit_transform(X_scaled)

    # Two-column t-SNE plots
    c1, c2 = two_cols()
    with c1:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=0.5, s=10, c="#777777", label="Customers")
        ax.set_title("t-SNE (all customers)")
        ax.legend()
        plot(fig)
    with c2:
        if "Churn" in df.columns and pd.api.types.is_bool_dtype(df["Churn"]):
            fig, ax = plt.subplots(constrained_layout=True)
            colors = df["Churn"].map({False: "tab:blue", True: "tab:orange"}).values
            ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=colors, alpha=0.55, s=12)
            ax.set_title("t-SNE colored by Churn")
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label="Loyal (blue)", markerfacecolor="tab:blue", markersize=6),
                plt.Line2D([0], [0], marker="o", color="w", label="Churned (orange)", markerfacecolor="tab:orange", markersize=6),
            ]
            ax.legend(handles=handles, loc="best", frameon=False)
            plot(fig)

    c3, c4 = two_cols()
    with c3:
        fig, ax = plt.subplots(constrained_layout=True)
        col = df["International plan"].map({"Yes": "tab:orange", "No": "tab:blue"}).values
        ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=col, alpha=0.5, s=12)
        ax.set_title("t-SNE by International plan")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label="No plan (blue)", markerfacecolor="tab:blue", markersize=6),
            plt.Line2D([0], [0], marker="o", color="w", label="Yes plan (orange)", markerfacecolor="tab:orange", markersize=6),
        ]
        ax.legend(handles=handles, loc="best", frameon=False)
        plot(fig)
    with c4:
        fig, ax = plt.subplots(constrained_layout=True)
        col = df["Voice mail plan"].map({"Yes": "tab:orange", "No": "tab:blue"}).values
        ax.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=col, alpha=0.5, s=12)
        ax.set_title("t-SNE by Voice mail plan")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label="No voicemail (blue)", markerfacecolor="tab:blue", markersize=6),
            plt.Line2D([0], [0], marker="o", color="w", label="Yes voicemail (orange)", markerfacecolor="tab:orange", markersize=6),
        ]
        ax.legend(handles=handles, loc="best", frameon=False)
        plot(fig)
