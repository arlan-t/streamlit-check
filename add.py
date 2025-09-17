import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Telecom Churn Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    # Normalize churn column to boolean
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map(
            {"yes": True, "true": True, "1": True, "no": False, "false": False, "0": False}
        )
    return df

df = load_data()

st.title("📊 Telecom Churn — Data Storyline")

# --- KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    st.metric("Numeric features", num_cols)
with col4:
    st.metric("Churn rate", f"{df['Churn'].mean()*100:.1f}%")

# --- Distribution of churn
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# --- Histograms & Density for selected features
st.subheader("Histograms & Density Plots")
features = ["Total day minutes", "Total intl calls"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df[features].hist(ax=axes)
st.pyplot(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df[features].plot(kind="density", subplots=True, layout=(1, 2), sharex=False, ax=axes)
st.pyplot(fig)

# --- Boxplot & Violin for "Total intl calls"
st.subheader("Boxplot & Violin")
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
sns.boxplot(y=df["Total intl calls"], ax=axes[0])
sns.violinplot(y=df["Total intl calls"], ax=axes[1])
st.pyplot(fig)

# --- Correlation heatmap
st.subheader("Correlation Heatmap (Numeric Features)")
numerical = list(
    set(df.select_dtypes(include=[np.number]).columns)
    - {"Total day charge","Total eve charge","Total night charge","Total intl charge"}
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numerical].corr(), annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Jointplots
st.subheader("Jointplots")
st.write("Relationship between **Total day minutes** and **Total night minutes**")
g = sns.jointplot(x="Total day minutes", y="Total night minutes", data=df, kind="scatter")
st.pyplot(g.fig)

g = sns.jointplot(x="Total day minutes", y="Total night minutes", data=df, kind="kde", color="g")
st.pyplot(g.fig)

# --- Pairplot
st.subheader("Pairplot of Numerical Features")
st.write("This may take a while ⏳")
sns.set(style="ticks")
g = sns.pairplot(df[numerical])
st.pyplot(g.fig)

# --- LM plot with churn hue
st.subheader("Scatter by Churn")
g = sns.lmplot(
    x="Total day minutes", y="Total night minutes", data=df,
    hue="Churn", fit_reg=False, aspect=1.2
)
st.pyplot(g.fig)

# --- Boxplots by Churn
st.subheader("Boxplots by Churn")
numerical_with_calls = numerical + ["Customer service calls"]
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
for idx, feat in enumerate(numerical_with_calls[:12]):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x="Churn", y=feat, data=df, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(feat)
fig.tight_layout()
st.pyplot(fig)

# --- Customer service calls distribution
st.subheader("Customer Service Calls")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.countplot(x="Customer service calls", data=df, ax=axes[0])
sns.countplot(x="Customer service calls", hue="Churn", data=df, ax=axes[1])
st.pyplot(fig)

# --- International / Voice mail plans vs churn
st.subheader("Plans vs Churn")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.countplot(x="International plan", hue="Churn", data=df, ax=axes[0])
sns.countplot(x="Voice mail plan", hue="Churn", data=df, ax=axes[1])
st.pyplot(fig)

# --- State churn rates
st.subheader("Churn Rate by State (Top States)")
state_churn = df.groupby("State")["Churn"].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 4))
state_churn.plot(kind="bar", ax=ax)
ax.set_ylabel("Churn rate")
st.pyplot(fig)

# --- t-SNE visualization
st.subheader("t-SNE Representation of Customers")

st.markdown("""
**t-distributed Stochastic Neighbor Embedding (t-SNE)** is a dimensionality reduction method.  
It projects high-dimensional data onto a 2D plane, while trying to preserve the local neighborhood structure:
- Points that were close in the original space remain close on the 2D plane.
- Points that were far apart remain far apart.

Essentially, it’s a way to **see clusters and patterns** in complex data.

⚠️ **Important caveats**:
- **Computationally heavy** → not practical for very large datasets (scikit-learn implementation can be slow).
- **Randomness** → different seeds can produce different pictures.
- **Interpretation** → you shouldn’t make final conclusions from t-SNE plots, but they can inspire hypotheses for further analysis.
""")

X = df.drop(["Churn","State"], axis=1)
X["International plan"] = X["International plan"].map({"Yes":1,"No":0})
X["Voice mail plan"] = X["Voice mail plan"].map({"Yes":1,"No":0})
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(random_state=17, n_iter=500, perplexity=30)
tsne_repr = tsne.fit_transform(X_scaled)

# Plot 1: Base t-SNE
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(tsne_repr[:,0], tsne_repr[:,1], alpha=0.5)
ax.set_title("t-SNE representation (all customers)")
st.pyplot(fig)

# Plot 2: Colored by churn
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(tsne_repr[:,0], tsne_repr[:,1],
           c=df["Churn"].map({False:"blue", True:"orange"}), alpha=0.5)
ax.set_title("t-SNE colored by Churn (Blue = Loyal, Orange = Churned)")
st.pyplot(fig)

# Plot 3: Colored by International/Voicemail plans
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, name in enumerate(["International plan","Voice mail plan"]):
    axes[i].scatter(
        tsne_repr[:,0], tsne_repr[:,1],
        c=df[name].map({"Yes":"orange","No":"blue"}), alpha=0.5
    )
    axes[i].set_title(name)
st.pyplot(fig)

st.markdown("""
💡 **Insight from this dataset**:
- Churned customers (orange) tend to cluster together in certain regions of the t-SNE plot.
- Many dissatisfied churners appear to overlap with **International Plan = Yes** but **no Voice mail plan** segment.
""")
