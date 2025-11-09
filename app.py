import streamlit as st
from pathlib import Path

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction (Jia & Ziqi) ",
    page_icon="🍷",
    layout="wide",
)

# -----------------------------
# Global Styles (CSS)
# -----------------------------
st.markdown(
    """
<style>
/* ---- Hero (title area) ---- */
.hero { text-align:center; margin-top:-6px; }
.hero h1 { 
  font-size: 38px; 
  margin-bottom: 6px; 
  font-weight: 800; 
}
.hero p { 
  font-size: 16px; 
  color:#444; 
  margin: 0 0 14px 0;
}
.hero a.gh-btn{
  display:inline-block; 
  background:#7b1113; 
  color:#fff !important; 
  padding:10px 16px; 
  border-radius:8px; 
  text-decoration:none; 
  font-weight:600;
}
.hero a.gh-btn:hover{ background:#9c1b1d; }

/* ---- Two column "cards" ---- */
/* 选中当前页面的第一组水平区块（你的主两列） */
div[data-testid="stHorizontalBlock"] > div:first-child {
  background:#FFF8F0;              /* 左侧：暖杏色 */
  padding:18px 18px 8px 18px;
  border-radius:14px;
  box-shadow:0 2px 10px rgba(0,0,0,0.04);
  border:1px solid #F1E6D6;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
  background:#F3F9FF;              /* 右侧：浅蓝色 */
  padding:18px 18px 8px 18px;
  border-radius:14px;
  box-shadow:0 2px 10px rgba(0,0,0,0.04);
  border:1px solid #DBEAFE;
}

/* 小的竖向间距工具类 */
.block-space { margin: 14px 0 10px 0; }
.section-space { margin: 30px 0 12px 0; }

.caption-tight { color:#666; font-size:13px; margin-top:-8px; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header (centered) with GitHub button
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>🍷 Wine Quality Prediction (Multiple Linear Regression)</h1>
  <p>Interactive demo: enter chemistry metrics to predict wine quality and view diagnostics.</p>
  <a class="gh-btn" href="https://github.com/giahuth-88/Wine-Quality-MLR" target="_blank">
    🔗 View on GitHub
  </a>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="block-space"></div>', unsafe_allow_html=True)

# -----------------------------
# Two Columns: Inputs & Prediction
# -----------------------------
col_left, col_right = st.columns([1.25, 1], gap="large")

with col_left:
    st.subheader("Input Features")
    st.caption("Move the sliders to see how chemistry features influence predicted quality.")

    # Ranges based on your EDA & dataset
    alcohol = st.slider("Alcohol (%)", 9.20, 12.50, 10.00, step=0.01)
    volatile_acidity = st.slider("Volatile acidity (g/dm³)", 0.27, 0.84, 0.50, step=0.01)
    sulphates = st.slider("Sulphates (g/dm³)", 0.47, 0.93, 0.65, step=0.01)
    citric_acid = st.slider("Citric acid (g/dm³)", 0.00, 0.60, 0.25, step=0.01)
    density = st.slider("Density (g/cm³)", 0.985, 1.005, 0.995, step=0.001)

with col_right:
    st.subheader("Predicted Quality")

    # -----------------------------
    # MLR prediction using your OLS coefficients (from your summary screenshot)
    # quality = const + b1*alcohol + b2*volatile_acidity + b3*sulphates + b4*citric_acid + b5*density
    # -----------------------------
    const = -12.5040
    b_alcohol = 0.3229
    b_volatile = -1.3015
    b_sulphates = 0.6797
    b_citric = -0.1550
    b_density = 15.1055

    y_pred = (
        const
        + b_alcohol * alcohol
        + b_volatile * volatile_acidity
        + b_sulphates * sulphates
        + b_citric * citric_acid
        + b_density * density
    )

    # optional: reasonable clamp
    y_display = max(3.0, min(8.5, y_pred))

    st.metric("Predicted Score", f"{y_display:.2f}")
    st.caption("95% CI: 4.0 – 6.6")

    st.subheader("Model Summary")
    st.markdown(
        """
**R²:** 0.337  
**Significant variables (p < 0.05):** alcohol, volatile acidity, sulphates  
**Other features shown:** citric acid, density
"""
    )

# -----------------------------
# Project Background (optional but nice for portfolio)
# -----------------------------
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("Project Background")
st.markdown(
    """
This mini-app uses the **UCI Wine Quality** dataset (red wine) to explore how
a subset of chemical properties relates to sensory quality. We fit a simple
**Multiple Linear Regression (MLR)** using `statsmodels` (OLS). The model's purpose
is illustrative: it provides a transparent baseline while the app lets you
interactively see how each feature shifts the predicted score.

**Key ideas**
- Alcohol tends to increase predicted quality (positive coefficient)  
- Volatile acidity tends to decrease it (negative coefficient)  
- Sulphates show a positive association  
- Citric acid & density are included for exploration even if not always significant in OLS  
"""
)

# -----------------------------
# Visualization Gallery
# -----------------------------
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("Visualization Gallery")
st.caption("Diagnostic & exploratory figures generated during the analysis.")

ASSETS_DIR = Path("Assets")
# 显示顺序与标题
items = [
    ("Correlation_Heatmap.png", "Correlation Heatmap — 变量间相关性"),
    ("Pairwise_Relationships.png", "Pairwise Relationships — 成对特征关系"),
    ("Correlation_with_Wine_Quality.png", "Correlation with Wine Quality — 与酒质线性关系"),
    ("Distribution_of_Wine_Features.png", "Distribution of Wine Features — 特征分布"),
    ("Outliers_Detection.png", "Outliers Detection — 箱线图/异常点"),
    ("Actual_vs_Predicted_Wine_Quality.png", "Distribution of Actual vs Predicted Wine Quality"),
]

# 自动过滤不存在的文件
existing = [(ASSETS_DIR / f, cap) for f, cap in items if (ASSETS_DIR / f).exists()]

if not existing:
    st.info("No images found under `Assets/` yet. Add your PNGs to display the gallery.")
else:
    # 三列展示
    n_cols = 3
    rows = [existing[i : i + n_cols] for i in range(0, len(existing), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for (p, cap), c in zip(row, cols):
            c.image(str(p), caption=cap, use_column_width=True)
