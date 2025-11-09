import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 页面基本设置
st.set_page_config(page_title="Wine Quality MLR", page_icon="🍷", layout="wide")

@st.cache_data
def load_data():
    """读取葡萄酒数据集"""
    df = pd.read_csv("Data/winequality-red.csv")
    cols = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density", "quality"]
    df = df.rename(columns=lambda c: c.strip())  # 清除列名空格
    return df[cols]

@st.cache_resource
def fit_model(df):
    """建立多元线性回归模型"""
    X = df[["alcohol", "volatile acidity", "sulphates", "citric acid", "density"]]
    X = sm.add_constant(X)
    y = df["quality"]
    model = sm.OLS(y, X).fit()
    return model

# 数据与模型加载
df = load_data()
model = fit_model(df)

# -------------------- 主体部分 --------------------
st.title("🍷 Wine Quality Prediction (Multiple Linear Regression)")
st.caption("Interactive demo: enter chemistry metrics to predict wine quality and view diagnostics.")

# 侧边栏：输入滑块
st.sidebar.header("Input Features")

def slider(label, s, e, v, step=0.01):
    return st.sidebar.slider(label, float(s), float(e), float(v), step=step)

q = df.quantile([0.05, 0.5, 0.95])
alcohol = slider("Alcohol (%)", q.loc[0.05,"alcohol"], q.loc[0.95,"alcohol"], q.loc[0.5,"alcohol"])
volatile = slider("Volatile acidity (g/dm³)", q.loc[0.05,"volatile acidity"], q.loc[0.95,"volatile acidity"], q.loc[0.5,"volatile acidity"])
sulphates = slider("Sulphates (g/dm³)", q.loc[0.05,"sulphates"], q.loc[0.95,"sulphates"], q.loc[0.5,"sulphates"])
citric = slider("Citric acid (g/dm³)", q.loc[0.05,"citric acid"], q.loc[0.95,"citric acid"], q.loc[0.5,"citric acid"])
density = slider("Density (g/cm³)", q.loc[0.05,"density"], q.loc[0.95,"density"], q.loc[0.5,"density"], step=0.0001)

# 预测
X_new = pd.DataFrame({
    "const":[1.0],
    "alcohol":[alcohol],
    "volatile acidity":[volatile],
    "sulphates":[sulphates],
    "citric acid":[citric],
    "density":[density],
})
pred = model.get_prediction(X_new)
res = pred.summary_frame(alpha=0.05)

# 输出
left, right = st.columns(2)
with left:
    st.subheader("Predicted Quality")
    st.metric("Predicted Score", f"{res['mean'].iloc[0]:.2f}")
    st.caption(f"95% CI: {res['obs_ci_lower'].iloc[0]:.2f} — {res['obs_ci_upper'].iloc[0]:.2f}")

with right:
    st.subheader("Model Summary")
    st.write(f"**R²:** {model.rsquared:.3f}")
    st.write("Significant variables (p < 0.05):")
    sig = model.pvalues[model.pvalues<0.05].index.tolist()
