import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="Preeclampsia Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# 读取模型（加缓存，防重复加载崩溃）
# =========================
@st.cache_resource
def load_models():
    PE_MODEL = joblib.load("PE_model.pkl")
    EARLY_MODEL = joblib.load("Early_PE_model.pkl")
    PDO_MODEL = joblib.load("PDO_model.pkl")
    GA_MODEL = joblib.load("GA_model.pkl")
    return PE_MODEL, EARLY_MODEL, PDO_MODEL, GA_MODEL

PE_MODEL, EARLY_MODEL, PDO_MODEL, GA_MODEL = load_models()

FEATURES = getattr(PE_MODEL, "feature_names_in_", None)

if FEATURES is None:
    st.error("模型缺少 feature_names_in_，请重新保存模型")
    st.stop()


# =========================
# 工具函数
# =========================
def calc_hsi(ast, alt, bmi):
    try:
        if ast <= 0 or alt <= 0:
            return 0.0
        return 8 * (alt / ast) + bmi + 2
    except:
        return 0.0


def risk_level(risk):
    risk = float(np.nan_to_num(risk, nan=0.0))
    if risk < 0.05:
        return "🟢 低风险"
    elif risk < 0.15:
        return "🟡 中风险"
    else:
        return "🔴 高风险"


def week_to_ga(x):
    try:
        x = float(x)
        week = int(x)
        day = round((x - week) * 7)
        if day == 7:
            week += 1
            day = 0
        return f"{week}+{day}"
    except:
        return "N/A"


def safe_proba(model, X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    return float(model.predict(X)[0])


# =========================
# UI
# =========================
st.title("🩺 子痫前期风险预测系统")
st.warning("⚠️ 本系统仅供科研与教学用途，不用于临床诊断或治疗决策")

st.markdown("基于 FMF核心指标 + 血小板 + HSI + 超声指标 构建的多模型预测系统。")


# =========================
# 输入
# =========================
st.header("① 基本信息")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("年龄", 15, 60, 30)

with c2:
    BMI = st.number_input("BMI", 10.0, 60.0, 24.0)

with c3:
    parity = st.number_input("产次", 0, 10, 0)


st.header("② 高危病史")
c1, c2, c3 = st.columns(3)

with c1:
    pe_history = st.selectbox("子痫前期既往史", [0, 1])

with c2:
    chronic_htn = st.selectbox("慢性高血压", [0, 1])

with c3:
    diabetes = st.selectbox("糖尿病", [0, 1])


st.header("③ FMF核心指标")
c1, c2, c3 = st.columns(3)

with c1:
    mom_pappa = st.number_input("MoM PAPP-A", value=1.0)

with c2:
    mom_pi = st.number_input("MoM PI", value=1.0)

with c3:
    mom_map = st.number_input("MoM MAP", value=1.0)


st.header("④ 血小板")
Plt = st.number_input("Platelet", 50.0, 1000.0, 250.0)


st.header("⑤ HSI计算")
c1, c2 = st.columns(2)

with c1:
    AST = st.number_input("AST", 1.0, 200.0, 20.0)

with c2:
    ALT = st.number_input("ALT", 1.0, 200.0, 20.0)

HSI = calc_hsi(AST, ALT, BMI)
st.metric("HSI", f"{HSI:.2f}")


st.header("⑥ 超声指标")
EFW_percentile = st.number_input("EFW Percentile", 0.0, 100.0, 50.0)


# =========================
# 预测按钮（关键：避免反复rerun崩UI）
# =========================
if "run" not in st.session_state:
    st.session_state.run = False

if st.button("🚀 开始预测"):
    st.session_state.run = True


if st.session_state.run:

    X = pd.DataFrame([{
        "age": age,
        "BMI": BMI,
        "parity": parity,
        "子痫前期既往史": pe_history,
        "慢性高血压": chronic_htn,
        "糖尿病": diabetes,
        "MoM_PAPP-A": mom_pappa,
        "MoM_PI": mom_pi,
        "MoM_MAP": mom_map,
        "Plt": Plt,
        "HSI": HSI,
        "EFW_percentile": EFW_percentile
    }])

    X = X.reindex(columns=FEATURES)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # ⚠️ 不用0填（医学上更安全）
    X = X.fillna(X.mean())

    X = X.astype(np.float32)

    # =========================
    # 预测
    # =========================
    pe_risk = safe_proba(PE_MODEL, X)
    early_risk = safe_proba(EARLY_MODEL, X)
    pdo_risk = safe_proba(PDO_MODEL, X)
    ga_pred = float(GA_MODEL.predict(X)[0])

    # 防 NaN
    pe_risk = float(np.nan_to_num(pe_risk))
    early_risk = float(np.nan_to_num(early_risk))
    pdo_risk = float(np.nan_to_num(pdo_risk))
    ga_pred = float(np.nan_to_num(ga_pred))

    # =========================
    # 输出
    # =========================
    st.divider()
    st.header("📊 预测结果")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("PE风险", f"{pe_risk*100:.1f}%")
        st.write(risk_level(pe_risk))

    with c2:
        st.metric("Early-PE风险", f"{early_risk*100:.1f}%")
        st.write(risk_level(early_risk))

    with c3:
        st.metric("PDO风险", f"{pdo_risk*100:.1f}%")
        st.write(risk_level(pdo_risk))

    with c4:
        st.metric("预测分娩孕周", week_to_ga(ga_pred))
        st.write(f"{ga_pred:.2f} 周")

    st.divider()

    st.subheader("模型输入数据")
    st.dataframe(X.astype(str))  # ⚠️ 防前端崩溃关键修复

    if st.button("🔄 重置"):
        st.session_state.run = False

    st.success("本系统仅供科研与教学用途，不用于临床诊断或治疗决策")

    st.divider()
    st.subheader("模型输入数据")
    st.dataframe(X)

    st.success("本系统仅供科研与教学用途，不用于临床诊断或治疗决策")
