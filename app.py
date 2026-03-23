import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# ==========================================
# 1. 页面配置与基础设定
# ==========================================
st.set_page_config(page_title="LGBM 临床预测工具", page_icon="🏥", layout="wide")

st.title("🏥 基于 LightGBM 的临床预测工具")
st.markdown("""
本工具基于 **LightGBM 梯度提升机器学习模型** 开发，用于临床风险预测。
系统会自动根据您输入的原始血常规和生化指标，计算所需的复合炎症指数并评估风险。
""")


# ==========================================
# 2. 加载模型、再校准器与特征列表
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load("deploy_models/FinalModel_LGBM.joblib")
    recalibrator = joblib.load("deploy_models/Recalibrator_LGBM.joblib")
    with open("deploy_models/feature_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, recalibrator, meta


try:
    lgbm_pipe, recal_model, feature_meta = load_models()
except Exception as e:
    st.error(
        f"⚠️ 无法加载模型文件，请确保 `deploy_models/` 目录下存在 "
        f"FinalModel_LGBM.joblib、Recalibrator_LGBM.joblib 和 feature_meta.json。\n\n"
        f"错误详情: {e}"
    )
    st.stop()

feature_names = feature_meta["feature_names"]

# 提取 pipeline 内部的分类器（用于 SHAP）
classifier = lgbm_pipe.steps[-1][1]
preprocessor = lgbm_pipe[:-1] if len(lgbm_pipe.steps) > 1 else None


# ==========================================
# 3. 安全除法函数
# ==========================================
def safe_div(num, den, eps=1e-6):
    return num / (den if abs(den) > eps else eps)


# ==========================================
# 4. 再校准函数
# ==========================================
def apply_recalibration(recal, p_hat, eps=1e-12):
    """对原始概率进行逻辑回归再校准 (logit 变换)"""
    p = np.clip(np.asarray(p_hat), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    return recal.predict_proba(logit_p)[:, 1]


# ==========================================
# 5. 构建侧边栏：原始指标输入区
# ==========================================
st.sidebar.header("📝 输入患者原始检验指标")

st.sidebar.subheader("血常规指标")
wbc = st.sidebar.number_input("WBC (白细胞计数, 10⁹/L)", value=6.00, min_value=0.10, format="%.2f")
ne = st.sidebar.number_input("NEU (中性粒细胞, 10⁹/L)", value=4.00, min_value=0.10, format="%.2f")
lym = st.sidebar.number_input("LYM (淋巴细胞, 10⁹/L)", value=2.00, min_value=0.10, format="%.2f")
mon = st.sidebar.number_input("MON (单核细胞, 10⁹/L)", value=0.50, min_value=0.01, format="%.2f")
mpv = st.sidebar.number_input("MPV (平均血小板体积, fL)", value=10.00, min_value=1.00, format="%.2f")

st.sidebar.subheader("生化与炎症指标")
crp = st.sidebar.number_input("CRP (C反应蛋白, mg/L)", value=5.00, min_value=0.00, format="%.2f")
esr = st.sidebar.number_input("ESR (红细胞沉降率, mm/h)", value=15.00, min_value=0.00, format="%.2f")
alb = st.sidebar.number_input("ALB (白蛋白, g/L)", value=40.00, min_value=1.00, format="%.2f")
hdl_c = st.sidebar.number_input("HDL-C (高密度脂蛋白胆固醇, mmol/L)", value=1.20, min_value=0.01, format="%.2f")

st.sidebar.markdown("---")
show_shap = st.sidebar.checkbox("显示 SHAP 特征解释", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("模型: LightGBM · 校准: 逻辑回归再校准 · 阈值: Youden's J")


# ==========================================
# 6. 主干逻辑：计算复合指标与预测
# ==========================================
if st.button("🚀 开始风险评估", type="primary", use_container_width=True):

    # --- 步骤 A：自动计算复合指标 ---
    dnlr = safe_div(ne, wbc - ne)          # dNLR = NE / (WBC - NE)
    cally = crp + lym * 0                  # CALLY = CRP + LYM × 0
    mhr = safe_div(mon, hdl_c)             # MHR = MON / HDL-C
    tp = alb + 0                           # TP = ALB + 0
    # ESR 和 MPV 直接使用原始输入值

    # 汇总模型需要的 6 个特征
    derived_features = {
        "dNLR": dnlr,
        "CALLY": cally,
        "ESR": esr,
        "MPV": mpv,
        "TP": tp,
        "MHR": mhr,
    }

    # 按模型训练时的特征顺序组装 DataFrame
    try:
        input_data = {feat: [derived_features[feat]] for feat in feature_names}
        input_df = pd.DataFrame(input_data)
    except KeyError as e:
        st.error(f"特征匹配错误，未找到必须的特征 {e}。请检查 feature_meta.json 的内容。")
        st.stop()

    # --- 步骤 B：模型预测与再校准 ---
    raw_prob = lgbm_pipe.predict_proba(input_df.values)[:, 1][0]
    final_prob = apply_recalibration(recal_model, np.array([raw_prob]))[0]

    # ==========================================
    # 7. 结果展示界面
    # ==========================================
    st.markdown("---")
    st.subheader("📊 评估结果与指标解析")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("💡 **核心感染风险概率**")
        risk_percent = final_prob * 100

        if final_prob >= 0.5:
            st.error(f"### 🛑 高风险：{risk_percent:.1f}%")
            st.write("**临床建议**：模型评估结果倾向于阳性（高风险），建议结合其他临床资料进一步评估。")
        else:
            st.success(f"### ✅ 低风险：{risk_percent:.1f}%")
            st.write("**临床建议**：模型评估结果倾向于阴性（低风险），但请结合患者具体情况进行最终判断。")

        st.progress(min(final_prob, 1.0))

    with col2:
        st.info("📋 **输入的原始检验指标**")
        display_df = pd.DataFrame({
            "检验项目": [
                "WBC (白细胞计数)", "NEU (中性粒细胞)", "LYM (淋巴细胞)",
                "MON (单核细胞)", "MPV (平均血小板体积)", "CRP (C反应蛋白)",
                "ESR (红细胞沉降率)", "ALB (白蛋白)", "HDL-C (高密度脂蛋白胆固醇)"
            ],
            "数值": [
                f"{wbc:.2f}", f"{ne:.2f}", f"{lym:.2f}",
                f"{mon:.2f}", f"{mpv:.2f}", f"{crp:.2f}",
                f"{esr:.2f}", f"{alb:.2f}", f"{hdl_c:.2f}"
            ],
            "单位": [
                "10⁹/L", "10⁹/L", "10⁹/L",
                "10⁹/L", "fL", "mg/L",
                "mm/h", "g/L", "mmol/L"
            ]
        })
        st.table(display_df)

    # ==========================================
    # 8. SHAP 可解释性分析
    # ==========================================
    if show_shap:
        st.markdown("---")
        st.subheader("🔍 SHAP 特征贡献分析（本次预测）")

        try:
            import shap
            import matplotlib.pyplot as plt

            # 获取预处理后的数据
            if preprocessor is not None:
                X_trans = preprocessor.transform(input_df.values)
            else:
                X_trans = input_df.values

            X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans_df)

            # 处理二分类 SHAP 输出
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            sv_flat = sv.flatten()

            # 特征中文显示名映射
            feature_display = {
                "dNLR": "dNLR 指数",
                "CALLY": "CALLY 指数",
                "ESR": "ESR (红细胞沉降率)",
                "MPV": "MPV (平均血小板体积)",
                "TP": "TP (总蛋白)",
                "MHR": "MHR 比值",
            }

            shap_df = pd.DataFrame({
                "特征": feature_names,
                "显示名": [feature_display.get(f, f) for f in feature_names],
                "SHAP 值": sv_flat,
                "绝对值": np.abs(sv_flat),
                "特征值": [derived_features[f] for f in feature_names],
            }).sort_values("绝对值", ascending=False)

            # 绘制 SHAP 条形图
            top_n = min(6, len(shap_df))
            top = shap_df.head(top_n).sort_values("SHAP 值")

            fig, ax = plt.subplots(figsize=(8, 0.5 * top_n + 1))
            colors = ["#d73027" if v > 0 else "#4575b4" for v in top["SHAP 值"]]
            ax.barh(
                range(len(top)),
                top["SHAP 值"].values,
                color=colors,
                edgecolor="none",
                height=0.65,
            )
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["显示名"].values, fontsize=11)
            ax.set_xlabel("SHAP 值（对预测结果的贡献）", fontsize=10)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("🔴 红色 = 推动预测概率升高 &emsp; 🔵 蓝色 = 推动预测概率降低")

        except ImportError:
            st.warning("SHAP 库未安装，请在 requirements.txt 中添加 `shap` 以启用特征解释。")
        except Exception as e:
            st.warning(f"无法生成 SHAP 解释: {e}")

    # 免责声明
    st.caption(
        "免责声明：本预测模型基于既往临床队列数据开发，仅作为辅助临床决策的参考工具，"
        "不可替代主治医师的独立临床判断。"
    )
