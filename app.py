import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# ==========================================
# 1. 页面配置与基础设定
# ==========================================
st.set_page_config(page_title="LGBM 临床预测工具", page_icon="🏥", layout="wide")

st.title("🏥 基于 LightGBM 的临床预测工具")
st.markdown("""
本工具基于 **LightGBM 梯度提升模型** 开发，通过已训练的机器学习流水线对患者的临床指标进行风险预测。
系统支持逻辑回归再校准，输出经校准的概率和风险分层结果。
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
display_names = feature_meta.get("display_names", {})
descriptions = feature_meta.get("descriptions", {})

# 提取 pipeline 内部的分类器（用于 SHAP）
classifier = lgbm_pipe.steps[-1][1]
preprocessor = lgbm_pipe[:-1] if len(lgbm_pipe.steps) > 1 else None


# ==========================================
# 3. 构建侧边栏：原始指标输入区
# ==========================================
st.sidebar.header("📝 输入患者临床指标")
st.sidebar.markdown("请在下方逐项输入患者的检验数值：")

input_values = {}
for feat in feature_names:
    disp = display_names.get(feat, feat)
    desc = descriptions.get(feat, None)
    input_values[feat] = st.sidebar.number_input(
        label=disp,
        value=0.0,
        format="%.4f",
        help=desc,
        key=f"input_{feat}",
    )

st.sidebar.markdown("---")
show_shap = st.sidebar.checkbox("显示 SHAP 特征解释", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("模型: LightGBM · 校准: 逻辑回归再校准 · 阈值: Youden's J")


# ==========================================
# 4. 再校准函数
# ==========================================
def apply_recalibration(recal, p_hat, eps=1e-12):
    """对原始概率进行逻辑回归再校准 (logit 变换)"""
    p = np.clip(np.asarray(p_hat), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    return recal.predict_proba(logit_p)[:, 1]


# ==========================================
# 5. 主干逻辑：预测与结果展示
# ==========================================
if st.button("🚀 开始风险评估", type="primary", use_container_width=True):

    # --- 步骤 A：组装输入 DataFrame ---
    input_df = pd.DataFrame({feat: [input_values[feat]] for feat in feature_names})

    # --- 步骤 B：模型预测与再校准 ---
    raw_prob = lgbm_pipe.predict_proba(input_df.values)[:, 1][0]
    final_prob = apply_recalibration(recal_model, np.array([raw_prob]))[0]

    # ==========================================
    # 6. 结果展示界面
    # ==========================================
    st.markdown("---")
    st.subheader("📊 评估结果与指标解析")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("💡 **核心预测风险概率**")
        risk_percent = final_prob * 100

        if final_prob >= 0.5:
            st.error(f"### 🛑 高风险：{risk_percent:.1f}%")
            st.write("**临床建议**：模型评估结果倾向于阳性（高风险），建议结合其他临床资料进一步评估。")
        else:
            st.success(f"### ✅ 低风险：{risk_percent:.1f}%")
            st.write("**临床建议**：模型评估结果倾向于阴性（低风险），但请结合患者具体情况进行最终判断。")

        st.progress(min(final_prob, 1.0))

    with col2:
        st.info("⚙️ **模型计算参数**")
        param_df = pd.DataFrame({
            "参数": ["原始模型概率", "校准后概率", "风险分层"],
            "数值": [
                f"{raw_prob:.4f} ({raw_prob * 100:.1f}%)",
                f"{final_prob:.4f} ({final_prob * 100:.1f}%)",
                "高风险" if final_prob >= 0.5 else "低风险",
            ]
        })
        st.table(param_df)

        # 展示输入特征汇总
        with st.expander("📋 查看输入特征汇总"):
            summary_df = pd.DataFrame({
                "特征名": feature_names,
                "显示名": [display_names.get(f, f) for f in feature_names],
                "输入值": [f"{input_values[f]:.4f}" for f in feature_names],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ==========================================
    # 7. SHAP 可解释性分析
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
            shap_df = pd.DataFrame({
                "特征": feature_names,
                "显示名": [display_names.get(f, f) for f in feature_names],
                "SHAP 值": sv_flat,
                "绝对值": np.abs(sv_flat),
                "输入值": [input_values[f] for f in feature_names],
            }).sort_values("绝对值", ascending=False)

            # 绘制 SHAP 条形图
            top_n = min(12, len(shap_df))
            top = shap_df.head(top_n).sort_values("SHAP 值")

            fig, ax = plt.subplots(figsize=(8, 0.45 * top_n + 1))
            colors = ["#d73027" if v > 0 else "#4575b4" for v in top["SHAP 值"]]
            ax.barh(
                range(len(top)),
                top["SHAP 值"].values,
                color=colors,
                edgecolor="none",
                height=0.65,
            )
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["显示名"].values, fontsize=10)
            ax.set_xlabel("SHAP 值（对预测结果的贡献）", fontsize=10)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("🔴 红色 = 推动预测概率升高 &nbsp;&nbsp; 🔵 蓝色 = 推动预测概率降低")

            with st.expander("📋 完整 SHAP 值表格"):
                st.dataframe(
                    shap_df[["特征", "显示名", "输入值", "SHAP 值"]]
                    .reset_index(drop=True),
                    use_container_width=True,
                )

        except ImportError:
            st.warning("SHAP 库未安装，请在 requirements.txt 中添加 `shap` 以启用特征解释。")
        except Exception as e:
            st.warning(f"无法生成 SHAP 解释: {e}")

    # 免责声明
    st.caption(
        "免责声明：本预测模型基于既往临床队列数据开发，仅作为辅助临床决策的参考工具，"
        "不可替代主治医师的独立临床判断。"
    )
