# ablation_balanced500_rf_final.py
"""
基于 500 点 1:1 平衡验证集的完整消融实验脚本
最终分类器：Random Forest (RF)

数据目录:
  F:\新\yb3\最终版_4000样本\Validation_500_balanced
  - Training_Set.csv
  - Validation_Set.csv

输出目录:
  F:\新\yb3\最终版_4000样本\Validation_500_balanced\Paper_Results_Models_Balanced500
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ----------------- 配置 -----------------
class CFG:
    data_dir = Path(r"F:\新\yb3\最终版_4000样本\Validation_500_balanced")
    out_dir = data_dir / "Paper_Results_Models_Balanced500"

    N_BOOTSTRAP = 30
    TRAIN_RATIO = 0.9
    SEED = 42

    # 最终用于独立验证与混淆矩阵的模型
    FINAL_MODEL = "RF"

CFG.out_dir.mkdir(parents=True, exist_ok=True)

# 绘图风格
sns.set_theme(style="white", font="Times New Roman")
mpl.rcParams.update({
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

print("=" * 70)
print("基于 500 点平衡验证集的完整消融实验")
print("=" * 70)

# ----------------- 工具函数 -----------------
def eval_metrics(y_true, y_pred):
    """返回 OA、Kappa、Macro-F1"""
    return {
        "OA": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro")
    }

def build_model(model_name, seed):
    """统一模型构建函数"""
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            bootstrap=True,
            random_state=seed,
            n_jobs=-1
        )
    elif model_name == "GBDT":
        return GradientBoostingClassifier(
            n_estimators=200,
            subsample=0.8,
            random_state=seed
        )
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss"
        )
    elif model_name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1
        )
    else:
        raise ValueError(f"未知模型名称: {model_name}")

def fmt_pct(row, col):
    """百分比格式：OA / F1"""
    return f"{row[(col, 'mean')] * 100:.2f} ± {row[(col, 'std')] * 100:.2f}"

def fmt_dec(row, col):
    """小数格式：Kappa"""
    return f"{row[(col, 'mean')]:.3f} ± {row[(col, 'std')]:.3f}"

# ----------------- 1. 加载数据与特征定义 -----------------
print("1. 加载训练集与验证集...")

df_train = pd.read_csv(CFG.data_dir / "Training_Set.csv")
df_val = pd.read_csv(CFG.data_dir / "Validation_Set.csv")

print(f"训练集: {len(df_train)} 样本, 类别: {df_train['label'].value_counts().to_dict()}")
print(f"验证集: {len(df_val)} 样本, 类别: {df_val['label'].value_counts().to_dict()}")

all_cols = df_train.columns.tolist()

# 光学物候特征：9 个指数前缀
opt_keywords = ['evi', 'ndvi', 'gndvi', 'msavi', 'ndgi', 'ndmi', 'ndpi', 'ndsvi', 'ndti']
f_opt = [
    c for c in all_cols
    if any(k in c.lower() for k in opt_keywords)
    and c not in ['label', 'id', 'lon', 'lat', 'split', 'SVI_star']
]

# 环境特征
f_env = [c for c in ['elevation', 'slope', 'aspect', 'twi'] if c in all_cols]

if 'SVI_star' not in all_cols:
    raise ValueError("未在训练集中找到 SVI_star 字段，请检查数据。")

feat_configs = {
    'Optical Only': f_opt,
    'Optical + Env': f_opt + f_env,
    'Optical + SVI*': f_opt + ['SVI_star'],
    'Full (Ours)': f_opt + ['SVI_star'] + f_env
}

print(f"光学特征数: {len(f_opt)}, 环境特征数: {len(f_env)}, Full 特征数: {len(feat_configs['Full (Ours)'])}")

# ----------------- 2. Bootstrap 重复实验 -----------------
print("\n2. 运行 Bootstrap 重复实验...")
all_results = []

for i in range(CFG.N_BOOTSTRAP):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{CFG.N_BOOTSTRAP}")

    np.random.seed(CFG.SEED + i * 7)
    seed = CFG.SEED + i

    # 训练集 90% 子采样
    train_idx = np.random.choice(len(df_train), int(len(df_train) * CFG.TRAIN_RATIO), replace=False)
    df_tr = df_train.iloc[train_idx]

    # 验证集 Bootstrap
    val_idx = np.random.choice(len(df_val), len(df_val), replace=True)
    df_va = df_val.iloc[val_idx]

    # ---- A. 特征消融 (XGBoost) ----
    for name, feats in feat_configs.items():
        X_tr = df_tr[feats].fillna(-9999)
        y_tr = df_tr['label']
        X_va = df_va[feats].fillna(-9999)
        y_va = df_va['label']

        model_xgb = build_model("XGBoost", seed)
        model_xgb.fit(X_tr, y_tr)
        pred = model_xgb.predict(X_va)

        metrics = eval_metrics(y_va.values, pred)
        metrics.update({'Type': 'Feature (XGB)', 'Config': name})
        all_results.append(metrics)

    # ---- B. 特征消融 (Random Forest) ----
    for name, feats in feat_configs.items():
        X_tr = df_tr[feats].fillna(-9999)
        y_tr = df_tr['label']
        X_va = df_va[feats].fillna(-9999)
        y_va = df_va['label']

        model_rf = build_model("RF", seed)
        model_rf.fit(X_tr, y_tr)
        pred = model_rf.predict(X_va)

        metrics = eval_metrics(y_va.values, pred)
        metrics.update({'Type': 'Feature (RF)', 'Config': name})
        all_results.append(metrics)

    # ---- C. 模型对比 (Full 特征) ----
    full_feats = feat_configs['Full (Ours)']
    X_tr_full = df_tr[full_feats].fillna(-9999)
    y_tr_full = df_tr['label']
    X_va_full = df_va[full_feats].fillna(-9999)
    y_va_full = df_va['label']

    models = {
        'RF': build_model("RF", seed),
        'GBDT': build_model("GBDT", seed),
        'XGBoost': build_model("XGBoost", seed),
        'LightGBM': build_model("LightGBM", seed)
    }

    for m_name, m_cls in models.items():
        m_cls.fit(X_tr_full, y_tr_full)
        pred = m_cls.predict(X_va_full)

        metrics = eval_metrics(y_va_full.values, pred)
        metrics.update({'Type': 'Model', 'Config': m_name})
        all_results.append(metrics)

# ----------------- 3. 汇总结果与表格输出 -----------------
print("\n3. 汇总结果并生成表格...")

df_res = pd.DataFrame(all_results)
summary = df_res.groupby(['Type', 'Config']).agg(['mean', 'std'])

feat_order = ['Optical Only', 'Optical + Env', 'Optical + SVI*', 'Full (Ours)']
model_order = ['RF', 'GBDT', 'XGBoost', 'LightGBM']

# ---- 表1：XGB 特征消融 ----
rows_xgb = []
for c in feat_order:
    r = summary.loc[('Feature (XGB)', c)]
    rows_xgb.append({
        '特征组合': c,
        'OA (%)': fmt_pct(r, 'OA'),
        'Kappa': fmt_dec(r, 'Kappa'),
        'F1-macro (%)': fmt_pct(r, 'F1_macro')
    })
df_t1 = pd.DataFrame(rows_xgb)
print("\n=== 表1  基于 XGBoost 的特征消融结果 ===")
print(df_t1.to_string(index=False))
df_t1.to_csv(CFG.out_dir / "Table1_Feature_XGB.csv", index=False)

# ---- 表2：RF 特征消融 ----
rows_rf = []
for c in feat_order:
    r = summary.loc[('Feature (RF)', c)]
    rows_rf.append({
        '特征组合': c,
        'OA (%)': fmt_pct(r, 'OA'),
        'Kappa': fmt_dec(r, 'Kappa'),
        'F1-macro (%)': fmt_pct(r, 'F1_macro')
    })
df_t2 = pd.DataFrame(rows_rf)
print("\n=== 表2  基于 Random Forest 的特征消融结果 ===")
print(df_t2.to_string(index=False))
df_t2.to_csv(CFG.out_dir / "Table2_Feature_RF.csv", index=False)

# ---- 表3：模型对比 ----
rows_m = []
for m in model_order:
    r = summary.loc[('Model', m)]
    rows_m.append({
        '分类器': m,
        'OA (%)': fmt_pct(r, 'OA'),
        'Kappa': fmt_dec(r, 'Kappa'),
        'F1-macro (%)': fmt_pct(r, 'F1_macro')
    })
df_t3 = pd.DataFrame(rows_m)
print("\n=== 表3  不同分类器在 Full 特征下的精度对比 ===")
print(df_t3.to_string(index=False))
df_t3.to_csv(CFG.out_dir / "Table3_Model_Comparison.csv", index=False)

# ----------------- 4. 最终模型：Full + RF，500 点独立验证 -----------------
print(f"\n4. 训练 Full + {CFG.FINAL_MODEL}（使用全部训练样本），在500点验证集上评估...")

X_tr_full_all = df_train[full_feats].fillna(-9999)
y_tr_full_all = df_train['label']
X_va_full_all = df_val[full_feats].fillna(-9999)
y_va_full_all = df_val['label']

final_model = build_model(CFG.FINAL_MODEL, CFG.SEED)
final_model.fit(X_tr_full_all, y_tr_full_all)
pred_final = final_model.predict(X_va_full_all)

oa_final = accuracy_score(y_va_full_all, pred_final)
kappa_final = cohen_kappa_score(y_va_full_all, pred_final)
f1_final = f1_score(y_va_full_all, pred_final, average='macro')
cm_final = confusion_matrix(y_va_full_all, pred_final, labels=[0, 1])

print(f"Full+{CFG.FINAL_MODEL} 在500点验证集上的一次性结果：OA={oa_final*100:.2f}%，Kappa={kappa_final:.3f}，F1-macro={f1_final*100:.2f}%")
print("混淆矩阵：")
print(cm_final)

# 保存最终 RF 验证结果
class_names = ['Grassland', 'Shrubland']
cm_df = pd.DataFrame(
    cm_final,
    index=['True_Grassland', 'True_Shrubland'],
    columns=['Pred_Grassland', 'Pred_Shrubland']
)
cm_df.to_csv(CFG.out_dir / "Final_RF_Confusion_Matrix.csv", index=True)

final_metrics_df = pd.DataFrame([{
    'Model': CFG.FINAL_MODEL,
    'OA': oa_final,
    'Kappa': kappa_final,
    'F1_macro': f1_final,
    'PA_Grassland': cm_final[0, 0] / cm_final[0].sum(),
    'PA_Shrubland': cm_final[1, 1] / cm_final[1].sum(),
    'UA_Grassland': cm_final[0, 0] / cm_final[:, 0].sum(),
    'UA_Shrubland': cm_final[1, 1] / cm_final[:, 1].sum()
}])
final_metrics_df.to_csv(CFG.out_dir / "Final_RF_Validation_Metrics.csv", index=False)

# ----------------- 5. 生成 Fig1：特征消融 + 模型对比 -----------------
print("\n5. 生成 Fig1：特征消融 + 模型对比 ...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(feat_order))
width = 0.32

m_xgb = [summary.loc[('Feature (XGB)', c), ('OA', 'mean')] * 100 for c in feat_order]
s_xgb = [summary.loc[('Feature (XGB)', c), ('OA', 'std')] * 100 for c in feat_order]
m_rf = [summary.loc[('Feature (RF)', c), ('OA', 'mean')] * 100 for c in feat_order]
s_rf = [summary.loc[('Feature (RF)', c), ('OA', 'std')] * 100 for c in feat_order]

axes[0].bar(
    x - width / 2, m_xgb, width, yerr=s_xgb,
    label='XGBoost',
    color='#1f78b4', edgecolor='black', linewidth=0.6,
    capsize=3, error_kw={'linewidth': 0.8}
)
axes[0].bar(
    x + width / 2, m_rf, width, yerr=s_rf,
    label='Random Forest',
    color='#33a02c', edgecolor='black', linewidth=0.6,
    capsize=3, error_kw={'linewidth': 0.8}
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(feat_order, rotation=15, ha='right')
axes[0].set_ylabel('Overall accuracy (%)')
axes[0].set_ylim(84, 96)
axes[0].set_title('(a) Feature ablation')
axes[0].legend(frameon=False, loc='lower right')
axes[0].grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=axes[0])

m_model = [summary.loc[('Model', m), ('OA', 'mean')] * 100 for m in model_order]
s_model = [summary.loc[('Model', m), ('OA', 'std')] * 100 for m in model_order]
palette_models = ['#33a02c', '#ff7f00', '#1f78b4', '#6a3d9a']

axes[1].bar(
    model_order, m_model, yerr=s_model,
    color=palette_models, edgecolor='black', linewidth=0.6,
    capsize=3, error_kw={'linewidth': 0.8}
)
axes[1].set_ylabel('Overall accuracy (%)')
axes[1].set_ylim(88, 96)
axes[1].set_title('(b) Classifier comparison (Full features)')
axes[1].grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(CFG.out_dir / "Fig1_Ablation_and_Models_RFfinal.png", dpi=600, bbox_inches='tight')
plt.savefig(CFG.out_dir / "Fig1_Ablation_and_Models_RFfinal.pdf", dpi=600, bbox_inches='tight')

# ----------------- 6. 生成 Fig2：RF 混淆矩阵 -----------------
print("6. 生成 Fig2：混淆矩阵 ...")

cm_norm = cm_final.astype(float) / cm_final.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(3.8, 3.6))
sns.heatmap(
    cm_norm, annot=True, fmt='.02f',
    cmap='Blues', cbar_kws={'shrink': 0.7},
    vmin=0, vmax=1, square=True,
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={'size': 11, 'color': 'white'},
    ax=ax
)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title(f'Confusion matrix (Full + {CFG.FINAL_MODEL}, 500-val)')
plt.tight_layout()
plt.savefig(CFG.out_dir / "Fig2_Confusion_Matrix_RFfinal.png", dpi=600, bbox_inches='tight')
plt.savefig(CFG.out_dir / "Fig2_Confusion_Matrix_RFfinal.pdf", dpi=600, bbox_inches='tight')

# ----------------- 7. 标准差检查 -----------------
print("\n7. 标准差检查")
std_check = df_res.groupby(['Type', 'Config'])['OA'].std() * 100
print(f"OA 标准差范围: {std_check.min():.2f}% ~ {std_check.max():.2f}%")
print(f"平均标准差:     {std_check.mean():.2f}%")

print(f"\n✅ 全部完成，结果与图件已保存至: {CFG.out_dir}")
for f in sorted(CFG.out_dir.glob('*')):
    print("  -", f.name)