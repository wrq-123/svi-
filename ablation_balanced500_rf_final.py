import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# ----------------- 配置 -----------------
class CFG:
    data_dir = Path(r"F:\新\yb3\最终版_4000样本\Validation_500_balanced")
    out_dir  = data_dir / "Paper_Results_Models_Balanced500"

    N_BOOTSTRAP = 30
    TRAIN_RATIO = 0.9
    SEED        = 42

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

print("="*70)
print("基于 500 点平衡验证集的完整消融实验（修订版）")
print("="*70)

# ----------------- 0. 版本信息（可复现性） -----------------
try:
    import sys, sklearn
    print("Python:", sys.version.split()[0])
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("xgboost:", xgb.__version__)
    print("lightgbm:", lgb.__version__)
except Exception as e:
    print("版本信息输出失败：", e)

# ----------------- 1. 加载数据与特征定义 -----------------
print("\n1. 加载训练集与验证集...")

df_train = pd.read_csv(CFG.data_dir / "Training_Set.csv")
df_val   = pd.read_csv(CFG.data_dir / "Validation_Set.csv")

print(f"训练集: {len(df_train)} 样本, 类别: {df_train['label'].value_counts().to_dict()}")
print(f"验证集: {len(df_val)} 样本, 类别: {df_val['label'].value_counts().to_dict()}")

# 预分层（用于分层 bootstrap，保证每次仍 1:1）
df_val_g = df_val[df_val['label'] == 0].reset_index(drop=True)
df_val_s = df_val[df_val['label'] == 1].reset_index(drop=True)
if len(df_val_g) == 0 or len(df_val_s) == 0:
    raise ValueError("验证集中某一类为空，无法进行分层 bootstrap。请检查 Validation_Set.csv 的 label。")

all_cols = df_train.columns.tolist()

# 光学物候特征：9 个指数前缀
opt_keywords = ['evi','ndvi','gndvi','msavi','ndgi','ndmi','ndpi','ndsvi','ndti']
f_opt = [
    c for c in all_cols
    if any(k in c.lower() for k in opt_keywords)
    and c not in ['label','id','lon','lat','split','SVI_star']
]

# 环境特征
f_env = [c for c in ['elevation','slope','aspect','twi'] if c in all_cols]

if 'SVI_star' not in all_cols:
    raise ValueError("未在训练集中找到 SVI_star 字段，请检查数据。")

feat_configs = {
    'Optical Only': f_opt,
    'Optical + Env': f_opt + f_env,
    'Optical + SVI*': f_opt + ['SVI_star'],
    'Full (Ours)': f_opt + ['SVI_star'] + f_env
}

print(f"光学特征数: {len(f_opt)}, 环境特征数: {len(f_env)}, Full 特征数: {len(feat_configs['Full (Ours)'])}")

# ----------------- 2. 评价函数 -----------------
def eval_metrics(y_true, y_pred):
    return {
        'OA': accuracy_score(y_true, y_pred),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'F1_macro': f1_score(y_true, y_pred, average='macro')
    }

# ----------------- 3. Bootstrap 重复实验 -----------------
print("\n2. 运行 Bootstrap 重复实验...")
all_results = []

# 统一 XGBoost 关键参数（确保 ablation 与 model comparison 一致）
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    n_jobs=-1,
    verbosity=0
)

for i in range(CFG.N_BOOTSTRAP):
    rep = i + 1
    if rep % 10 == 0:
        print(f"  Progress: {rep}/{CFG.N_BOOTSTRAP}")

    # rng：用于抽样；seed：用于模型 random_state
    rng  = np.random.default_rng(CFG.SEED + i*7)
    seed = CFG.SEED + i

    # 训练集 90% 子采样
    train_idx = rng.choice(len(df_train), int(len(df_train)*CFG.TRAIN_RATIO), replace=False)
    df_tr = df_train.iloc[train_idx].reset_index(drop=True)

    # 验证集分层 Bootstrap（保持 1:1）
    idx_g = rng.choice(len(df_val_g), len(df_val_g), replace=True)
    idx_s = rng.choice(len(df_val_s), len(df_val_s), replace=True)
    df_va = pd.concat([df_val_g.iloc[idx_g], df_val_s.iloc[idx_s]], axis=0, ignore_index=True)
    df_va = df_va.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    # ---- A. 特征消融 (XGBoost) ----
    for name, feats in feat_configs.items():
        X_tr = df_tr[feats].fillna(-9999)
        y_tr = df_tr['label']
        X_va = df_va[feats].fillna(-9999)
        y_va = df_va['label']

        model_xgb = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS)
        model_xgb.fit(X_tr, y_tr)
        pred = model_xgb.predict(X_va)

        metrics = eval_metrics(y_va.values, pred)
        metrics.update({'Type': 'Feature (XGB)', 'Config': name, 'Rep': rep, 'Seed': seed})
        all_results.append(metrics)

    # ---- B. 特征消融 (RandomForest) ----
    for name, feats in feat_configs.items():
        X_tr = df_tr[feats].fillna(-9999)
        y_tr = df_tr['label']
        X_va = df_va[feats].fillna(-9999)
        y_va = df_va['label']

        model_rf = RandomForestClassifier(
            n_estimators=300, max_features='sqrt',
            bootstrap=True, random_state=seed, n_jobs=-1
        )
        model_rf.fit(X_tr, y_tr)
        pred = model_rf.predict(X_va)

        metrics = eval_metrics(y_va.values, pred)
        metrics.update({'Type': 'Feature (RF)', 'Config': name, 'Rep': rep, 'Seed': seed})
        all_results.append(metrics)

    # ---- C. 模型对比 (Full 特征) ----
    full_feats = feat_configs['Full (Ours)']
    X_tr_full = df_tr[full_feats].fillna(-9999)
    y_tr_full = df_tr['label']
    X_va_full = df_va[full_feats].fillna(-9999)
    y_va_full = df_va['label']

    models = {
        'RF': RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        'GBDT': GradientBoostingClassifier(n_estimators=200, subsample=0.8, random_state=seed),
        # 统一 XGBoost 关键参数（与 ablation 一致）
        'XGBoost': xgb.XGBClassifier(random_state=seed, **XGB_PARAMS),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1
        )
    }

    for m_name, m_cls in models.items():
        m_cls.fit(X_tr_full, y_tr_full)
        pred = m_cls.predict(X_va_full)

        metrics = eval_metrics(y_va_full.values, pred)
        metrics.update({'Type': 'Model', 'Config': m_name, 'Rep': rep, 'Seed': seed})
        all_results.append(metrics)

# ----------------- 4. 汇总结果与表格输出 -----------------
print("\n3. 汇总结果并生成表格...")

df_res = pd.DataFrame(all_results)

# 导出全量 bootstrap 结果（便于复现与审稿核查）
df_res.to_csv(CFG.out_dir / "All_bootstrap_results.csv", index=False)

summary = df_res.groupby(['Type','Config']).agg(['mean','std'])

def fmt(row, col):
    return f"{row[(col,'mean')]*100:.2f} ± {row[(col,'std')]*100:.2f}"

feat_order  = ['Optical Only','Optical + Env','Optical + SVI*','Full (Ours)']
model_order = ['RF','GBDT','XGBoost','LightGBM']

# ---- 表1：XGB 特征消融 ----
rows_xgb = []
for c in feat_order:
    r = summary.loc[('Feature (XGB)', c)]
    rows_xgb.append({
        '特征组合': c,
        'OA (%)': fmt(r,'OA'),
        'Kappa': fmt(r,'Kappa'),
        'F1-macro (%)': fmt(r,'F1_macro')
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
        'OA (%)': fmt(r,'OA'),
        'Kappa': fmt(r,'Kappa'),
        'F1-macro (%)': fmt(r,'F1_macro')
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
        'OA (%)': fmt(r,'OA'),
        'Kappa': fmt(r,'Kappa'),
        'F1-macro (%)': fmt(r,'F1_macro')
    })
df_t3 = pd.DataFrame(rows_m)
print("\n=== 表3  不同分类器在 Full 特征下的精度对比 ===")
print(df_t3.to_string(index=False))
df_t3.to_csv(CFG.out_dir / "Table3_Model_Comparison.csv", index=False)

# ----------------- 4b. Paired t-test（用于论文“paired t-tests”复现） -----------------
print("\n3b. 计算 paired t-tests（基于 bootstrap 重复的配对样本）...")

def paired_ttest(df, type_name, cfg_a, cfg_b, metric='OA'):
    da = df[(df['Type']==type_name) & (df['Config']==cfg_a)].sort_values('Rep')[metric].values
    db = df[(df['Type']==type_name) & (df['Config']==cfg_b)].sort_values('Rep')[metric].values
    if len(da) != len(db):
        raise ValueError(f"配对长度不一致：{type_name}, {cfg_a} vs {cfg_b}")
    t, p = ttest_rel(db, da)  # b - a
    return t, p, float(np.mean(db-da)), float(np.std(db-da, ddof=1))

tests = []
# 常用配对：Optical Only vs Optical + SVI*（你文中常用这个显著性）
for type_name in ['Feature (XGB)', 'Feature (RF)']:
    for metric in ['OA', 'Kappa', 'F1_macro']:
        t, p, dmean, dstd = paired_ttest(df_res, type_name, 'Optical Only', 'Optical + SVI*', metric=metric)
        tests.append({
            'Type': type_name,
            'Metric': metric,
            'A': 'Optical Only',
            'B': 'Optical + SVI*',
            'MeanDiff(B-A)': dmean,
            'StdDiff(B-A)': dstd,
            't': t,
            'p': p
        })

df_tests = pd.DataFrame(tests)
df_tests.to_csv(CFG.out_dir / "Paired_ttests_OpticalOnly_vs_OpticalPlusSVI.csv", index=False)
print(df_tests)

# ----------------- 5. 在完整训练+验证集上训练 Full+XGB，用于画混淆矩阵 -----------------
print("\n4. 训练 Full + XGBoost（使用全部训练样本），在500点验证集上评估...")

full_feats = feat_configs['Full (Ours)']
X_tr_full_all = df_train[full_feats].fillna(-9999)
y_tr_full_all = df_train['label']
X_va_full_all = df_val[full_feats].fillna(-9999)
y_va_full_all = df_val['label']

final_xgb = xgb.XGBClassifier(random_state=CFG.SEED, **XGB_PARAMS)
final_xgb.fit(X_tr_full_all, y_tr_full_all)
pred_final = final_xgb.predict(X_va_full_all)

oa_final = accuracy_score(y_va_full_all, pred_final)
kappa_final = cohen_kappa_score(y_va_full_all, pred_final)
f1_final = f1_score(y_va_full_all, pred_final, average='macro')
cm_final = confusion_matrix(y_va_full_all, pred_final)

print(f"Full+XGB 在500点验证集上的一次性结果：OA={oa_final*100:.2f}%，Kappa={kappa_final:.3f}，F1-macro={f1_final*100:.2f}%")
print("混淆矩阵：")
print(cm_final)

# ----------------- 6. 生成图：特征消融 + 模型对比 -----------------
print("\n5. 生成 Fig1：特征消融 + 模型对比 ...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(feat_order))
width = 0.32

m_xgb = [summary.loc[('Feature (XGB)', c), ('OA','mean')]*100 for c in feat_order]
s_xgb = [summary.loc[('Feature (XGB)', c), ('OA','std')]*100 for c in feat_order]
m_rf  = [summary.loc[('Feature (RF)',  c), ('OA','mean')]*100 for c in feat_order]
s_rf  = [summary.loc[('Feature (RF)',  c), ('OA','std')]*100 for c in feat_order]

axes[0].bar(x - width/2, m_xgb, width, yerr=s_xgb,
            label='XGBoost',
            color='#1f78b4', edgecolor='black', linewidth=0.6,
            capsize=3, error_kw={'linewidth':0.8})
axes[0].bar(x + width/2, m_rf,  width, yerr=s_rf,
            label='Random Forest',
            color='#33a02c', edgecolor='black', linewidth=0.6,
            capsize=3, error_kw={'linewidth':0.8})

axes[0].set_xticks(x)
axes[0].set_xticklabels(feat_order, rotation=15, ha='right')
axes[0].set_ylabel('Overall accuracy (%)')
axes[0].set_ylim(84, 96)
axes[0].set_title('(a) Feature ablation')
axes[0].legend(frameon=False, loc='lower right')
axes[0].grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=axes[0])

m_model = [summary.loc[('Model', m), ('OA','mean')]*100 for m in model_order]
s_model = [summary.loc[('Model', m), ('OA','std')]*100 for m in model_order]
palette_models = ['#33a02c','#ff7f00','#1f78b4','#6a3d9a']

axes[1].bar(model_order, m_model, yerr=s_model,
            color=palette_models, edgecolor='black', linewidth=0.6,
            capsize=3, error_kw={'linewidth':0.8})
axes[1].set_ylabel('Overall accuracy (%)')
axes[1].set_ylim(88, 96)
axes[1].set_title('(b) Classifier comparison (Full features)')
axes[1].grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(CFG.out_dir / "Fig1_Ablation_and_Models_v3.png", dpi=600, bbox_inches='tight')
plt.savefig(CFG.out_dir / "Fig1_Ablation_and_Models_v3.pdf", dpi=600, bbox_inches='tight')

# ----------------- 7. 生成 Fig2：混淆矩阵 -----------------
print("6. 生成 Fig2：混淆矩阵 ...")

class_names = ['Grassland','Shrubland']
cm_norm = cm_final.astype(float) / cm_final.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(3.8, 3.6))
sns.heatmap(
    cm_norm, annot=True, fmt='.02f',
    cmap='Blues', cbar_kws={'shrink':0.7},
    vmin=0, vmax=1, square=True,
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={'size':11,'color':'white'},
    ax=ax
)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title('Confusion matrix (Full + XGBoost, 500-val)')
plt.tight_layout()
plt.savefig(CFG.out_dir / "Fig2_Confusion_Matrix_v3.png", dpi=600, bbox_inches='tight')
plt.savefig(CFG.out_dir / "Fig2_Confusion_Matrix_v3.pdf", dpi=600, bbox_inches='tight')

# ----------------- 8. 标准差检查 -----------------
print("\n7. 标准差检查")
std_check = df_res.groupby(['Type','Config'])['OA'].std() * 100
print(f"OA 标准差范围: {std_check.min():.2f}% ~ {std_check.max():.2f}%")
print(f"平均标准差:     {std_check.mean():.2f}%")

print(f"\n✅ 全部完成，结果与图件已保存至: {CFG.out_dir}")
for f in sorted(CFG.out_dir.glob('*')):
    print("  -", f.name)
