# ============================================================
# SPH6004 Assignment 1 — v3: SMOTE + Tomek Links + More Features
# Target: icu_death_flag
# Changes from v2:
#   - SMOTE-Tomek on training data to handle class imbalance
#   - RFE target 35 features (instead of 25)
#   - Models trained on SMOTE-resampled data, tested on original
# ============================================================

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, 
                                       f_classif, mutual_info_classif, RFE)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, fbeta_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             average_precision_score, balanced_accuracy_score,
                             precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

RFE_TARGET = 35  # more features than v2's 25

# ── 1. DATA LOADING ──
print("=" * 70)
print("1. DATA LOADING")
print("=" * 70)
df = pd.read_csv(r'D:\study\SPH 6004\Assignment1_mimic dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nTarget variable (icu_death_flag) distribution:")
print(df['icu_death_flag'].value_counts())
print(f"\nClass proportions:")
print(df['icu_death_flag'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

# ── 2. DATA PREPROCESSING ──
print("\n" + "=" * 70)
print("2. DATA PREPROCESSING")
print("=" * 70)

leakage_cols = ['subject_id', 'hadm_id', 'stay_id',
                'deathtime', 'hospital_expire_flag',
                'intime', 'outtime',
                'los',
                'first_careunit', 'last_careunit']
leakage_present = [c for c in leakage_cols if c in df.columns]
df_clean = df.drop(columns=leakage_present)
print(f"Removed {len(leakage_present)} leakage/ID columns: {leakage_present}")
print(f"Remaining: {df_clean.shape[1]} columns")

target_col = 'icu_death_flag'
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in df_clean.select_dtypes(include=['number']).columns if c != target_col]
print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

df_encoded = df_clean.copy()
if 'race' in df_encoded.columns:
    race_map = {}
    for race in df_encoded['race'].unique():
        r = str(race).upper()
        if 'WHITE' in r: race_map[race] = 'WHITE'
        elif 'BLACK' in r: race_map[race] = 'BLACK'
        elif 'ASIAN' in r: race_map[race] = 'ASIAN'
        elif 'HISPANIC' in r or 'LATINO' in r: race_map[race] = 'HISPANIC'
        else: race_map[race] = 'OTHER'
    df_encoded['race'] = df_encoded['race'].map(race_map)
if 'gender' in df_encoded.columns:
    df_encoded['gender'] = (df_encoded['gender'] == 'M').astype(int)
cat_cols_remaining = df_encoded.select_dtypes(include=['object']).columns.tolist()
if cat_cols_remaining:
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols_remaining, drop_first=True)
print(f"After encoding: {df_encoded.shape}")

missing_pct = df_encoded.isnull().mean()
high_missing = missing_pct[missing_pct > 0.5].index.tolist()
if target_col in high_missing:
    high_missing.remove(target_col)
df_encoded = df_encoded.drop(columns=high_missing)
print(f"Dropped {len(high_missing)} columns with >50% missing. Remaining: {df_encoded.shape[1]}")

y = df_encoded[target_col]
X = df_encoded.drop(columns=[target_col])
num_cols_final = X.select_dtypes(include=['number']).columns
cat_cols_final = X.select_dtypes(include=['object', 'bool']).columns
X[num_cols_final] = X[num_cols_final].fillna(X[num_cols_final].median())
if len(cat_cols_final) > 0:
    X[cat_cols_final] = X[cat_cols_final].fillna(X[cat_cols_final].mode().iloc[0])
print(f"Final: X={X.shape}, y={y.shape}, missing={X.isnull().sum().sum()}")

# ── 3. FEATURE ENGINEERING ──
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING")
print("=" * 70)

vital_sign_triplets = [
    ('heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'heart_rate', 80),
    ('sbp_min', 'sbp_max', 'sbp_mean', 'sbp', 120),
    ('dbp_min', 'dbp_max', 'dbp_mean', 'dbp', 80),
    ('mbp_min', 'mbp_max', 'mbp_mean', 'mbp', 93),
    ('resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'resp_rate', 16),
    ('spo2_min', 'spo2_max', 'spo2_mean', 'spo2', 97),
    ('temperature_min', 'temperature_max', 'temperature_mean', 'temperature', 37.0),
    ('glucose_min', 'glucose_max', 'glucose_mean', 'glucose', 100),
]
new_features = []
for col_min, col_max, col_mean, name, normal_val in vital_sign_triplets:
    if all(c in X.columns for c in [col_min, col_max, col_mean]):
        X[f'{name}_range'] = X[col_max] - X[col_min]
        X[f'{name}_cv'] = X[f'{name}_range'] / (X[col_mean].abs() + 1e-6)
        X[f'{name}_dev_from_normal'] = (X[col_mean] - normal_val).abs()
        new_features.extend([f'{name}_range', f'{name}_cv', f'{name}_dev_from_normal'])

lab_pairs = [
    ('wbc_min', 'wbc_max'), ('hemoglobin_min', 'hemoglobin_max'),
    ('platelets_min', 'platelets_max'), ('sodium_min', 'sodium_max'),
    ('potassium_min', 'potassium_max'), ('bicarbonate_min', 'bicarbonate_max'),
    ('chloride_min', 'chloride_max'), ('bun_min', 'bun_max'),
    ('creatinine_min', 'creatinine_max'), ('glucose_lab_min', 'glucose_lab_max'),
]
for col_min, col_max in lab_pairs:
    if all(c in X.columns for c in [col_min, col_max]):
        feat_name = f'{col_min.replace("_min", "")}_lab_range'
        X[feat_name] = X[col_max] - X[col_min]
        new_features.append(feat_name)

X[new_features] = X[new_features].fillna(X[new_features].median())
new_corr = X[new_features].corrwith(y).abs().sort_values(ascending=False)
print(f"Engineered {len(new_features)} new features. Top 5 by |r| with target:")
for feat, corr in new_corr.head(5).items():
    print(f"  {feat}: r = {corr:.4f}")
print(f"Total features: {X.shape[1]}")

# ── 4. TRAIN-TEST SPLIT ──
print("\n" + "=" * 70)
print("4. TRAIN-TEST SPLIT & SCALING")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                             columns=X_test.columns, index=X_test.index)
print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Train target: {y_train.value_counts(normalize=True).to_dict()}")

# ── 5. FEATURE SELECTION (on original imbalanced data — no SMOTE yet) ──
print("\n" + "=" * 70)
print("5. FEATURE SELECTION")
print("=" * 70)

n_stage0 = X_train_scaled.shape[1]

vt = VarianceThreshold(threshold=0.01)
vt.fit(X_train_scaled)
vt_features = X_train_scaled.columns[vt.get_support()].tolist()
X_train_vt = X_train_scaled[vt_features]
n_stage1 = len(vt_features)
print(f"Stage 1 (Variance Threshold): {n_stage0} -> {n_stage1}")

corr_matrix = X_train_vt.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.85)]
corr_features = [c for c in vt_features if c not in high_corr_cols]
X_train_corr = X_train_vt[corr_features]
n_stage2 = len(corr_features)
print(f"Stage 2 (Correlation Filter): {n_stage1} -> {n_stage2}")

k_anova = min(len(corr_features), 50)
selector_f = SelectKBest(f_classif, k=k_anova)
selector_f.fit(X_train_corr, y_train)
f_selected = set(X_train_corr.columns[selector_f.get_support()])
mi_scores = mutual_info_classif(X_train_corr, y_train, random_state=42)
mi_selected = set(X_train_corr.columns[mi_scores > 0.01])
union_selected = sorted(f_selected | mi_selected)
X_train_uni = X_train_corr[union_selected]
n_stage3 = len(union_selected)
print(f"Stage 3 (ANOVA F + MI union): {n_stage2} -> {n_stage3}")

lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=1000, random_state=42)
lasso.fit(X_train_uni, y_train)
lasso_mask = np.abs(lasso.coef_[0]) > 1e-6
lasso_selected = [f for f, m in zip(union_selected, lasso_mask) if m]
X_train_lasso = X_train_uni[lasso_selected]
n_stage4 = len(lasso_selected)
print(f"Stage 4 (L1/Lasso): {n_stage3} -> {n_stage4}")

rfe_target = min(RFE_TARGET, n_stage4)  # can't select more than available
rfe = RFE(estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
          n_features_to_select=rfe_target)
rfe.fit(X_train_lasso, y_train)
final_features = [f for f, m in zip(lasso_selected, rfe.support_) if m]
n_stage5 = len(final_features)
print(f"Stage 5 (RFE): {n_stage4} -> {n_stage5}")
print(f"\nFinal {n_stage5} features: {final_features}")

stage_counts = [n_stage0, n_stage1, n_stage2, n_stage3, n_stage4, n_stage5]

X_train_final = X_train_scaled[final_features]
X_test_final = X_test_scaled[final_features]

# ── 5.5 SMOTE + Tomek Links on training data ──
print("\n" + "=" * 70)
print("5.5 SMOTE + TOMEK LINKS (Resampling)")
print("=" * 70)
print(f"Before SMOTE: {X_train_final.shape}, class distribution: {y_train.value_counts().to_dict()}")

smote_tomek = SMOTETomek(
    smote=SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5),
    random_state=42
)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_final, y_train)
X_train_resampled = pd.DataFrame(X_train_resampled, columns=final_features)

print(f"After SMOTE-Tomek: {X_train_resampled.shape}, class distribution: "
      f"{pd.Series(y_train_resampled).value_counts().to_dict()}")
smote_ratio = pd.Series(y_train_resampled).value_counts(normalize=True).to_dict()
print(f"New class proportions: {smote_ratio}")

# Also keep original (un-resampled) for feature selection diagnostics
X_train_lasso_for_plots = X_train_scaled[lasso_selected]

print("\nPre-computing Lasso path...")
C_values_path = np.logspace(-3, 1, 20)
coef_paths = []
for C_val in C_values_path:
    lr_temp = LogisticRegression(penalty='l1', C=C_val, solver='liblinear', max_iter=1000, random_state=42)
    lr_temp.fit(X_train_scaled[X_train_lasso.columns], y_train)
    coef_paths.append(lr_temp.coef_[0])
coef_paths = np.array(coef_paths)

print("Pre-computing RFE learning curve...")
n_features_range = list(range(5, len(lasso_selected) + 1, 3))
rfe_auc_means = []
rfe_auc_stds = []
lr_rfe_curve = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced')
for n_feat in n_features_range:
    rfe_temp = RFE(estimator=DecisionTreeClassifier(max_depth=5, random_state=42), n_features_to_select=n_feat)
    X_rfe_temp = rfe_temp.fit_transform(X_train_lasso, y_train)
    scores_temp = cross_val_score(lr_rfe_curve, X_rfe_temp, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    rfe_auc_means.append(scores_temp.mean())
    rfe_auc_stds.append(scores_temp.std())
rfe_auc_means = np.array(rfe_auc_means)
rfe_auc_stds = np.array(rfe_auc_stds)

# ═══════════════════════════════════════════════════════════════
# 6. MODEL TRAINING (on SMOTE-resampled data, test on original)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. MODEL TRAINING & EVALUATION (SMOTE-resampled training set)")
print("=" * 70)

all_results = []
all_predictions = {}
all_probabilities = {}

def evaluate_model(model, X_tr, X_te, y_tr, y_te, name):
    """Train, cross-validate (on resampled data), and evaluate on original test set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    results = {
        'Model': name,
        'CV AUC Mean': cv_scores.mean(),
        'CV AUC Std': cv_scores.std(),
        'CV AUC (mean+/-std)': f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}",
        'Test Accuracy': accuracy_score(y_te, y_pred),
        'Test Precision': precision_score(y_te, y_pred, zero_division=0),
        'Test Recall': recall_score(y_te, y_pred, zero_division=0),
        'Test F1': f1_score(y_te, y_pred, zero_division=0),
        'Test AUC': roc_auc_score(y_te, y_prob),
    }
    print(f"  {name}: CV AUC = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}, "
          f"Test AUC = {results['Test AUC']:.4f}, Recall = {results['Test Recall']:.4f}, "
          f"Precision = {results['Test Precision']:.4f}, F1 = {results['Test F1']:.4f}")
    return results, y_pred, y_prob

# Model 1: LR — no class_weight since SMOTE already balanced
print("\n[1/6] Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(penalty='l2', max_iter=1000, random_state=42, solver='lbfgs'),
    {'C': [0.01, 0.1, 1.0, 10.0]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Best params: {lr_grid.best_params_}")
lr_model = lr_grid.best_estimator_
lr_results, lr_pred, lr_prob = evaluate_model(
    lr_model, X_train_resampled, X_test_final, y_train_resampled, y_test, 'Logistic Regression')
all_results.append(lr_results)
all_predictions['Logistic Regression'] = lr_pred
all_probabilities['Logistic Regression'] = lr_prob

# Model 2: DT
print("\n[2/6] Decision Tree...")
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, criterion='gini'),
    {'max_depth': [4, 6, 8, 10, 12], 'min_samples_leaf': [5, 10, 20]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
dt_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Best params: {dt_grid.best_params_}")
dt_model = dt_grid.best_estimator_
dt_results, dt_pred, dt_prob = evaluate_model(
    dt_model, X_train_resampled, X_test_final, y_train_resampled, y_test, 'Decision Tree')
all_results.append(dt_results)
all_predictions['Decision Tree'] = dt_pred
all_probabilities['Decision Tree'] = dt_prob

# Model 3: SVM RBF — subsample from resampled data
print("\n[3/6] SVM (RBF Kernel)...")
SVM_SAMPLE = 15000
n_resampled = len(X_train_resampled)
if n_resampled > SVM_SAMPLE:
    from sklearn.model_selection import train_test_split as tts_sub
    X_sub, _, y_sub, _ = tts_sub(
        X_train_resampled, y_train_resampled,
        train_size=SVM_SAMPLE, random_state=42, stratify=y_train_resampled)
else:
    X_sub, y_sub = X_train_resampled, y_train_resampled

svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
    cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
svm_grid.fit(X_sub, y_sub)
print(f"  Best params: {svm_grid.best_params_}")
cv_svm_mean = svm_grid.best_score_
cv_svm_std = svm_grid.cv_results_['std_test_score'][svm_grid.best_index_]
svm_model = svm_grid.best_estimator_
svm_pred = svm_model.predict(X_test_final)
svm_prob = svm_model.predict_proba(X_test_final)[:, 1]
svm_results = {
    'Model': 'SVM (RBF Kernel)',
    'CV AUC Mean': cv_svm_mean,
    'CV AUC Std': cv_svm_std,
    'CV AUC (mean+/-std)': f"{cv_svm_mean:.3f} +/- {cv_svm_std:.3f}",
    'Test Accuracy': accuracy_score(y_test, svm_pred),
    'Test Precision': precision_score(y_test, svm_pred, zero_division=0),
    'Test Recall': recall_score(y_test, svm_pred, zero_division=0),
    'Test F1': f1_score(y_test, svm_pred, zero_division=0),
    'Test AUC': roc_auc_score(y_test, svm_prob),
}
print(f"  SVM (RBF Kernel): CV AUC = {cv_svm_mean:.4f} +/- {cv_svm_std:.4f}, "
      f"Test AUC = {svm_results['Test AUC']:.4f}, Recall = {svm_results['Test Recall']:.4f}, "
      f"Precision = {svm_results['Test Precision']:.4f}, F1 = {svm_results['Test F1']:.4f}")
all_results.append(svm_results)
all_predictions['SVM (RBF Kernel)'] = svm_pred
all_probabilities['SVM (RBF Kernel)'] = svm_prob

# Control: SVM Linear
print("\n  [Control] SVM Linear Kernel...")
linear_svm = CalibratedClassifierCV(
    LinearSVC(C=0.01, max_iter=10000, random_state=42, dual=True), cv=3, method='sigmoid')
linear_svm.fit(X_sub, y_sub)
y_prob_linear = linear_svm.predict_proba(X_test_final)[:, 1]
cv_linear = cross_val_score(
    CalibratedClassifierCV(LinearSVC(C=0.01, max_iter=10000, random_state=42, dual=True), cv=3),
    X_sub, y_sub, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
auc_linear = roc_auc_score(y_test, y_prob_linear)
print(f"  SVM Linear: CV AUC = {cv_linear.mean():.4f}, Test AUC = {auc_linear:.4f}")
print(f"  SVM RBF vs Linear gap: {svm_results['Test AUC'] - auc_linear:.4f}")

# Model 4: RF
print("\n[4/6] Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200], 'max_depth': [6, 10, 15], 'min_samples_leaf': [5, 10]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Best params: {rf_grid.best_params_}")
rf_model = rf_grid.best_estimator_
rf_results, rf_pred, rf_prob = evaluate_model(
    rf_model, X_train_resampled, X_test_final, y_train_resampled, y_test, 'Random Forest')
all_results.append(rf_results)
all_predictions['Random Forest'] = rf_pred
all_probabilities['Random Forest'] = rf_prob

# Model 5: XGBoost — no scale_pos_weight since SMOTE already balanced
print("\n[5/6] XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    {'n_estimators': [100, 200], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Best params: {xgb_grid.best_params_}")
xgb_model = xgb_grid.best_estimator_
xgb_results, xgb_pred, xgb_prob = evaluate_model(
    xgb_model, X_train_resampled, X_test_final, y_train_resampled, y_test, 'XGBoost')
all_results.append(xgb_results)
all_predictions['XGBoost'] = xgb_pred
all_probabilities['XGBoost'] = xgb_prob

# Model 6: AdaBoost
print("\n[6/6] AdaBoost...")
ada_base = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_grid = GridSearchCV(
    AdaBoostClassifier(estimator=ada_base, random_state=42),
    {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
ada_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Best params: {ada_grid.best_params_}")
ada_model = ada_grid.best_estimator_
ada_results, ada_pred, ada_prob = evaluate_model(
    ada_model, X_train_resampled, X_test_final, y_train_resampled, y_test, 'AdaBoost')
all_results.append(ada_results)
all_predictions['AdaBoost'] = ada_pred
all_probabilities['AdaBoost'] = ada_prob

# ═══════════════════════════════════════════════════════════════
# 7. RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 90)
results_df = pd.DataFrame(all_results)
display_cols = ['Model', 'CV AUC (mean+/-std)', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']
results_display = results_df[display_cols].copy()
for col in ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']:
    results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}")
print(results_display.to_string(index=False))

# All vs Selected (using original imbalanced data for fair comparison)
print("\n" + "=" * 70)
print("ALL vs SELECTED FEATURES COMPARISON")
print("=" * 70)
cv_comp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison_results = []
models_for_comparison = {
    'Logistic Regression': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42, class_weight='balanced'),
}
for mn, mod in models_for_comparison.items():
    m_all = type(mod)(**mod.get_params())
    cv_a = cross_val_score(m_all, X_train_scaled, y_train, cv=cv_comp, scoring='roc_auc', n_jobs=-1).mean()
    m_all.fit(X_train_scaled, y_train)
    auc_a = roc_auc_score(y_test, m_all.predict_proba(X_test_scaled)[:, 1])
    f1_a = f1_score(y_test, m_all.predict(X_test_scaled))
    m_sel = type(mod)(**mod.get_params())
    cv_s = cross_val_score(m_sel, X_train_final, y_train, cv=cv_comp, scoring='roc_auc', n_jobs=-1).mean()
    m_sel.fit(X_train_final, y_train)
    auc_s = roc_auc_score(y_test, m_sel.predict_proba(X_test_final)[:, 1])
    f1_s = f1_score(y_test, m_sel.predict(X_test_final))
    comparison_results.append({
        'Model': mn, 'All Features AUC': auc_a, 'Selected Features AUC': auc_s,
        'All Features F1': f1_a, 'Selected Features F1': f1_s,
        'Feature Reduction': f"{X_train_scaled.shape[1]} -> {len(final_features)}"
    })
print(pd.DataFrame(comparison_results).to_string(index=False))

# Extended Clinical Metrics
print("\n" + "=" * 70)
print("EXTENDED CLINICAL METRICS")
print("=" * 70)
extended_results = []
for r in all_results:
    name = r['Model']
    y_pred_ext = all_predictions[name]
    y_prob_ext = all_probabilities[name]
    pr_auc = average_precision_score(y_test, y_prob_ext)
    bal_acc = balanced_accuracy_score(y_test, y_pred_ext)
    cm_ext = confusion_matrix(y_test, y_pred_ext)
    tn_e, fp_e, fn_e, tp_e = cm_ext.ravel()
    npv = tn_e / (tn_e + fn_e) if (tn_e + fn_e) > 0 else 0
    extended_results.append({'Model': name, 'PR-AUC': pr_auc, 'Balanced Accuracy': bal_acc, 'NPV': npv})
print(f"{'Model':>25s}  {'PR-AUC':>8s}  {'Bal.Acc':>8s}  {'NPV':>8s}")
for ext_r in extended_results:
    print(f"{ext_r['Model']:>25s}  {ext_r['PR-AUC']:>8.4f}  {ext_r['Balanced Accuracy']:>8.4f}  {ext_r['NPV']:>8.4f}")

# Confusion matrix analysis
print(f"\n{'='*70}")
print("CONFUSION MATRIX ANALYSIS")
print(f"{'='*70}")
for model_name, y_pred in all_predictions.items():
    cm_val = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm_val.ravel()
    print(f"{model_name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, "
          f"Sensitivity={tp/(tp+fn):.4f}, Specificity={tn/(tn+fp):.4f}")

# ═══════════════════════════════════════════════════════════════
# 8. THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION")
print("=" * 80)

threshold_results = []
for r in all_results:
    name = r['Model']
    y_prob_th = all_probabilities[name]
    y_pred_default = all_predictions[name]
    fpr_th, tpr_th, roc_thresholds = roc_curve(y_test, y_prob_th)
    j_scores = tpr_th - fpr_th
    youden_threshold = roc_thresholds[np.argmax(j_scores)]
    prec_th, rec_th, pr_thresholds = precision_recall_curve(y_test, y_prob_th)
    f1_scores_th = 2 * (prec_th[:-1] * rec_th[:-1]) / (prec_th[:-1] + rec_th[:-1] + 1e-10)
    f1_threshold = pr_thresholds[np.argmax(f1_scores_th)]
    beta = 2
    f2_scores_th = (1 + beta**2) * (prec_th[:-1] * rec_th[:-1]) / (beta**2 * prec_th[:-1] + rec_th[:-1] + 1e-10)
    f2_threshold = pr_thresholds[np.argmax(f2_scores_th)]
    cm_def = confusion_matrix(y_test, y_pred_default)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel()
    threshold_results.append({
        'Model': name, 'Strategy': 'Model Default', 'Threshold': np.nan,
        'Precision': precision_score(y_test, y_pred_default, zero_division=0),
        'Recall': recall_score(y_test, y_pred_default, zero_division=0),
        'F1': f1_score(y_test, y_pred_default, zero_division=0),
        'F2': fbeta_score(y_test, y_pred_default, beta=2, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred_default),
        'NPV': tn_d / (tn_d + fn_d) if (tn_d + fn_d) > 0 else 0,
        'Specificity': tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0,
    })
    for strategy, thresh in [('Youden J', youden_threshold), ('F1-optimal', f1_threshold), ('F2-optimal', f2_threshold)]:
        y_pred_th = (y_prob_th >= thresh).astype(int)
        cm_th = confusion_matrix(y_test, y_pred_th)
        tn_th, fp_th, fn_th, tp_th = cm_th.ravel()
        threshold_results.append({
            'Model': name, 'Strategy': strategy, 'Threshold': thresh,
            'Precision': precision_score(y_test, y_pred_th, zero_division=0),
            'Recall': recall_score(y_test, y_pred_th, zero_division=0),
            'F1': f1_score(y_test, y_pred_th, zero_division=0),
            'F2': fbeta_score(y_test, y_pred_th, beta=2, zero_division=0),
            'Accuracy': accuracy_score(y_test, y_pred_th),
            'NPV': tn_th / (tn_th + fn_th) if (tn_th + fn_th) > 0 else 0,
            'Specificity': tn_th / (tn_th + fp_th) if (tn_th + fp_th) > 0 else 0,
        })

threshold_df = pd.DataFrame(threshold_results)
for name in [r['Model'] for r in all_results]:
    model_th = threshold_df[threshold_df['Model'] == name]
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    for _, row in model_th.iterrows():
        th_str = f"{row['Threshold']:.3f}" if not np.isnan(row['Threshold']) else "predict"
        print(f"  {row['Strategy']:<16s}  Thresh={th_str:>7s}  Prec={row['Precision']:.3f}  "
              f"Rec={row['Recall']:.3f}  F1={row['F1']:.3f}  F2={row['F2']:.3f}")

best_auc_model = max(all_results, key=lambda x: x['Test AUC'])
best_recall_model = max(all_results, key=lambda x: x['Test Recall'])
best_f1_model = max(all_results, key=lambda x: x['Test F1'])
print(f"\n{'='*70}")
print(f"Best by AUC:    {best_auc_model['Model']} ({best_auc_model['Test AUC']:.4f})")
print(f"Best by F1:     {best_f1_model['Model']} ({best_f1_model['Test F1']:.4f})")
print(f"Best by Recall: {best_recall_model['Model']} ({best_recall_model['Test Recall']:.4f})")

# ═══════════════════════════════════════════════════════════════
# 9. FIGURE EXPORT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. FIGURE EXPORT")
print("=" * 70)

report_fig_dir = r'D:\study\SPH 6004\report_figures_v3'
os.makedirs(report_fig_dir, exist_ok=True)

# Fig 1: Target Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['icu_death_flag'].value_counts().sort_index()
labels = ['Survived (0)', 'ICU Death (1)']
colors_fig = ['#2ecc71', '#e74c3c']
bars = ax.bar(labels, counts.values, color=colors_fig, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    pct = val / counts.sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Target Variable Distribution (icu_death_flag)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig1_target_distribution.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 2: Feature Reduction
reduction_labels = ['Original', 'Stage 1\nVariance', 'Stage 2\nCorrelation',
                    'Stage 3\nUnivariate', 'Stage 4\nL1/Lasso', 'Stage 5\nRFE']
fig, ax = plt.subplots(figsize=(8, 4.5))
cmap_b = plt.cm.Blues
colors_bar = [cmap_b(0.3 + 0.7 * i / (len(stage_counts)-1)) for i in range(len(stage_counts))]
bars = ax.bar(reduction_labels, stage_counts, color=colors_bar, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, stage_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, str(val),
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_title('Progressive Feature Reduction Through Pipeline', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(stage_counts) * 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig2_feature_reduction.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 2b: SMOTE class distribution before/after
fig, axes_smote = plt.subplots(1, 2, figsize=(10, 4))
# Before
before_counts = y_train.value_counts().sort_index()
axes_smote[0].bar(['Survived (0)', 'ICU Death (1)'], before_counts.values,
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
for i, v in enumerate(before_counts.values):
    axes_smote[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
axes_smote[0].set_title('Before SMOTE', fontsize=12, fontweight='bold')
axes_smote[0].set_ylabel('Count')
# After
after_counts = pd.Series(y_train_resampled).value_counts().sort_index()
axes_smote[1].bar(['Survived (0)', 'ICU Death (1)'], after_counts.values,
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
for i, v in enumerate(after_counts.values):
    axes_smote[1].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
axes_smote[1].set_title('After SMOTE-Tomek', fontsize=12, fontweight='bold')
axes_smote[1].set_ylabel('Count')
plt.suptitle('Training Set Class Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig2b_smote_distribution.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 3: ROC Curves
fig, ax = plt.subplots(figsize=(7, 5))
_roc_colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_results), 9)))
for idx_roc, result in enumerate(all_results):
    nm = result['Model']
    fpr_fig, tpr_fig, _ = roc_curve(y_test, all_probabilities[nm])
    ax.plot(fpr_fig, tpr_fig, color=_roc_colors[idx_roc], lw=2, label=f'{nm} (AUC = {result["Test AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig3_roc_curves.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 4: Confusion Matrices
_n_mod = len(all_results); _ncols4 = min(_n_mod, 3); _nrows4 = (_n_mod + _ncols4 - 1) // _ncols4
fig, axes_cm = plt.subplots(_nrows4, _ncols4, figsize=(5 * _ncols4, 4 * _nrows4))
axes_cm_flat = np.array(axes_cm).flatten()
for idx, result in enumerate(all_results):
    nm = result['Model']
    cm_fig = confusion_matrix(y_test, all_predictions[nm])
    sns.heatmap(cm_fig, annot=True, fmt='d', cmap='Blues', ax=axes_cm_flat[idx],
                xticklabels=['Survived', 'ICU Death'], yticklabels=['Survived', 'ICU Death'], annot_kws={'size': 11})
    axes_cm_flat[idx].set_title(nm, fontsize=10, fontweight='bold')
    axes_cm_flat[idx].set_ylabel('Actual' if idx % _ncols4 == 0 else '')
    axes_cm_flat[idx].set_xlabel('Predicted')
for idx in range(_n_mod, len(axes_cm_flat)):
    axes_cm_flat[idx].set_visible(False)
plt.suptitle('Confusion Matrices Comparison', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig4_confusion_matrices.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 8: Lasso Path
fig, ax = plt.subplots(figsize=(8, 5))
for j_idx in range(coef_paths.shape[1]):
    ax.plot(np.log10(C_values_path), coef_paths[:, j_idx], lw=1.2, alpha=0.7)
ax.axvline(x=np.log10(0.1), color='red', linestyle='--', lw=1.5, label='C=0.1 (selected)')
ax.set_xlabel('log10(C)', fontsize=11); ax.set_ylabel('Coefficient value', fontsize=11)
ax.set_title('Lasso Regularization Path (L1 Logistic Regression)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig8_lasso_path.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 9: RFE Learning Curve
optimal_n = n_features_range[np.argmax(rfe_auc_means)]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_features_range, rfe_auc_means, 'b-o', markersize=4, lw=1.5, label='Mean CV AUC')
ax.fill_between(n_features_range, rfe_auc_means - rfe_auc_stds, rfe_auc_means + rfe_auc_stds, alpha=0.2, color='blue')
ax.axvline(x=rfe_target, color='red', linestyle='--', lw=1.5, label=f'n={rfe_target} (selected)')
ax.axvline(x=optimal_n, color='green', linestyle=':', lw=1.5, label=f'n={optimal_n} (optimal)')
ax.set_xlabel('Number of Features', fontsize=11); ax.set_ylabel('CV AUC (3-fold)', fontsize=11)
ax.set_title('RFE Feature Selection Learning Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig9_rfe_learning_curve.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 10: PR Curves
_pr_colors2 = plt.cm.Set1(np.linspace(0, 1, max(len(all_results), 9)))
fig, ax = plt.subplots(figsize=(7, 5))
for idx_pr2, result in enumerate(all_results):
    nm = result['Model']
    prec_v, rec_v, _ = precision_recall_curve(y_test, all_probabilities[nm])
    pr_auc_v = average_precision_score(y_test, all_probabilities[nm])
    ax.plot(rec_v, prec_v, color=_pr_colors2[idx_pr2], lw=2, label=f'{nm} (PR-AUC = {pr_auc_v:.3f})')
prevalence_fig = y_test.mean()
ax.axhline(y=prevalence_fig, color='grey', linestyle='--', lw=1, label=f'Baseline (prevalence = {prevalence_fig:.3f})')
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9); ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig10_pr_curves.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Fig 11: Threshold Optimization
top3 = sorted(all_results, key=lambda x: x['Test AUC'], reverse=True)[:3]
top3_names = [m['Model'] for m in top3]
fig, axes_th = plt.subplots(1, 3, figsize=(18, 5))
th_colors = {'Youden J': '#2196F3', 'F1-optimal': '#FF9800', 'F2-optimal': '#E91E63'}
for ax_i, nm_th in enumerate(top3_names):
    ax_th = axes_th[ax_i]
    y_prob_th_fig = all_probabilities[nm_th]
    th_range = np.linspace(0.01, 0.99, 200)
    prec_c, rec_c, f1_c, f2_c = [], [], [], []
    for t in th_range:
        yp_t = (y_prob_th_fig >= t).astype(int)
        prec_c.append(precision_score(y_test, yp_t, zero_division=0))
        rec_c.append(recall_score(y_test, yp_t, zero_division=0))
        f1_c.append(f1_score(y_test, yp_t, zero_division=0))
        f2_c.append(fbeta_score(y_test, yp_t, beta=2, zero_division=0))
    ax_th.plot(th_range, prec_c, 'b-', lw=2, label='Precision')
    ax_th.plot(th_range, rec_c, 'r-', lw=2, label='Recall')
    ax_th.plot(th_range, f1_c, 'g--', lw=1.5, label='F1', alpha=0.7)
    ax_th.plot(th_range, f2_c, 'm--', lw=1.5, label='F2', alpha=0.7)
    mth = threshold_df[(threshold_df['Model'] == nm_th) & (threshold_df['Strategy'] != 'Model Default')]
    for _, rw in mth.iterrows():
        if rw['Strategy'] in th_colors:
            c = th_colors[rw['Strategy']]
            ax_th.axvline(x=rw['Threshold'], color=c, linestyle=':', lw=1.5, alpha=0.8)
            ax_th.plot(rw['Threshold'], rw['F2'], 'o', color=c, markersize=8, zorder=5)
    ax_th.set_xlabel('Decision Threshold', fontsize=11); ax_th.set_ylabel('Score', fontsize=11)
    ax_th.set_title(nm_th, fontsize=12, fontweight='bold')
    ax_th.legend(fontsize=8, loc='center left'); ax_th.set_xlim([0,1]); ax_th.set_ylim([0,1]); ax_th.grid(True, alpha=0.3)
from matplotlib.lines import Line2D as L2D
leg_el = [L2D([0],[0], color=v, linestyle=':', lw=2, label=k) for k,v in th_colors.items()]
fig.legend(handles=leg_el, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Threshold Impact on Precision-Recall Trade-off (Top 3 Models)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(report_fig_dir, 'fig11_threshold_optimization.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\nAll figures exported to: {report_fig_dir}")
for f in sorted(os.listdir(report_fig_dir)):
    print(f"  {f}")

# ═══════════════════════════════════════════════════════════════
# 10. SAVE RESULTS TO JSON
# ═══════════════════════════════════════════════════════════════
report_data = {
    'target_col': target_col,
    'dataset_shape': list(df.shape),
    'class_distribution': df['icu_death_flag'].value_counts().to_dict(),
    'stage_counts': stage_counts,
    'final_features': final_features,
    'n_train': int(X_train_final.shape[0]),
    'n_test': int(X_test_final.shape[0]),
    'n_total_features': int(X_train_scaled.shape[1]),
    'n_train_resampled': int(X_train_resampled.shape[0]),
    'smote_class_distribution': pd.Series(y_train_resampled).value_counts().to_dict(),
    'all_results': [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for r in all_results],
    'extended_results': extended_results,
    'comparison_results': comparison_results,
    'svm_linear_auc': float(auc_linear),
    'svm_linear_cv_auc': float(cv_linear.mean()),
    'best_params': {
        'LR': lr_grid.best_params_,
        'DT': dt_grid.best_params_,
        'SVM': svm_grid.best_params_,
        'RF': rf_grid.best_params_,
        'XGB': xgb_grid.best_params_,
        'Ada': ada_grid.best_params_,
    },
}

# Add confusion matrix details
cm_details = {}
for model_name, y_pred in all_predictions.items():
    cm_val = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm_val.ravel()
    cm_details[model_name] = {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
report_data['confusion_matrices'] = cm_details

# Add threshold optimization details
th_summary = {}
for name in [r['Model'] for r in all_results]:
    model_th = threshold_df[threshold_df['Model'] == name]
    th_summary[name] = {}
    for _, row in model_th.iterrows():
        th_summary[name][row['Strategy']] = {
            'Threshold': float(row['Threshold']) if not np.isnan(row['Threshold']) else None,
            'Precision': float(row['Precision']),
            'Recall': float(row['Recall']),
            'F1': float(row['F1']),
            'F2': float(row['F2']),
        }
report_data['threshold_optimization'] = th_summary

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy(i) for i in obj]
    return obj

report_data = convert_numpy(report_data)

with open(r'D:\study\SPH 6004\v3_results.json', 'w') as f:
    json.dump(report_data, f, indent=2)
print(f"\nResults saved to v3_results.json")
print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
