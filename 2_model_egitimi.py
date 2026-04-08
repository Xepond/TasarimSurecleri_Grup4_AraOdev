"""
XAI ile Onyargi Tespiti Projesi
Adim 2: Model Egitimi ve Fairness Metrikleri

3 Model: Logistic Regression, Random Forest, XGBoost
Fairness: Demographic Parity, Equalized Odds, Disparate Impact
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style='whitegrid')

print("=" * 60)
print("  ADIM 2: MODEL EGITIMI VE FAIRNESS METRIKLERI")
print("=" * 60)

# ============================================================
# 1. VERILERI YUKLE
# ============================================================
print("\n[1/4] Islenmis veriler yukleniyor...")
data = joblib.load(os.path.join(OUTPUT_DIR, 'processed_data.pkl'))

X_train_processed = data['X_train_processed']
X_test_processed = data['X_test_processed']
y_train = data['y_train']
y_test = data['y_test']
sex_test = data['sex_test']
race_test = data['race_test']
feature_names = data['feature_names']

print(f"  Egitim seti: {X_train_processed.shape}")
print(f"  Test seti:   {X_test_processed.shape}")

# ============================================================
# 2. MODELLERI EGIT
# ============================================================
print("\n[2/4] Modeller egitiliyor...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss'
    )
}

trained_models = {}
predictions = {}
metrics_table = []

for name, model in models.items():
    print(f"\n  --- {name} ---")
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    trained_models[name] = model
    predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics_table.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'AUC-ROC': auc
    })

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

# Performans tablosunu kaydet
metrics_df = pd.DataFrame(metrics_table)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performans.csv'), index=False)
print(f"\n  [OK] Performans tablosu kaydedildi: outputs/model_performans.csv")

# ============================================================
# 3. GORSELLESTIRMELER
# ============================================================
print("\n[3/4] Gorsellestirmeler olusturuluyor...")

# 3.1 Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Tahmin')
    axes[i].set_ylabel('Gercek')
plt.suptitle('Confusion Matrix Karsilastirmasi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '2_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print("  [OK] Confusion matrix grafikleri kaydedildi")

# 3.2 ROC Egrileri
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71']
for (name, preds), color in zip(predictions.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
    auc = roc_auc_score(y_test, preds['y_proba'])
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_title('ROC Egrileri Karsilastirmasi', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '2_roc_curves.png'), bbox_inches='tight')
plt.close()
print("  [OK] ROC egrileri grafigi kaydedildi")

# ============================================================
# 4. FAIRNESS METRIKLERI
# ============================================================
print("\n[4/4] Fairness metrikleri hesaplaniyor...")


def disparate_impact_ratio(y_pred, sensitive):
    """Disparate Impact Orani: min(grup oranlari) / max(grup oranlari)"""
    groups = sensitive.unique()
    rates = {}
    for g in groups:
        mask = (sensitive == g).values if hasattr(sensitive, 'values') else (sensitive == g)
        if mask.sum() > 0:
            rates[g] = y_pred[mask].mean()
    vals = list(rates.values())
    return min(vals) / max(vals) if max(vals) > 0 else np.nan


fairness_results = {}

for name, preds in predictions.items():
    y_pred = preds['y_pred']
    print(f"\n  --- {name} ---")

    # Cinsiyet (sex)
    dp_sex = demographic_parity_difference(y_test, y_pred, sensitive_features=sex_test)
    eo_sex = equalized_odds_difference(y_test, y_pred, sensitive_features=sex_test)
    di_sex = disparate_impact_ratio(y_pred, sex_test)

    print(f"    CINSIYET:")
    print(f"      Demographic Parity Difference: {dp_sex:.4f}")
    print(f"      Equalized Odds Difference:     {eo_sex:.4f}")
    print(f"      Disparate Impact Ratio:         {di_sex:.4f}  {'!! ONYARGI' if di_sex < 0.8 else 'Adil'}")

    # Irk (race)
    dp_race = demographic_parity_difference(y_test, y_pred, sensitive_features=race_test)
    eo_race = equalized_odds_difference(y_test, y_pred, sensitive_features=race_test)
    di_race = disparate_impact_ratio(y_pred, race_test)

    print(f"    IRK:")
    print(f"      Demographic Parity Difference: {dp_race:.4f}")
    print(f"      Equalized Odds Difference:     {eo_race:.4f}")
    print(f"      Disparate Impact Ratio:         {di_race:.4f}  {'!! ONYARGI' if di_race < 0.8 else 'Adil'}")

    fairness_results[name] = {
        'sex': {'dp': dp_sex, 'eo': eo_sex, 'di': di_sex},
        'race': {'dp': dp_race, 'eo': eo_race, 'di': di_race}
    }

# Fairness gorsellestirme
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sex_data = pd.DataFrame({
    name: [abs(r['sex']['dp']), abs(r['sex']['eo']), 1 - r['sex']['di']]
    for name, r in fairness_results.items()
}, index=['Demographic\nParity Diff', 'Equalized\nOdds Diff', '1 - Disparate\nImpact'])
sex_data.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Cinsiyet Bazli Fairness Metrikleri', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Deger (0 = Adil)')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(fontsize=9)

race_data = pd.DataFrame({
    name: [abs(r['race']['dp']), abs(r['race']['eo']), 1 - r['race']['di']]
    for name, r in fairness_results.items()
}, index=['Demographic\nParity Diff', 'Equalized\nOdds Diff', '1 - Disparate\nImpact'])
race_data.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
axes[1].set_title('Irk Bazli Fairness Metrikleri', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Deger (0 = Adil)')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(fontsize=9)

plt.suptitle('Fairness Karsilastirmasi (Yuksek = Daha fazla onyargi)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '2_fairness_metrics.png'), bbox_inches='tight')
plt.close()
print("\n  [OK] Fairness metrikleri grafigi kaydedildi")

# Pozitif tahmin oranlari (gruplar arasi)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sex_rates = {}
for name, preds in predictions.items():
    rates = {}
    for val in sex_test.unique():
        mask = (sex_test == val).values
        rates[val] = preds['y_pred'][mask].mean()
    sex_rates[name] = rates
pd.DataFrame(sex_rates).plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Cinsiyete Gore Pozitif Tahmin Orani', fontsize=13, fontweight='bold')
axes[0].set_ylabel('P(Gelir > 50K)')
axes[0].tick_params(axis='x', rotation=0)

race_rates = {}
for name, preds in predictions.items():
    rates = {}
    for val in race_test.unique():
        mask = (race_test == val).values
        rates[val] = preds['y_pred'][mask].mean()
    race_rates[name] = rates
pd.DataFrame(race_rates).plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
axes[1].set_title('Irka Gore Pozitif Tahmin Orani', fontsize=13, fontweight='bold')
axes[1].set_ylabel('P(Gelir > 50K)')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Gruplar Arasi Pozitif Tahmin Oranlari', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '2_pozitif_tahmin_oranlari.png'), bbox_inches='tight')
plt.close()
print("  [OK] Pozitif tahmin oranlari grafigi kaydedildi")

# ============================================================
# KAYDET
# ============================================================
joblib.dump({
    'models': trained_models,
    'predictions': predictions,
    'fairness_results': fairness_results
}, os.path.join(OUTPUT_DIR, 'models_and_results.pkl'))
print(f"\n  [OK] Modeller kaydedildi: outputs/models_and_results.pkl")

print("\n" + "=" * 60)
print("  ADIM 2 TAMAMLANDI!")
print("  Sonraki adim: python 3_xai_analiz.py")
print("=" * 60)
