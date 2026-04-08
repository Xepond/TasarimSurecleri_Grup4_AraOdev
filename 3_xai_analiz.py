"""
XAI ile Onyargi Tespiti Projesi
Adim 3: XAI Analizi - LIME, SHAP, Feature Flip (Counterfactual)

Ana model: Random Forest (tum XAI yontemleriyle uyumlu)
* Düzeltme: Çalışmayan DiCE kaldırıldı, Feature Flip korundu.
* Düzeltme: SHAP Dependence Plot sürekli değişken (yaş) ile entegre edildi.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
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
print("  ADIM 3: XAI ANALIZI")
print("  LIME - SHAP - Feature Flip Explanations")
print("=" * 60)

# ============================================================
# 1. VERILERI VE MODELLERI YUKLE
# ============================================================
print("\n[1/4] Veriler ve modeller yukleniyor...")
data = joblib.load(os.path.join(OUTPUT_DIR, 'processed_data.pkl'))
results = joblib.load(os.path.join(OUTPUT_DIR, 'models_and_results.pkl'))

X_test = data['X_test']
X_train_processed = data['X_train_processed']
X_test_processed = data['X_test_processed']
y_test = data['y_test']
sex_test = data['sex_test']
feature_names = data['feature_names']
preprocessor = data['preprocessor']

rf_model = results['models']['Random Forest']
print("  [OK] Random Forest modeli yuklendi")

X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# ============================================================
# 2. LIME ANALIZI
# ============================================================
print("\n[2/4] LIME analizi yapiliyor...")

lime_explainer = LimeTabularExplainer(
    X_train_processed,
    feature_names=feature_names,
    class_names=['<=50K', '>50K'],
    mode='classification',
    random_state=42
)

sample_indices = []
sample_labels = []

mask_male_pos = (sex_test.values == 'Male') & (results['predictions']['Random Forest']['y_pred'] == 1)
if len(np.where(mask_male_pos)[0]) > 0:
    sample_indices.append(np.where(mask_male_pos)[0][0])
    sample_labels.append('Erkek - Yuksek Gelir Tahmini')

mask_female_neg = (sex_test.values == 'Female') & (results['predictions']['Random Forest']['y_pred'] == 0)
if len(np.where(mask_female_neg)[0]) > 0:
    sample_indices.append(np.where(mask_female_neg)[0][0])
    sample_labels.append('Kadin - Dusuk Gelir Tahmini')

fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 5 * len(sample_indices)))
if len(sample_indices) == 1:
    axes = [axes]

lime_feature_weights = {}

for i, (idx, label) in enumerate(zip(sample_indices, sample_labels)):
    instance = X_test_processed[idx]
    exp = lime_explainer.explain_instance(instance, rf_model.predict_proba, num_features=10)

    top_label = exp.available_labels()[0]
    feature_weight_list = exp.as_list(label=top_label)
    lime_feature_weights[label] = feature_weight_list

    features_sorted = sorted(feature_weight_list, key=lambda x: abs(x[1]), reverse=True)
    feat_names_plot = [f[0][:40] for f in features_sorted]
    weights = [f[1] for f in features_sorted]
    colors_bar = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]

    axes[i].barh(range(len(feat_names_plot)), weights, color=colors_bar, edgecolor='black')
    axes[i].set_yticks(range(len(feat_names_plot)))
    axes[i].set_yticklabels(feat_names_plot, fontsize=9)
    axes[i].set_title(f'LIME Aciklamasi: {label}', fontsize=12, fontweight='bold')
    axes[i].invert_yaxis()

plt.suptitle('LIME - Bireysel Tahmin Aciklamalari', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '3_lime_aciklamalari.png'), bbox_inches='tight')
plt.close()
print("  [OK] LIME aciklamalari kaydedildi")

# LIME ortalama ağırlıklar
n_lime_samples = min(100, len(X_test_processed))
lime_sex_weights = []
rng = np.random.RandomState(42)
for idx in rng.choice(len(X_test_processed), n_lime_samples, replace=False):
    exp = lime_explainer.explain_instance(X_test_processed[idx], rf_model.predict_proba, num_features=len(feature_names))
    weight_dict = dict(exp.as_list(label=exp.available_labels()[0]))
    lime_sex_weights.append(sum(abs(v) for k, v in weight_dict.items() if 'sex' in k.lower()))

print(f"    Ortalama |sex| agirligi:  {np.mean(lime_sex_weights):.4f}")

# ============================================================
# 3. SHAP ANALIZI
# ============================================================
print("\n[3/4] SHAP analizi yapiliyor...")
shap_explainer = shap.TreeExplainer(rf_model)
n_shap_samples = min(500, len(X_test_processed))
X_shap = X_test_df.iloc[:n_shap_samples]
shap_values = shap_explainer.shap_values(X_shap.values)

sv = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values)

# SHAP Summary ve Bar Plot
plt.figure(figsize=(12, 10))
shap.summary_plot(sv, X_shap, show=False, max_display=15)
plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIGURE_DIR, '3_shap_summary.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(sv, X_shap, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Ortalama Ozellik Onemleri', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIGURE_DIR, '3_shap_bar.png'), bbox_inches='tight')
plt.close()

# DÜZELTME: Anlamlı SHAP Dependence Plot (Sürekli değişken + Kategorik etkileşimi)
age_features = [f for f in feature_names if 'age' in f.lower()]
sex_features = [f for f in feature_names if 'sex' in f.lower()]

if age_features and sex_features:
    age_col = age_features[0]
    sex_col = sex_features[0] # Örn: cat__sex_Female
    age_idx = list(X_shap.columns).index(age_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_shap[age_col].values, sv[:, age_idx], 
                         c=X_shap[sex_col].values, cmap='coolwarm', alpha=0.7, s=30, edgecolors='k')
    ax.set_xlabel('Yaş (Ölçeklendirilmiş)', fontsize=12)
    ax.set_ylabel('Yaş Özelliğinin SHAP Değeri', fontsize=12)
    ax.set_title(f'SHAP Etkileşimi: Yaş ve Cinsiyet ({sex_col})', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{sex_col} (0=Hayır, 1=Evet)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '3_shap_dependence_etkilesim.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Anlamlı SHAP dependence plot kaydedildi")

# ============================================================
# 4. COUNTERFACTUAL (FEATURE FLIP) ANALIZI
# ============================================================
print("\n[4/4] Feature Flip (Basit Counterfactual) analizi yapiliyor...")

X_test_flipped_sex = X_test.copy()
sex_map = {'Male': 'Female', 'Female': 'Male'}
X_test_flipped_sex['sex'] = X_test_flipped_sex['sex'].map(sex_map)

X_test_orig_proc = preprocessor.transform(X_test)
X_test_flip_proc = preprocessor.transform(X_test_flipped_sex)

y_pred_orig = rf_model.predict(X_test_orig_proc)
y_pred_flip = rf_model.predict(X_test_flip_proc)

flip_rate_sex = (y_pred_orig != y_pred_flip).mean()
print(f"    Cinsiyet degistirildiginde tahmin degisim orani: {flip_rate_sex:.1%}")

female_mask = (sex_test.values == 'Female')
female_to_male_flip = (y_pred_flip[female_mask] > y_pred_orig[female_mask]).mean()
male_mask = (sex_test.values == 'Male')
male_to_female_flip = (y_pred_flip[male_mask] < y_pred_orig[male_mask]).mean()

# Gorsellestirme
fig, ax = plt.subplots(figsize=(10, 6))
flip_data = {
    'Genel Tahmin\nDegisimi': flip_rate_sex * 100,
    'Kadin->Erkek\nPozitif Gecis': female_to_male_flip * 100,
    'Erkek->Kadin\nNegatif Gecis': male_to_female_flip * 100
}
bars = ax.bar(flip_data.keys(), flip_data.values(), color=['#9b59b6', '#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_ylabel('Oran (%)')
ax.set_title('Feature Flip: Cinsiyet Degistirme Etkisi', fontsize=13, fontweight='bold')
for bar, val in zip(bars, flip_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '3_feature_flip_analizi.png'), bbox_inches='tight')
plt.close()
print("  [OK] Feature flip grafigi kaydedildi")

# Sonuclari kaydet
xai_results = {
    'lime_sex_weights_mean': np.mean(lime_sex_weights),
    'shap_values': sv,
    'shap_feature_names': feature_names,
    'flip_rate_sex': flip_rate_sex,
    'female_to_male_flip': female_to_male_flip,
    'male_to_female_flip': male_to_female_flip
}
joblib.dump(xai_results, os.path.join(OUTPUT_DIR, 'xai_results.pkl'))
print(f"\n  [OK] XAI sonuclari kaydedildi: outputs/xai_results.pkl")