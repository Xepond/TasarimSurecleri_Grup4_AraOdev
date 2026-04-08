"""
XAI ile Önyargı Tespiti Projesi
Adım 1: Veri Yükleme, Keşifsel Analiz ve Ön İşleme

Veri Seti: Adult Census Income (UCI)
Hassas Değişkenler: sex (cinsiyet), race (ırk)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Yollar ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style='whitegrid', palette='muted')

NUMERIC_FEATURES = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_FEATURES = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

print("=" * 60)
print("  ADIM 1: VERI YUKLEME VE KESIFSEL ANALIZ")
print("=" * 60)

# ============================================================
# 1. VERI SETINI YUKLE
# ============================================================
print("\n[1/5] Veri seti yukleniyor (Adult Census Income)...")
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.data.copy()

# Hedef degiskeni temizle (bazi versiyonlarda '.' olabiliyor)
target = adult.target.str.strip().str.rstrip('.')
df['income'] = target

print(f"  Veri seti boyutu: {df.shape[0]} satir, {df.shape[1]} sutun")
print(f"  Hedef degisken dagilimi:")
print(df['income'].value_counts().to_string())

# ============================================================
# 2. VERI SETINI INCELE
# ============================================================
print("\n[2/5] Veri seti inceleniyor...")
print(f"\n  Eksik veri sayilari:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing.to_string())
else:
    print("  Eksik veri yok.")

print(f"\n  Cinsiyet dagilimi: {dict(df['sex'].value_counts())}")
print(f"  Irk dagilimi: {dict(df['race'].value_counts())}")

# ============================================================
# 3. KESIFSEL VERI ANALIZI (EDA)
# ============================================================
print("\n[3/5] Kesifsel veri analizi yapiliyor...")

# 3.1 Cinsiyete ve irka gore gelir dagilimi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ct_sex = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
ct_sex.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('Cinsiyete Gore Gelir Dagilimi (%)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cinsiyet')
axes[0].set_ylabel('Yuzde (%)')
axes[0].legend(['<=50K', '>50K'])
axes[0].tick_params(axis='x', rotation=0)

ct_race = pd.crosstab(df['race'], df['income'], normalize='index') * 100
ct_race.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[1].set_title('Irka Gore Gelir Dagilimi (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Irk')
axes[1].set_ylabel('Yuzde (%)')
axes[1].legend(['<=50K', '>50K'])
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '1_gelir_dagilimi_demografik.png'), bbox_inches='tight')
plt.close()
print("  [OK] Demografik gelir dagilimi grafigi kaydedildi")

# 3.2 Sayisal ozelliklerin dagilimi
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(NUMERIC_FEATURES):
    ax = axes[i // 3, i % 3]
    df[col].hist(bins=30, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col} Dagilimi', fontsize=12, fontweight='bold')
plt.suptitle('Sayisal Ozelliklerin Dagilimlari', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '1_sayisal_ozellik_dagilimi.png'), bbox_inches='tight')
plt.close()
print("  [OK] Sayisal ozellik dagilim grafigi kaydedildi")

# 3.3 Korelasyon matrisi
fig, ax = plt.subplots(figsize=(10, 8))
df_corr = df[NUMERIC_FEATURES].copy()
df_corr['income_binary'] = (df['income'] == '>50K').astype(int)
sns.heatmap(df_corr.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax, linewidths=0.5)
ax.set_title('Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, '1_korelasyon_matrisi.png'), bbox_inches='tight')
plt.close()
print("  [OK] Korelasyon matrisi grafigi kaydedildi")

# ============================================================
# 4. ON ISLEME
# ============================================================
print("\n[4/5] Veri on isleme yapiliyor...")

y = (df['income'] == '>50K').astype(int)
X = df.drop(columns=['income'])

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hassas ozellikleri ayri sakla (fairness analizi icin)
sex_train, sex_test = X_train['sex'].copy(), X_test['sex'].copy()
race_train, race_test = X_train['race'].copy(), X_test['race'].copy()

print(f"  Egitim seti: {X_train.shape[0]} ornek")
print(f"  Test seti:   {X_test.shape[0]} ornek")

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = list(preprocessor.get_feature_names_out())
print(f"  Islenmis ozellik sayisi: {len(feature_names)}")

# ============================================================
# 5. VERILERI KAYDET
# ============================================================
print("\n[5/5] Islenmis veriler kaydediliyor...")
data_to_save = {
    'X_train': X_train, 'X_test': X_test,
    'X_train_processed': X_train_processed, 'X_test_processed': X_test_processed,
    'y_train': y_train, 'y_test': y_test,
    'sex_train': sex_train, 'sex_test': sex_test,
    'race_train': race_train, 'race_test': race_test,
    'feature_names': feature_names,
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'preprocessor': preprocessor
}
joblib.dump(data_to_save, os.path.join(OUTPUT_DIR, 'processed_data.pkl'))
print(f"  [OK] Veriler kaydedildi: outputs/processed_data.pkl")

print("\n" + "=" * 60)
print("  ADIM 1 TAMAMLANDI!")
print("  Sonraki adim: python 2_model_egitimi.py")
print("=" * 60)
