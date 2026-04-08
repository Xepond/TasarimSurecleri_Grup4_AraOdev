# Açıklanabilir Yapay Zeka (XAI) ile Algoritmik Adalet ve Önyargı Azaltma

Bu proje, makine öğrenmesi sistemlerinde oluşabilecek demografik (cinsiyet ve ırk temelli) önyargıları **(Bias)** tespit etmeyi ve gidermeyi amaçlamaktadır. Projede **Adult Census Income** veri seti kullanılarak eğitilmiş bir Random Forest modeli üzerinden adalet değerlendirmeleri yapılmış; LIME, SHAP ve Karşıolgusal (Counterfactual) analiz teknikleriyle şeffaflık sağlanmıştır. Geliştirilen Post-Processing mimarisi sayesinde modeldeki ayrımcılık giderilmiş ve gerçek bir algoritmik adalet elde edilmiştir.

## 🚀 Proje Kapsamı ve Özellikler

* **Makine Öğrenmesi Modelleri:** Logistic Regression, Random Forest, XGBoost algoritmalarının eğitimi ve adalet metrikleri üzerinden kıyaslanması.
* **XAI Yöntemleri:** 
  * **LIME & SHAP:** Siyah kutu modellerin arkasındaki asıl dinamiklerin çözülmesi; evlilik veya çalışma saatleri gibi değişkenlerin "Proxy (Vekil) Değişken" görevi üstlendiğinin ispatı.
  * **Counterfactual Explanations:** Modelin tersine karar vermesi için gerekli şartların "Feature Flip" mekanizmasıyla sınır testi.
* **Bias Mitigation (Önyargı Azaltma):** Cinsiyet özelliklerinin basitçe silinmesinin önyargıyı engelleyemediğinin ispatlanması ve ardından `Fairlearn - ThresholdOptimizer` metoduyla asıl adaletin (**Demographic Parity**) sağlanması.

## 📂 Dosya Yapısı

* **`1_veri_hazirlama.py`**: Adult Census Income veri setinin çekilmesi, EDA (Keşifsel Veri Analizi) ve Feature Engineering işlemleri.
* **`2_model_egitimi.py`**: Makine öğrenmesi algoritmalarının eğitilmesi, Accuracy ve Fairness hesaplamaları.
* **`3_xai_analiz.py`**: SHAP, LIME değer atamaları ve Counterfactual analiz algoritmaları.
* **`4_sonuclar.py`**: Modelin ThresholdOptimizer uygulanarak yeniden kalibre edilmesi ve adalet kıyaslamalarının son raporu.
* **`outputs/`**: Analiz esnasında üretilen .pkl ağırlıkları ve 16 adet veri görselleştirmesi.

## ⚙️ Kurulum ve Çalıştırma

Projenin test edilmesi için modern bir Python bariyeri (3.9+) yeterlidir.

1. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Sırayla Çalıştırın:** Projenin her sekmesi bir sonrakine ağırlık pasladığı için (pipeline mantığı) adımlar sırasıyla yorumlanmalıdır.
   ```bash
   python 1_veri_hazirlama.py
   python 2_model_egitimi.py
   python 3_xai_analiz.py
   python 4_sonuclar.py
   ```

## 📊 Kilit Sonuçlar

| Metrik | Orijinal RF Modeli | Optimize Edilmiş RF Modeli | Durum Özeti |
|---|---|---|---|
| **F1-Score** | 0.6710 | 0.6066 | Adalet için mecburi, kabul edilebilir fedakarlık (Trade-off) |
| **Demographic Parity Hata Oranı** | 0.1563 | 0.0051 | ~0 Oranı ile Kusursuz Dağılım |
| **Disparate Impact Ratio (%80 Kuralı)**| 0.3054 ❌| **Geçerli** ✅ | Sistematik Ayrımcılık Bertaraf Edildi |

*SHAP analizleri açıkça göstermiştir ki, `relationship_Husband` gibi değişkenler cinsiyet bazındaki önyargıyı tamamen proxy üzerinden sürdürebilmektedir. Yalnızca salt kolon silimi ("Kör Yaklaşım") değil; esnek eşik sınırlandırmaları gereklidir.*

## 📜 Lisans & Teşekkür
Tasarım Süreçleri projeksiyonu kapsamında akademik olarak modellenmiştir. Veri Seti [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)'den açık kaynak olarak ithal edilmiştir.
