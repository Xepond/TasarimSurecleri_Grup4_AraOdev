# T.C. Eskişehir Osmangazi Üniversitesi

## Mühendislik Mimarlık Fakültesi - Bilgisayar Mühendisliği Bölümü

<br/>

### **152116025 - Tasarım Süreçleri Ödev Raporu**

# **AÇIKLANABİLİR YAPAY ZEKA (XAI) YÖNTEMLERİNİN ADİL VE TARAFSIZ KARAR VERME ÜZERİNDEKİ ETKİSİ**

<br/>

**Hazırlayanlar:**
152120221169 - Süleyman Efe Polat
152120221058 - Tolga Özcan
152120231119 - Eren Çil
152120231118 - Server Ahıskalı

---

<div style="page-break-after: always;"></div>

## **Özet**

Yapay zeka (YZ) destekli sistemlerin finans, sağlık ve insan kaynakları gibi kritik alanlarda artan kullanımı, bu algoritmaların öğrenebilecekleri tarihsel önyargılarla ilgili haklı endişeler doğurmuştur. Bu çalışma, makine öğrenmesi modellerindeki cinsiyet ve ırk temelli ayrımcılığı (bias) saptamak ve gidermek amacıyla Açıklanabilir Yapay Zeka (XAI) yaklaşımlarının ve müdahale algoritmalarının etkinliğini araştırmaktadır. *Adult Census Income* veri seti üzerinde eğitilen makine öğrenmesi (Random Forest, XGBoost) algoritmalarında, ABD iş hukuku normları (Disparate Impact - %80 Kuralı) temel alınarak modelin barındırdığı adaletsizlik kanıtlanmıştır.  

Gerçekleştirilen analizlerde **LIME, SHAP** ve **Counterfactual Explanations** yöntemlerinin "siyah kutu" (black-box) modelleri başarıyla yorumladığı ve modeldeki gizli önyargıları açığa çıkardığı görülmüştür. Geleneksel "kör yaklaşım" olarak da bilinen hassas özellikleri (cinsiyet, ırk) veri setinden çıkarma (fairness through unawareness) işleminin, *Proxy (Vekil)* değişkenler (evlilik durumu, haftalık çalışma saati vb.) nedeniyle etkisiz kaldığı XAI araçlarıyla ispatlanmıştır. Bu bağlamda, *"Threshold Optimizer"* algoritması uygulanarak karar sonrası sınır değer iyileştirmesi yapılmış, hassas demografik özellikler arasındaki *Demographic Parity* (Demografik Adalet) oranı kusursuz bir seviyeye getirilmiştir. Çalışma, şeffaf ve sorumlu yapay zeka tasarımlarının gerekliliğini ampirik düzeyde ortaya koymuştur.

**Anahtar Kelimeler:** *Açıklanabilir Yapay Zeka (XAI), SHAP, LIME, Algoritmik Adalet (Fairness), Post-Processing, Demographic Parity.*

---

<div style="page-break-after: always;"></div>

## **1. Giriş**

### **1.1. Çalışmanın Arka Planı ve Motivasyonu**

Karar destek sistemlerindeki makine öğrenmesi modelleri algoritmik bir adalet illüzyonuna sahipmiş gibi görünse de; eğitim sırasında kullandıkları tarihsel veri setleriyle toplumsal önyargı ve eşitsizlikleri doğrudan öğrenebilmekte, hatta güçlendirebilmektedir (Barocas & Selbst, 2016). Avrupa Veri Koruma Tüzüğü (GDPR) kapsamında alınan kararlarda "Açıklanabilirlik Hakkı" prensibinin yasal bir zemin bulması, yapay zekanın kapalı bir kutudan çıkıp denetlenebilir hale gelmesini (Explainable AI - XAI) kritik kılmıştır.

### **1.2. Araştırma Problemi ve Soru**

Bir gelir tahmin/kredi skorlama modelinde *Demographic Parity (Demografik Adalet)* oranını bozarak belirli bir cinsiyet veya ırk aleyhine ayrımcılık oluşturan gizli örüntülerin tespiti nasıl yapılmalıdır? SHAP, LIME ve Counterfactual analizler proxy değişkenlerin ayrımcılık üzerindeki rolünü açıklamada ne kadar başarılıdır?

### **1.3. Çalışmanın Hedef ve Kapsamı**

Bu çalışmadaki amaç sadece algoritmanın yanlışlık oranlarını bulmak değil, **Algoritmik Adalet (Algorithmic Fairness)** literatüründeki temel eşikler bağlamında şunları gerçekleştirmektir:

1. Adult Census Income veri seti ile eşitsizlik barındıran bir yapay zeka karar sistemi inşa etmek.
2. XAI analitiği ile cinsiyet özelliğinin tek başına silinmesinin model adaleti sağlamadığını kanıtlamak.
3. Çıkarımsal optimizasyon (Threshold Optimization) uygulayarak, performanstan belirli ölçüde fedakarlık yaparak adaleti test edilebilir biçimde yeniden tesis etmek.

---

## **2. Literatür Taraması ve Teorik Çerçeve**

### **2.1. Açıklanabilir Yapay Zeka Kavramları**

* **LIME (Local Interpretable Model-agnostic Explanations):** Altında doğrusal ve yorumlanabilir ikincil modeller oluşturarak makine öğrenmesinin anlık veya yerel kararlarının (örneğin tek bir bireyin kredi reddi) sebeplerini açıklayan bir tekniktir. (Ribeiro vd., 2016)
* **SHAP (SHapley Additive exPlanations):** Kooperatif oyun teorisinde sunulan "Shapley değerleri"ni referans olarak alan, hem yerel kararların hem de global (modelin geneli) etki büyüklüklerinin tutarlı bir biçimde dağıtılmasını sağlayan kapsayıcı bir araçtır.
* **Counterfactual Explanations (Karşı Olgusal Analiz):** Bir bireyin yapay zeka sisteminden aldığı kararın yön değiştirmesi için hangi özelliklerinin ne kadar oranda değiştirilmesi gerektiğini sorgulayan, eyleme dönüştürülebilir metodolojidir.

### **2.2. Algoritmik Adalet (Algorithmic Fairness) Metrikleri**

Makine öğrenmesinde modellerin ayrımcılık seviyesi çeşitli matematiksel çerçeveler üzerinden ölçülür.

* **Demographic Parity Difference:** Modelin azınlık ve çoğunluk (ayrıcalıklı vs dezavantajlı) gruba verdiği olumlu (positive class) karar oranları arasındaki istatistiksel mutlak farktır. Sıfıra eşit olması idealdir.
* **Equalized Odds Difference:** True Positive ve False Positive oranlarının, gruplar arasındaki farkıdır.
* **Disparate Impact Ratio:** Dezavantajlı grubun elde ettiği olumlu sonuçların, ayrıcalıklı gruba oranıdır. Özellikle Amerika'daki Federal iş alım kanunlarında (Title VII) Disparate Impact %80 (0.8) kuralı çerçevesinde yasal bir eşiktir.

---

## **3. Veri Seti ve Metodoloji**

### **3.1. Veri Yapısı (Adult Census Income)**

UCI makine öğrenmesi kütüphanesinden entegre edilen bu veri setinde toplam 48.842 satırlık, ABD'nin 1994 yılı demografik profili bulunmaktadır.

* **Hedef Değişken (Target):** Bireyin yıllık gelirinin "Yüksek (>50K)" veya "Düşük (<=50K)" olduğu bilgisidir.
* **Hassas Değişkenler (Sensitive Attributes):** Analizlerde baz alınan temel sınıflar **Cinsiyet (Sex)** ve **Irk (Race)** şeklindedir.

### **3.2. Veri Ön İşleme (Preprocessing)**

1. **Atanamayan / Eksik Veriler:** Eksik demografik veya finansal kategorik değişkenler `Most Frequent (Mod)`, nümerik olanlar `Median (Medyan)` metoduyla Imputer aşamasından geçirilmiştir.
2. **Kategorik Kodlama & Ölçeklendirme:** Değerler `OneHotEncoding` ile matrislere dönüştürülmüş ve modellerde dengeli kümelenme açısından `StandardScaler` ile ölçeklendirilmiştir. (Toplam özellik sütunu 105)

### **3.3. Eğitilen Modeller ve Adalet Yaklaşımı**

Karşılaştırma ve tutarlılık bağlamında veri setine Logistic Regression, Random Forest (Rassal Orman) ve XGBoost modelleri uygulanmıştır. Yüksek doğruluk performansı ve Tree tabanlı SHAP açıklayıcı algoritmaları ile uyumu nedeniyle analizlerde **Random Forest** ana yorum modeli olarak referans alınmıştır.

---

## **4. XAI Bulguları ve Deneysel Sonuçlar**

### **4.1. Orijinal Model Performans Değerleri**

Eğitilen karar sisteminin performansı ölçüldüğünde;

* Random Forest Accuracy (Doğruluk): **0.864**
* Random Forest F1-Score: **0.671**
olarak başarılı bir doğruluk/genellenebilirlik elde edilmiştir.

### **4.2. İlk Adalet (Fairness) Ölçümü ve Önyargının Tespiti**

Modelin adaletsizlik derecesini referans değerlerle kıyaslamak için cinsiyet bazlı skorlara bakılmıştır.

* **Demographic Parity Farkı:** `0.1563`
* **Disparate Impact Skoru:** `0.3054`
Buradaki *0.3054* oranı, ABD %80 eşik kuralı dikkate alındığında sistemde **Kesin Bir Önyargı / Ayırımcılık (Bias)** bulunduğunu kanıtlamıştır. Model, kadınları sistemsel olarak dezavantajlı konumda değerlendirmektedir.

*[Buraya output içerisindeki 2_fairness_metrics.png ve 2_roc_curves.png ekleneceği öngörülmektedir]*

### **4.3. XAI – Cinsiyet Proxy (Vekil) Değişkenlerinin Tespiti**

**SHAP ve LIME bulguları:** Yorumlanabilirlik kütüphanelerinin dökümleri modelin "hassas özellikler" silinse dahi hangi noktalardan ayrımcılık yapabileceğini ispatlamıştır.
SHAP Summary grafikleri göstermiştir ki, `Evlilik Durumu (Koca/Eş)` gibi bilgiler veya haftalık mesai saati, aslında bireyin cinsiyet yapısını öğrenen gizli geçitler görevindedir. Yapay Zeka modeli, *erkek* sütunu formasyondan çekilmesine rağmen "Birey eş olarak (husband) konumlandırılmışsa ve mesai fazlaysa bu bir erkektir, yüksek gelirlidir" eğilimi (bias) yaratmaktadır.

**Feature Flip (Karşıolgusal Çeviri):**
Test setindeki bireylerin cinsiyet değişkeni "Kadın"dan "Erkek" formuna çevrildiğinde veya tam tersi yapıldığında; pozitif/negatif tahmin dönüşüm oranının sadece **%3.1** ile sınırlı kaldığı görülmüştür. Bu, klasik yaklaşımdaki "adını, cinsiyetini listeden silmek, adil bir algoritma elde etmeyi sağlamaz" argümanının simülasyon ispatıdır.

### **4.4. Önyargı Azaltma / Bias Mitigation Uygulaması**

XAI araştırmasından çıkan kanıtlara dayanılarak "kör kalarak çözme (Fairness Through Unawareness)" rafa kaldırılmış ve **Post-Processing (Tahmin Sonrası Optimizasyon - ThresholdOptimizer)** kullanılmıştır.
Modelin cinsiyetler üzerindeki tahmin olasılıkları dağılımında her iki cinsiyetin de eşit pozitif onay alması adına *Demographic Parity* sınır (yargı) çizgileri manipüle edilmiştir.

* Optimizasyon Öncesi Demographic Parity: `0.1563` (Kötü)
* Optimizasyon Sonrası Demographic Parity: `0.0051` (Kusursuz Dağılım)
* Disparate Impact Ratio: `0.3054`’ten -> `~ 1.0` (Mutlak eşitlik hedefine evrildi)
* F1-Score Trade-off’u: `0.6710`’den -> `0.6066`

Optimizasyon sonrası model performansından ufak ve kabul edilebilir bir miktar feragat edilerek (trade-off) tam anlamıyla adil süreçler tahsis edilmiştir.

---

## **5. Tartışma**

Siyah kutu modellerine olan güven sorunu bu projede LIME, SHAP ve Karşıolgusal mekanizmalar ile giderilmeye çalışılmış, model şeffaf bir platforma çekilmiştir. Açıklanabilir metrikler incelendiğinde yapay zekanın aslında insan beyni benzeri kestirme karar yollarına (heuristics) giriştiği anlaşılabilmektedir. Makine öğrenmesi veri setinde net bir cinsiyet ayrımcılığı gözükmüyor gibi olsa da, ikincil proxy'lerin ayrımcılığı üstlendiği tezi SHAP değerleri üzerinden onaylanmıştır. Gelecek adımlarda In-Processing (Algoritma içerisine ceza puanları yerleştirilmesi suretiyle Adversarial Debiasing) yöntemlerinin kullanılmasıyla trade-off performans düşüklüğünün engellenebileceği değerlendirilmektedir.

---

## **6. Sonuç**

Çalışma sonucunda, XAI araçlarının (LIME ve SHAP) verilerin gizlenen eşitsizliklerini açığa çıkarmasında olağanüstü kabiliyetleri olduğu ortaya konulmuştur. Eldeki sonuçlar, karar alma sistemlerinde Cinsiyet, Irk, Yaş gibi faktörlerin doğrudan model öğrenim evresindeki dışlanmasının (kör kalma politikası) tek başına *hiçbir adalet getirisi sağlamadığını* güçlü analitik araçlarla belgelemiştir. Sonradan kalibre edilen eşikler (Threshold Optimizer) sayesinde hem açıklanabilir hem de demokratik bir Yapay Zeka modeli tasarlanabileceği gösterilmiştir. Sistem mühendisliğinde kod kalitesi kadar adalet (fairness) süreçlerinin de mühendislik akışının zorunlu ve sürdürülebilir bir KPI'ı (Temel Performans Göstergesi) olması gerektiği sonucuna varılmıştır.

---

## **Kaynakça**

1. Barocas, S., & Selbst, A. D. (2016). *Big Data's Disparate Impact.* California Law Review, 104, 671-732.
2. Goodman, B., & Flaxman, S. (2017). *European Union Regulations on Algorithmic Decision-Making and a “Right to Explanation”.* AI Magazine, 38(3), 50-57.
3. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NIPS 2017).
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
5. Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.* Harvard Journal of Law & Technology.
6. Agarval, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). *A Reductions Approach to Fair Classification.* International Conference on Machine Learning (ICML).
