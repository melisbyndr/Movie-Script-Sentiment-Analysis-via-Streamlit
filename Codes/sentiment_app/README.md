# Sentiment Analysis App

Bu proje, film senaryolarını analiz ederek duygu ve karakter analizi yapan gelişmiş bir Streamlit uygulamasıdır. PDF formatındaki senaryoları işleyerek, diyalogları çıkarır ve çeşitli makine öğrenmesi modelleri kullanarak duygu analizi, karakter analizi ve model karşılaştırması yapar.

## Özellikler

### Ana Özellikler
- **PDF Senaryo İşleme**: PDF formatındaki film senaryolarını otomatik olarak işler
- **Diyalog Çıkarma**: Senaryolardan sahne ve diyalog bilgilerini çıkarır
- **Çoklu Model Desteği**: Farklı HuggingFace modelleri ile analiz yapabilme
- **Sonuç Saklama**: Analiz sonuçlarını kaydedip daha sonra karşılaştırabilme
- **Görselleştirme**: İnteraktif grafikler ve tablolar ile sonuçları görselleştirme

### Analiz Türleri

#### 1. Genel Analiz
- Senaryo istatistikleri (toplam sahne, diyalog, karakter sayısı)
- Karakter bazlı diyalog dağılımı
- Sahne uzunluk analizi
- Kelime sayısı ve ortalama diyalog uzunluğu

#### 2. Duygu Analizi
- **Sentiment Analizi**: Pozitif, negatif, nötr duygu sınıflandırması
- **Emotion Analizi**: 7 temel duygu kategorisi (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Duygu Trendleri**: Zaman içinde duygu değişimlerini analiz etme
- **Bağlam Duyarlı Analiz**: Rolling window ile bağlam duyarlı sentiment analizi
- **Duygu Yoğunluğu**: Duygu skorlarının yoğunluk analizi

#### 3. Karakter Analizi
- **Karakter Arkı**: Karakterlerin duygu gelişimini takip etme
- **Kişilik Profili**: Karakterlerin duygu dağılımına göre kişilik analizi
- **Etkileşim Kalitesi**: Karakterler arası etkileşimlerin analizi
- **Protagonist/Antagonist Belirleme**: Ana karakterleri ve antagonistleri tespit etme

#### 4. Model Karşılaştırması
- **Performans Metrikleri**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Model performansını görsel olarak analiz etme
- **Hata Analizi**: Yanlış tahminlerin detaylı analizi
- **Sonuç İndirme**: Karşılaştırma sonuçlarını CSV formatında indirme

#### 5. Model Yönetimi
- **Çoklu Analiz Saklama**: Farklı modellerle yapılan analizleri kaydetme
- **Analiz Karşılaştırması**: Kaydedilen analizleri karşılaştırma
- **Analiz Yönetimi**: Kaydedilen analizleri silme ve dışa aktarma

## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### Adım Adım Kurulum

1. **Projeyi klonlayın veya indirin**
   ```bash
   git clone <repository-url>
   cd sentiment_app
   ```

2. **Gerekli paketleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **Uygulamayı çalıştırın**
   ```bash
   streamlit run app.py
   ```

4. **Tarayıcınızda açın**
   ```
   http://localhost:8501
   ```

## Kullanım

### İlk Kullanım
1. Ana sayfada "Upload PDF Script" butonuna tıklayın
2. Analiz etmek istediğiniz film senaryosunu PDF formatında yükleyin
3. "Analyze Script" butonuna tıklayarak analizi başlatın

### Sayfa Rehberi

#### 1. Genel Analiz
- Senaryo hakkında genel istatistikleri görüntüler
- Karakter ve sahne dağılımlarını gösterir
- Temel metrikleri sunar

#### 2. Duygu Analizi
- **Model Seçimi**: Sentiment ve emotion modellerini seçin
- **Analiz Çalıştırma**: "Run Analysis" ile analizi başlatın
- **Sonuç Saklama**: Analiz sonuçlarını isimle kaydedin
- **Görselleştirme**: İnteraktif grafiklerle sonuçları inceleyin

#### 3. Karakter Analizi
- Karakterlerin duygu gelişimini takip edin
- Kişilik profillerini analiz edin
- Karakter etkileşimlerini inceleyin

#### 4. Model Karşılaştırması
- Etiketli veri yükleyin (opsiyonel)
- Model performansını karşılaştırın
- Metrikleri ve confusion matrix'i inceleyin

#### 5. Model Yönetimi
- Kaydedilen analizleri görüntüleyin
- Farklı analizleri karşılaştırın
- Analizleri silin veya dışa aktarın

## Desteklenen Modeller

### Sentiment Modelleri
- **DistilBERT (Varsayılan)**: Hızlı ve etkili sentiment analizi
- **BERT Large**: Daha detaylı analiz için
- **RoBERTa Large**: Gelişmiş performans için

### Emotion Modelleri
- **DistilRoBERTa (Varsayılan)**: 7 duygu kategorisi
- **RoBERTa GoEmotions**: Genişletilmiş duygu seti

## Dosya Yapısı

```
sentiment_app/
├── app.py                 # Ana Streamlit uygulaması
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── analysis_cache/       # Analiz sonuçları cache'i
├── data/                 # Örnek veriler
├── pages/                # Streamlit sayfaları
│   ├── 1_General_Analysis.py
│   ├── 2_Sentiment_Analysis.py
│   ├── 3_Character_Analysis.py
│   ├── 4_Model_Comparison.py
│   └── 5_Model_Management.py
└── utils/                # Yardımcı fonksiyonlar
    ├── analyze.py        # Genel analiz fonksiyonları
    ├── extract_data.py   # PDF veri çıkarma
    ├── plot_helpers.py   # Görselleştirme yardımcıları
    └── sentiment_analysis.py # Duygu analizi fonksiyonları
```

## Teknik Detaylar

### Veri İşleme
- PDF'ler PyPDF2 kullanılarak işlenir
- Diyaloglar regex ile çıkarılır
- Karakter ve sahne bilgileri otomatik tespit edilir

### Makine Öğrenmesi
- HuggingFace Transformers kütüphanesi kullanılır
- Pipeline API ile model yükleme ve tahmin
- Batch processing ile performans optimizasyonu

### Görselleştirme
- Plotly ile interaktif grafikler
- Seaborn ve Matplotlib ile statik grafikler
- Streamlit ile kullanıcı dostu arayüz

### Veri Saklama
- Parquet formatında verimli veri saklama
- JSON metadata dosyaları
- Hash tabanlı dosya tanımlama

## Sorun Giderme

### Yaygın Sorunlar

1. **Model yükleme hatası**
   - İnternet bağlantınızı kontrol edin
   - Daha küçük modelleri deneyin

2. **PDF yükleme sorunu**
   - PDF'in metin tabanlı olduğundan emin olun
   - Görüntü tabanlı PDF'ler desteklenmez

3. **Bellek hatası**
   - Daha küçük modelleri kullanın
   - Büyük dosyaları parçalara bölün

### Performans İpuçları
- İlk kullanımda modeller indirilir, sabırlı olun
- Büyük senaryolar için daha güçlü bir bilgisayar gerekebilir
- Cache kullanarak tekrar analiz süresini kısaltın

## Katkıda Bulunma

Bu proje açık kaynak kodludur. Katkılarınızı bekliyoruz:

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız veya önerileriniz için:
- GitHub Issues kullanın
- Pull Request gönderin

## Gelecek Özellikler

- Daha fazla dil desteği
- Gelişmiş karakter analizi
- Senaryo yazım önerileri
- API entegrasyonu
- Mobil uygulama desteği 