# Veri seti ve model lisansları

Kontrol tarihi: 18 Kasım 2026. Kaynak: veri setlerinin kendi sayfaları (GitHub README, Zenodo kaydı, makale).
Hukuki görüş değildir; ticari ürüne geçmeden önce her satır için yazarlardan yazılı teyit alınmalı.

## Kavga / şiddet videoları

| Veri seti | İçerik | Lisans / koşul | Ticari kullanım | Bu prototipte |
|---|---|---|---|---|
| **RWF-2000** (Cheng ve ark., ICPR 2021) | 2.000 CCTV klibi, 5 sn, 30 fps (YouTube kaynaklı) | README: SMIIP Lab onayı olmadan **değiştirilemez, yeniden dağıtılamaz, ticari amaçla kullanılamaz**. Videolar "gizlilik nedeniyle" sitede yok; erişim laboratuvardan istenir. | **Hayır** (izinsiz) | **Kullanılmadı.** Kaggle / Hugging Face aynaları izinsiz yeniden dağıtım sayılır; onlardan da indirilmedi. |
| **Hockey Fight** (Nievas ve ark., 2011) | 1.000 NHL maç klibi, ~2 sn | Açık lisans yok. Makale: veri "yazarlardan istek üzerine". Görüntüler NHL yayınından, telif yayıncıda. Academic Torrents aynasında lisans alanı boş. | **Hayır / belirsiz** | **Kullanılmadı.** Ayrıca hepsi buz hokeyi; hastane/okul ortamına benzemiyor. |
| **Bus Violence** (Ciampi ve ark., Sensors 2022; Zenodo 7044203) | 1.400 klip (700 şiddet / 700 normal), otobüs içi kameralar, oyuncularla sahnelenmiş, 1–1,6 sn | Zenodo lisans alanı **CC BY 4.0**; ama kaydın açıklaması "ticari olmayan kullanım için Creative Commons Atıf" diyor. **Çelişkili.** | **Belirsiz** → yazarlardan teyit | Kullanıldı: resmî train/test bölmesi (eğitimin %15'i doğrulama). |
| **AIRTLab Violence** (Bianculli ve ark., Data in Brief 2020) | 350 klip, 2 kamera, oyuncular; normal kliplerde sarılma, çak, alkış, sevinç, el kol hareketi | README: "**araştırma ve eğitim amaçlı** serbest", atıf istenir. Depoda lisans dosyası yok. | **Hayır** (yalnız araştırma/eğitim) | Kullanıldı: kavga + zor olumsuz (sarılma/alkış). Aynı olayın iki kamerası aynı bölümde. |
| **Surveillance Camera Fight** (Aktı ve ark., IPTA 2019) | 300 klip (150/150), 2 sn, YouTube güvenlik kamerası kesitleri | Depo **MIT**; ama videolar üçüncü kişilerin YouTube yüklemelerinden kesilmiş (`videos.txt` kaynakları listeler). MIT yalnız depo sahibinin hakları için geçerli; görüntülerin telifi yükleyicilerde. | **Riskli** | Kullanıldı (araştırma). Aynı YouTube kaynağından gelen klipler aynı bölümde. |

## Zor olumsuz (kavga olmayan) videolar

| Veri | İçerik | Lisans | Ticari | Bu prototipte |
|---|---|---|---|---|
| **GMDCSA-24** (Alam ve ark., Data in Brief 2024; Zenodo 12921216) | 81 düşme + 79 günlük hareket klibi, 4 denek, ev ortamı | **CC BY 4.0** | **Evet** (atıfla) | Kullanıldı: "hasta düşmesi" ve "günlük hareket" olumsuzları. Denek 1–2 eğitim, 3 doğrulama, 4 test. |
| **Wikimedia Commons** videoları | Okul bahçesinde çocuklar, spor, kalabalık AVM/hastane | Dosya başına CC BY / CC BY-SA / kamu malı (`araclar/commons_indir.py` her dosyanın lisansını ve yazarını kaydeder) | Çoğu evet (atıf; BY-SA'da aynı lisans) | Bkz. README "Açık riskler": indirme sırasında Wikimedia 429 (çok istek) engeli. |

| **Yapay zekâ ile üretilmiş klipler** (Higgsfield, Seedance 2.5) | 8 zor olumsuz sahne (çocuk, spor, kucaklaşma, kalabalık, hasta düşmesi) + 2 bayılma sahnesi | Bizim ürettiğimiz; Higgsfield kullanım koşullarına tabi | Evet (Higgsfield koşullarıyla) | Yalnız ölçüm; eğitime girmedi. Depoda yalnız bir klibin pikselli çıktısı var (`docs/`). |

## Ses testi

| Veri | İçerik | Lisans | Ticari | Bu prototipte |
|---|---|---|---|---|
| **ESC-50** (Piczak 2015) | 2.000 çevre sesi, 50 sınıf, 5 sn (Freesound kaynaklı) | Bütünü **CC BY-NC 3.0**; ESC-10 alt kümesi CC BY 3.0 | **Hayır** (bütünü) | Yalnız ses olayı algılayıcısının ölçümü ve gündelik ses bindirme senaryosu. Eğitimde kullanılmadı. |

## Modeller

| Model | Görev | Lisans | Ticari |
|---|---|---|---|
| **YOLO11m-pose** (Ultralytics) + ByteTrack (Ultralytics içi) | Kişi, iskelet, takip | **AGPL-3.0**; kapalı kaynak ticari ürün için Ultralytics Enterprise lisansı gerekir | AGPL koşullarıyla (Kafe Kamera ile aynı durum) |
| **PANNs Cnn14** (Kong ve ark. 2020; ağırlık Zenodo 3987831) | Ses olayı (AudioSet 527 sınıf) | Ağırlıklar **CC BY 4.0**; özgün kod MIT (`siddet/ses.py` içindeki Cnn14 tanımı o koda dayanır) | Evet (atıfla). Not: AudioSet etiketleri CC BY 4.0, ama eğitildiği sesler YouTube'dan. |
| **İskelet kavga sınıflandırıcısı** (`modeller/iskelet_kavga.pt`) | Kavga / normal | Bizim eğitimimiz; ama **yukarıdaki araştırma lisanslı verilerle eğitildi** | **Hayır** — ürün için lisansı temiz veriyle yeniden eğitilmeli |

## Sonuç

- Bu prototipin kavga modeli **yalnız araştırma / iç değerlendirme** içindir. Eğitimde kullanılan kavga
  verilerinin hiçbiri temiz bir ticari lisans vermiyor (Bus Violence belirsiz, AIRTLab araştırma, Surveillance Fight YouTube).
- Ürün yolu: (1) kendi sahnelenmiş çekimlerimiz (izinli oyuncular, okul/hastane benzeri mekân, birden çok kamera açısı),
  (2) Bus Violence için yazarlardan ticari kullanım teyidi, (3) RWF-2000 için SMIIP Lab'e ticari izin başvurusu,
  (4) gerekirse yapay zekâ ile üretilmiş klipler (üretim öncesi çekim listesi + maliyet onayı ile).
- Ses tarafı (PANNs, CC BY 4.0) ve düşme verisi (GMDCSA-24, CC BY 4.0) ticari kullanıma uygundur (atıfla).
