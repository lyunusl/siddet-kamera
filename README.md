# Şiddet Kamera (prototip)

Canlı kullanmak için → [surv.augelab.com](https://surv.augelab.com)

Hastane, okul gibi yerlerde güvenlik kamerası görüntüsü ve sesinden **kavga / şiddet olayı** şüphesini ve
**düşme, bayılma şüphesi, hareketsiz yatma** olaylarını algılayan araştırma prototipi. Bir video dosyasından
alarm zaman çizelgesi üretir.

- **Duygu tanıma yapmaz.** Kimsenin yüz ifadesine, ses tonuna ya da duygusuna bakılmaz (AB Yapay Zekâ Yasası
  okul ve işyerinde duygu tanımayı yasaklar). Yalnız **olay** algılanır: kavga hareketi (iskelet), çığlık/bağırma,
  cam kırılması, darbe sesi.
- **Çıktıda yüzler gizlenir.** İşaretli videoda (`--video-yaz`, isteğe bağlı) bütün sahne kaba piksele indirilir,
  her kişinin başı ve YuNet'in bulduğu yüzler ayrıca pikselleştirilir; iskelet bunun üstüne çizilir. Pikselleme
  tanınmayı zorlaştırır, anonimlik garantisi değildir (kıyafet, boy, yer, zaman yine bilgi taşır). Analiz sırasında
  ham görüntü bilgisayarın belleğinde işlenir.
- **Ses kaydedilmez.** Ses ffmpeg'den doğrudan belleğe akar, diske yazılmaz, çıktı videosu sessizdir. Sesten geriye
  yalnız olay puanları kalır. `zaman_cizelgesi.json` ve `.html` ayrıca video dosyasının adını ve 0,5 sn'lik
  görüntü/ses puan serisini içerir (görüntü ve ses içermez).
- **Ticari ürün değildir.** Kavga modeli araştırma lisanslı verilerle eğitildi, bkz. [LISANSLAR.md](LISANSLAR.md).

## Nasıl çalışır?

```
video ──► YOLO11m-pose + ByteTrack (15 kare/sn) ──► 1,6 sn'lik iskelet pencereleri (0,5 sn adım)
                                                    └─► iskelet kavga sınıflandırıcısı ──► görüntü puanı
ses (yalnız bellekte) ──► PANNs Cnn14 (AudioSet, eğitimsiz) ──► çığlık / cam kırılması / darbe puanı
                                         görüntü puanı + ses puanı ──► birleştirme kuralı ──► ALARM
```

1. **Görüntü:** her pencerede en çok 4 kişi seçilir (en çok hareket eden ve birbirine en yakın olanlar). Her
   kişi için kendi boyuna göre normalleştirilmiş poz, eklem hızları, gruba göre konum ve en yakın kişiye uzaklık
   çıkarılır. Küçük bir ağ (kişi kodlayıcı → kişiler üstünde havuzlama → zamansal evrişim → çift yönlü GRU,
   144 bin parametre) "kavga / normal" puanı verir. 3 farklı tohumla eğitilen modelin ortalaması kullanılır.
   Pencerede 2'den az kişi varsa puan 0,2 ile çarpılır (alarm hattında; klip düzeyi ölçümde bu kural yok).
2. **Ses:** hazır PANNs Cnn14 her 0,5 sn'de 2 sn'lik sesi 527 AudioSet sınıfına göre puanlar; yalnız üç olay
   grubu kullanılır: *çığlık/bağırma* (Screaming, Yell, Shout, Battle cry), *cam kırılması* (Shatter, Glass,
   Breaking, Smash), *darbe* (Slap, Whack, Thump, Bang).
3. **Birleştirme** (`siddet/birlestir.py`), alarm iki yoldan çıkar:
   - **Görüntü + ses:** görüntü puanı ≥ 0,75 **ve** ±2 sn içinde ses olayı ≥ 0,30 (en az 0,5 sn)
   - **Yalnız görüntü:** görüntü puanı ≥ 0,95, en az 1,5 sn kesintisiz

   10 sn içindeki koşullar tek alarmda birleşir.
4. **Düşme, bayılma, hareketsizlik** (`siddet/dusme.py`, kural tabanlı, eğitimsiz): her kişi için gövde açısı
   (omuz–kalça doğrusu), kutu oranı, kalça yüksekliği ve 1 sn'lik eklem hareketi izlenir.
   - **Düşme:** kişi yatay hale geçer (gövde açısı > 60° ya da kutu yatay) ve 0,6 sn içinde kalçası vücut
     uzunluğunun en az %25'i kadar iner; düşmeden önceki 2 sn'de dik görülmüş olmalı (görülmediyse, ör. yataktan
     yuvarlanma, %45 iniş gerekir).
   - **Bayılma şüphesi:** düşmeden sonra 10 sn yerde hareketsiz (`--bayilma-sn`).
   - **Hareketsiz yatıyor:** düşme görülmese de 30 sn yatay ve hareketsiz (`--hareketsiz-sn`); yatak/sedye
     bölgelerinde (`--yatak bolgeler.json`) üretilmez.
   - Düşme anına denk gelen kavga alarmı silinmez ama "düşmeyle çakışıyor" diye işaretlenir (yere düşen kişiye
     yardım eden biri de "temas + ani hareket" gibi görünür).

## Kurulum

Python 3.12, Windows ya da Linux; NVIDIA GPU önerilir (işlemcide de çalışır, yavaş).

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python araclar/modelleri_indir.py
```

`ffmpeg` PATH'te olmalı. `modelleri_indir.py` poz, ses ve yüz modellerini resmî kaynaklarından indirip özetlerini
doğrular. **İskelet kavga sınıflandırıcısının ağırlıkları depoda yok:** araştırma lisanslı verilerle eğitildiği
için dağıtılmıyor (bkz. [LISANSLAR.md](LISANSLAR.md)); veri setlerini kendi kaynaklarından indirip aşağıdaki
"Yeniden üretmek" adımlarıyla eğitin. Düşme / bayılma modülü eğitim gerektirmez.

## Çalıştırma

Kavga algılama için önce `modeller/iskelet_kavga.pt` eğitilmiş olmalı ("Yeniden üretmek"). Bu dosya yoksa analiz
yine çalışır, ama yalnız düşme / bayılma / hareketsizlik üretir (ekrana uyarı yazar).

```
.venv\Scripts\python -m siddet.analiz VIDEO.mp4 --video-yaz
```

Çıktı `cikti/<video adı>/`: `zaman_cizelgesi.json` (alarmlar, ses olayları, 0,5 sn'lik puanlar),
`zaman_cizelgesi.html` (grafik + alarm tablosu), `isaretli.mp4` (gizlilikli, sessiz).

**Demo:** `ornekler/demo_test_karisik.mp4` (83 sn), eğitimde hiç görülmemiş test kliplerinden kuruldu:
normal sokak → sokak kavgası → sarılma/çak → sahnelenmiş kavga → iki hasta düşmesi (GMDCSA-24) → otobüs normal →
otobüs kavgası → günlük hareket → yapay zekâ ile üretilmiş hasta düşmesi. Bölümler arasında 1,5 sn siyah ara var. Demo videosu veri setlerinin
ham görüntüsünden kurulduğu için (yüzler açık, araştırma lisanslı) depoda yok; `araclar/demo_videosu.py` veri
setleri indirildikten sonra aynı videoyu kurar.

```
demo_test_karisik.mp4: 83.4 sn, 3 alarm, analiz 37.5 sn (RTX 5070 Ti, model yükleme dahil)
  ALARM 31.3 sn  yalnız görüntü  1.00  ← AIRTLab kavgası (27,6–32,2 sn)               DOĞRU
  ALARM 57.3 sn  yalnız görüntü  0.95  ← otobüs kavgası (52,4–57,8 sn)                DOĞRU
  ALARM 77.3 sn  yalnız görüntü  1.00  ← hastaya yardım (73,8–81,9 sn)                YANLIŞ, "düşmeyle çakışıyor"
  DÜŞME 77.3 sn                        ← yapay zekâ ile üretilmiş hasta düşmesi        DOĞRU
```

Sarılma/çak, normal otobüs ve günlük hareket için alarm yok. **Kaçanlar:** sokak kavgası (7,5–13,7 sn; düşük
çözünürlüklü YouTube kaydı, iskeletler kötü çıkıyor) ve GMDCSA'nın iki düşmesi (yatağa geriye düşme; yatak
kenarından öne düşme). Bayılma ve hareketsizlik demoda gösterilemiyor: elimizdeki bütün klipler 15 sn'den kısa.

Örnek çıktı (yapay zekâ ile üretilmiş bir hastane koridoru klibi, sahne pikselli): `docs/hastane_bayilma_isaretli.mp4`,
kareler `docs/bayilma_kareler.jpg`.

### Yeniden üretmek

1. Veri setlerini kendi sayfalarından `veri/ham/` altına indirin (bkz. LISANSLAR.md): `bus/` (Bus Violence,
   Zenodo 7044203), `airtlab/` (GitHub airtlab), `survfight/` (GitHub seymanurakti), `gmdcsa/` (Zenodo 12921216),
   ses ölçümü için `ESC-50-master/`.
2. `araclar/liste_olustur.py` → `araclar/poz_toplu.py` → `python -m siddet.egit` → `python -m siddet.olc` →
   `araclar/dusme_olc.py` → `araclar/demo_videosu.py`.

## Ölçüm sonuçları (test kümesi, eğitimde görülmedi)

Bölmeler sızıntıyı önler: Bus Violence'ın resmî test listesi; AIRTLab'da aynı olayın iki kamerası aynı
bölümde; Surveillance Fight'ta aynı YouTube kaynağı aynı bölümde; GMDCSA-24'te 4. denek yalnız testte.

### 1) Klip düzeyinde (yalnız görüntü, eşik 0,75)

| Küme | Klip | Kesinlik | Duyarlılık | Yanlış pozitif | AUC |
|---|---|---|---|---|---|
| **Hepsi** | 447 (212 kavga) | **0,85** | **0,61** | 23 / 235 | **0,87** |
| Bus Violence (otobüs) | 280 | 0,87 | 0,56 | 12 / 140 | 0,87 |
| AIRTLab (oda, oyuncular) | 80 | 0,85 | 0,88 | 8 / 30 | 0,88 |
| Surveillance Fight (YouTube CCTV) | 50 | 0,70 | 0,32 | 3 / 28 | 0,67 |

Zor olumsuzlarda yanlış pozitif: **hasta düşmesi 0/17**, **günlük hareket (yatma, oturma, eğilme) 0/20**,
sarılma/çak/alkış 8/30, normal otobüs 12/140, normal sokak 3/28. Eşik 0,97'de kesinlik 0,91, duyarlılık 0,36.

### 2) Alarm düzeyinde: yanlış alarm sıklığı

Test klipleri iskelet düzeyinde uç uca eklenerek uzun kamera akışları kuruldu. **Olumsuz akış**: 235 kavga
olmayan klip, **13,9 dakika** kesintisiz zor hareket (her an biri sarılıyor, düşüyor, otobüste itişiyor vb.).
**Olaylar**: 103 kavga (otobüs ve sokakta art arda 3 klip ≈ 5–6 sn; AIRTLab'da tek klip).

| "Yalnız görüntü" eşiği | Yanlış alarm (13,9 dk) | Zor hareket saati başına | %95 güven aralığı | Kavga yakalama |
|---|---|---|---|---|
| 0,80 | 10 | 43 | 21–79 | 70/103 (%68) |
| 0,90 | 6 | 26 | 9–56 | 59/103 (%57) |
| **0,95 (seçilen)** | **4** | **17** | **5–44** | **46/103 (%45)** |
| 0,98 | 2 | 9 | 1–31 | 36/103 (%35) |

0,95'te 4 yanlış alarmın **dördü de sahnelenmiş sarılma/çak/alkış kliplerinden** (3,1 dk). Hasta düşmesi,
günlük hareket, otobüs ve sokak akışlarında (10,8 dk) yanlış alarm yok.

**Kamera başına günde yanlış alarm** = (zor hareket saati başına alarm) × (kameranın günde gördüğü zor hareket
süresi). Örnek varsayım: okul koridorunda teneffüslerde günde ~1,5 saat yoğun hareket → **günde ~26
(güven aralığı 7–66)**. Bu, ürün için kabul edilemez (hedef: kamera başına haftada birkaç). Tahmin 14 dakikalık
ölçüme dayanır; aralık geniştir. **Boş ya da sakin koridor ölçülmedi** (orada iki kişiden az olduğu için puan
zaten düşer). Koşan/oynayan çocuk, spor ve kalabalık koridor **ölçülemedi** (aşağıda "Açık riskler").

### 3) Düşme, bayılma, hareketsizlik (`araclar/dusme_olc.py` → `test-sonuclari/dusme_olcumu.json`)

Eşikler GMDCSA-24'ün 1–3. deneklerinde tarandı, 4. denekte (hiç görülmemiş ev ve kişi) test edildi.

| Küme | Düşme yakalama | Günlük harekette yanlış düşme |
|---|---|---|
| Denek 1–3 (ayar) | 44/62 (%71) · yere düşme 43/57 (%75) | 1/61 |
| **Denek 4 (test)** | **7/17 (%41) · yere düşme 6/11 (%55)** · yatağa düşme 1/6 | **2/20** (şınav; yatağa uzanma) |

- **Yanlış düşme başka yerde yok:** normal otobüs 0/140, sarılma/çak 0/30, normal sokak 0/28, üretilmiş 7 olumsuz klip
  (koşan çocuk, şakalaşma, basketbol, kucaklaşma, kalabalık koridor, ajite hasta, okul çıkışı) 0/7.
  Üretilmiş hasta düşmesi yakalandı. Kavga kliplerinde 6/212 "düşme" çıktı (kavgada yere düşülüyor; doğru sayılabilir).
- **Bayılma: yapay zekâ ile üretilmiş 2 klipte (15 sn) sınandı** (`araclar/bayilma_olc.py` →
  `test-sonuclari/bayilma_olcumu.json`).

  | Klip | Kişinin görüldüğü kare | Düşme | Bayılma (10 sn eşik) | Bayılma (4 sn eşik) |
  |---|---|---|---|---|
  | Hastane koridoru, yaşlı adam yığılıyor (640) | 205/226 | 9,0 sn ✓ | yok (klip bitti) | 14,4 sn ✓ |
  | Okul koridoru, uzaktaki öğrenci düşüyor (640) | 4/226 | yok | yok | yok |
  | Aynı okul klibi, `--imgsz 1280` | 205/226 | 3,75 sn ✓ | yok (klip bitti) | 9,2 sn ✓ |

  İki düşme de yakalandı; ardından 4 sn'lik eşikle "bayılma şüphesi" çıktı. Klipler 15 sn olduğu için 10 sn'lik
  varsayılan eşik dolmadı. Okul klibinde öğrenci çok küçük: varsayılan 640 çözünürlükte poz modeli onu neredeyse
  hiç görmüyor, 1280'de görüyor. 30 sn'lik "hareketsiz yatıyor" hiç sınanmadı. Çıktı kareleri:
  `docs/bayilma_kareler.jpg`.
- **GMDCSA'da vekil ölçüm:** klipler 2–17 sn; düşmeden sonra en çok birkaç saniye kayıt var.
  1,5 sn'lik vekil eşikle: düşmeden sonra yerde hareketsiz kalma 18/79 düşme klibinde görüldü; günlük harekette
  (yatakta uzanma) 8/81 klipte "hareketsiz yatıyor" çıktı. Yatak bölgesi tanımlanmadan hastane odasında bu modül
  uzanan hastalar için uyarı üretir. 10 ve 30 sn'lik gerçek eşikler için uzun klip gerekir.
- Düşme modülü eğitimsiz ve GMDCSA-24 (CC BY 4.0) ile ayarlandı: kavga modelinin aksine **lisans engeli yok**.

### 4) Üretilmiş zor olumsuz kliplerde kavga alarmı (`araclar/uretilen_olc.py`)

Yapay zekâ ile üretilmiş 8 sahne (Seedance 2.5, 720p, 8 sn), yalnız ölçüm için:

| Klip | En yüksek kavga puanı | Kavga alarmı |
|---|---|---|
| Koridorda koşan çocuklar | 1,00 | **evet (yanlış)** |
| Çimde güreşerek şakalaşan çocuklar | 1,00 | **evet (yanlış)** |
| Basketbol (temaslı) | 0,99 | **evet (yanlış)** |
| Hastane girişinde kucaklaşma | 0,68 | hayır |
| Kalabalık hastane koridoru, sedye | 0,86 | hayır |
| Hasta düşmesi + hemşire yardımı | 1,00 | **evet (yanlış, düşmeyle çakışıyor)** |
| Ajite hastayı sakinleştirme | 0,14 | hayır |
| Okul çıkışı kalabalık | 0,13 | hayır |

**8 klipten 4'ü (toplam 64 sn) yanlış kavga alarmı verdi.** Okulda koşan, şakalaşan çocuklar ve spor, bu modelin
en zayıf yeri; eğitim verisinde hiç çocuk ve spor yok. Bu sahneler gerçek bir okulda her teneffüste olur.
Üretilmiş klipler yapay; gerçek kamerada oran farklı olabilir, ama yön açık.

### 5) Ses

- Test videolarının çoğu sessiz (Bus, AIRTLab) ya da sesi sonradan eklenmiş müzik (YouTube). **Gerçek bir
  kavganın sesiyle ölçüm yapılamadı**; ses yolu bu yüzden üç senaryoda ölçüldü:
  - **Gündelik ses bindirme** (olumsuz kliplere ESC-50'den bebek ağlaması, gülme, alkış, ayak sesi, kapı,
    öksürük, elektrikli süpürge, siren...): **ek yanlış alarm 0**.
  - **İyimser üst sınır** (her kavgada güçlü bir çığlık olduğu varsayımı): yakalama %45 → **%91**, yanlış alarm
    aynı (17/saat). Gerçek değer bu ikisinin arasında; kavgaların ne kadarında çığlık/darbe sesi olduğuna bağlı.
- **ESC-50** (2.000 klip, 50 sınıf) üzerinde ses olayı algılayıcısı tek başına: **cam kırılması 33/40 (%82)**
  yakalandı; diğer 49 sınıfta 1.960 klipten yalnız 3'ünde olay çıktı (gülme 2/40, bebek ağlaması 1/40),
  yani gündelik seste **saatte ~1 ses olayı**. Tek başına alarm üretmez; görüntüyle birlikte gerekir.
- Çığlık sınıfı için etiketli test sesi yok (ESC-50'de çığlık sınıfı yok).

## Açık riskler

1. **Yanlış alarm hâlâ yüksek.** En büyük kaynak sarılma, çak, kucaklaşma gibi yakın temaslı hızlı hareket.
   Bu sınıf için eğitimde yalnız 86 klip var. Zor olumsuzu 3–6 kat ağırlıklandırmak sonucu değiştirmedi;
   daha çok ve çeşitli olumsuz veri gerekiyor.
2. **Kaçan kavgalar.** Duyarlılık %45 (yalnız görüntü). Düşük çözünürlük, uzak kamera, örtüşen kişiler ve
   yerde boğuşmada iskelet çıkmıyor; Surveillance Fight'ta AUC yalnız 0,67.
3. **Çocuk ve spor sahneleri yanlış kavga alarmı veriyor.** Gerçek görüntü yerine (Wikimedia Commons bu
   bilgisayarı geçici olarak engelledi, HTTP 429; betik hazır: `araclar/commons_indir.py`) yapay zekâ ile üretilmiş
   8 klip ölçüldü: koşan çocuklar, şakalaşma ve basketbolda yanlış alarm çıktı (bkz. ölçüm 4). Okulda kullanmadan
   önce bu sahnelerle eğitmek şart.
4. **Ortam farkı.** Eğitim verisi otobüs, tek bir oda ve YouTube CCTV kayıtları. Hastane/okul kamerası (tavan
   açısı, geniş koridor, çocuk boyları, sedye, tekerlekli sandalye) hiç görülmedi. Hasta-personel boğuşması
   (ajite hasta zaptı), çocuk itişmesi ve şakalaşma gibi gri durumlar tanımlı değil.
5. **Ses hiç gerçek kavgada sınanmadı.** Hastanede sık olan bağırma, ağlama ve alarm sesleri gibi koşullar
   ölçülmedi. Okul koridorunda "bağırma" çok sık olur; bu yüzden ses tek başına alarm üretmez.
6. **Lisans.** Kavga modelinin eğitim verisi ticari kullanıma uygun değil (bkz. LISANSLAR.md). YOLO11 AGPL-3.0.
7. **Gizlilik ve hukuk.** Görüntü işleme yerel; ama hastane/okulda kamera analitiği KVKK (aydınlatma, açık
   rıza ya da meşru menfaat, veri sorumlusu kaydı) ve kurum onayı ister. Ham video bu prototipte analiz için
   açılır; yalnız *çıktı* gizlidir. Çocuklar söz konusu olduğunda ek dikkat gerekir.
8. **Kötüye kullanım.** Sistem "kavga şüphesi" der, kim olduğunu söylemez; bir kişiyi suçlu ilan etmek ya da
   disiplin kararı için kullanılmamalı. Her alarmı bir insan doğrulamalı.
9. **Düşme modülü sınırlı ölçüldü.** Test deneğinde yere düşmelerin yarısı kaçtı (4 deneğin 1'i, 17 klip). Kameraya
   doğru ya da kameradan uzağa düşüş, kısmen kadraj dışında kalan beden, yataktan yere yuvarlanma zor. Şınav,
   yere oturma, yatağa hızla uzanma yanlış düşme üretebilir. Bayılma (10 sn) ve hareketsiz yatma (30 sn) eşikleri
   yalnız 2 yapay klipte, kısaltılmış eşikle sınandı; yatak/sedye bölgeleri tanımlanmazsa yatan hastalar uyarı
   üretir. Uzaktaki küçük kişiler için `--imgsz 1280` gerekir (daha yavaş).
10. **Hız.** RTX 5070 Ti'de 62 sn'lik video 36 sn'de işlendi (model yükleme dahil). İşlemcide ve çok kamerada
   ölçülmedi; AugeLab Studio'ya bağlanmadı (şimdilik saf Python).

## Sonraki adımlar (önerilen)

1. Kendi sahnelenmiş çekimlerimiz: izinli oyuncularla okul koridoru ve hastane bekleme salonu benzeri mekânda,
   2–3 kamera açısı; kavga, itişme, sarılma, şakalaşma, koşma, düşme, hasta zaptı. Hem ticari temiz veri hem doğru ortam.
2. Commons engeli kalkınca gerçek zor olumsuzları (`araclar/commons_indir.py`) ekleyip ölçümü yenilemek; çocuk
   ve spor sahnelerini olumsuz örnek olarak eğitime katmak (en büyük yanlış alarm kaynağı).
3. Bus Violence yazarlarından ticari kullanım teyidi; RWF-2000 için SMIIP Lab'e başvuru.
4. Yapay zekâ ile üretilmiş zor olumsuz klipler: aşağıdaki 8 klip üretildi ve ölçüldü.
5. Bayılma / hareketsizlik: 10 ve 30 sn'lik eşikler için gerçek, uzun (en az 1 dk) kayıt gerekir.

### Yapay zekâ çekim listesi (üretildi, 720p)

Amaç: hastane/okul ortamında **zor olumsuz** örnekler (yanlış alarm ölçümü için). Kavga sahnesi üretilmez
(modeller reddedebilir; ayrıca sentetik kavgayla eğitim riskli). Hepsi 16:9, 720p, 8 sn, sessiz, sabit tavan
güvenlik kamerası açısı.

| # | Sahne | Prompt (İngilizce, modele giden) |
|---|---|---|
| 1 | Teneffüste koşan çocuklar | `Fixed ceiling CCTV view of a school corridor during break, 8-10 year old children running and chasing each other, playful, wide angle, slight fisheye, realistic security camera footage, 720p` |
| 2 | Okul bahçesinde güreş-şakalaşma | `Fixed high CCTV view of a schoolyard, two children play-wrestling and laughing on grass, other kids around, realistic surveillance footage` |
| 3 | Beden eğitimi basketbol | `Fixed CCTV view of a school gym, teenagers playing basketball, fast contact under the hoop, realistic security camera footage` |
| 4 | Hastane girişinde kucaklaşma | `Fixed ceiling CCTV view of a hospital lobby, family members hugging emotionally, people walking past, realistic surveillance footage` |
| 5 | Kalabalık hastane koridoru | `Fixed ceiling CCTV view of a crowded hospital corridor, patients, nurses pushing a stretcher, visitors walking quickly, realistic surveillance footage` |
| 6 | Hasta düşmesi ve yardım | `Fixed ceiling CCTV view of a hospital ward, elderly patient slips and falls next to the bed, a nurse rushes to help him up, realistic surveillance footage` |
| 7 | Ajite hastayı sakinleştirme | `Fixed CCTV view of a hospital emergency waiting area, a nurse gently holds the arms of an agitated patient and calms him down, no hitting, realistic surveillance footage` |
| 8 | Okul çıkışı itiş-kakış kalabalık | `Fixed CCTV view of a school exit door at dismissal time, dense crowd of students pushing gently through the door, realistic security camera footage` |

Model: Seedance 2.5 (Higgsfield), her sahneden tek deneme. Üretilen klipler yalnız **ölçüm** için kullanılır, eğitime girmez (sentetik görüntüden öğrenilen yanlılık riskini önlemek için).

### Bayılma klipleri (üretildi: B1, B2)

Amaç: 10 sn'lik bayılma ve 30 sn'lik hareketsiz yatma eşiklerini sınamak; düşmeden sonra kişinin uzun süre yerde
kalması gerekir. Hepsi 16:9, 720p, **15 sn** (modelin en uzun süresine yakın), sessiz, sabit tavan güvenlik kamerası.

| # | Sahne | Beklenen | Prompt |
|---|---|---|---|
| B1 | Hastane koridorunda yürürken bayılıp yığılan yaşlı adam, yerde kıpırdamadan kalıyor | düşme + bayılma | `Fixed ceiling CCTV view of an empty hospital corridor, an elderly man walking slowly suddenly faints and collapses to the floor, then lies completely still on the floor until the end, nobody around, realistic surveillance footage` |
| B2 | Okul koridorunda bayılan öğrenci, yerde hareketsiz | düşme + bayılma | `Fixed ceiling CCTV view of a quiet school corridor, a teenage student walking alone faints and falls to the floor, then lies motionless on the floor for the rest of the video, realistic security camera footage` |

B1 ve B2, Seedance 2.5 ile 15 sn, 720p, sessiz olarak üretildi (tek deneme).

## Dosyalar

| Yol | Ne |
|---|---|
| `siddet/poz.py` | YOLO11-pose + ByteTrack, yüz gizleme (`YuzGizleyici`) |
| `siddet/ozellik.py` | iskelet penceresi → özellik tensörü (T=24, M=4, F=93) |
| `siddet/model.py` | iskelet sınıflandırıcı + topluluk |
| `siddet/egit.py` | eğitim, eşik seçimi, klip düzeyinde test → `test-sonuclari/iskelet_test.json` |
| `siddet/ses.py` | PANNs Cnn14 ses olayı (bellekte, kayıtsız) |
| `siddet/birlestir.py` | alarm kuralı |
| `siddet/dusme.py` | düşme, bayılma şüphesi, hareketsiz yatma (kural) |
| `siddet/analiz.py` | video → zaman çizelgesi (JSON + HTML) + gizlilikli video |
| `siddet/olc.py` | alarm düzeyinde ölçüm → `test-sonuclari/alarm_olcumu.json` |
| `araclar/` | bayılma ölçümü (`bayilma_olc.py`), model indirme, veri listesi, toplu iskelet çıkarma, Commons indirme, demo videosu, düşme ölçümü (`dusme_olc.py`), üretilmiş klip ölçümü (`uretilen_olc.py`) |
| `LISANSLAR.md` | veri seti ve model lisans tablosu |

`veri/`, `cikti/`, `ornekler/` ve model ağırlıkları depoya girmez (`.gitignore`). Modeller: `yolo11m-pose.pt`
(Ultralytics), `Cnn14_mAP=0.431.pth` (Zenodo 3987831), `yunet_yuz.onnx` (OpenCV Zoo) `araclar/modelleri_indir.py`
ile iner; `iskelet_kavga.pt` eğitimle üretilir.

## Lisans

Kod **AGPL-3.0** (Ultralytics YOLO11 AGPL-3.0 olduğu için). Veri setleri ve model ağırlıkları kendi lisanslarına
tabidir: [LISANSLAR.md](LISANSLAR.md). Bu bir araştırma prototipidir; onaylı bir güvenlik ya da tıbbi uyarı
sistemi değildir, her alarmı bir insan doğrulamalıdır.

## Atıflar

Bus Violence (Ciampi ve ark., Sensors 2022) · AIRTLab violence dataset (Bianculli ve ark., Data in Brief 2020) ·
Surveillance Camera Fight (Aktı ve ark., IPTA 2019) · GMDCSA-24 (Alam ve ark., Data in Brief 2024) ·
ESC-50 (Piczak, ACM MM 2015, CC BY-NC 3.0, yalnız ses ölçümünde) · PANNs (Kong ve ark., IEEE/ACM TASLP 2020) ·
Ultralytics YOLO11 · YuNet (Shiqi Yu, OpenCV Zoo).
