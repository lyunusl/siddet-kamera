"""Düşme, bayılma şüphesi ve hareketsiz yatma algılama (iskelet tabanlı, kural).

Her takip edilen kişi için her karede duruş çıkarılır:
  - gövde açısı: omuz ortası → kalça ortası doğrusunun dikeyle açısı
  - kutu oranı (genişlik / yükseklik), gövde görünmezse yedek ölçüt
  - kalça merkezinin yüksekliği, ayakta duruştaki boya göre
  - hareket: eklemlerin saniyedeki yer değiştirmesi, vücut uzunluğuna göre
Olaylar:
  DUSME      : kişi son DUSUS_PENCERE sn içinde dik duruyordu, kalçası boyunun en az DUSUS_ORAN'ı
               kadar aşağı indi ve en az YATMA_KALICI sn yatay kaldı.
  BAYILMA    : düşmenin ardından BAYILMA_SN boyunca yerde hareketsiz kaldı ("bayılma şüphesi").
  HAREKETSIZ : düşme görülmese de HAREKETSIZ_SN boyunca yatay ve hareketsiz (yatak bölgesi dışında).
Yatak/sedye bölgeleri (çokgen, 0-1 göreli koordinat) içinde yatmak olağandır: orada HAREKETSIZ
üretilmez; yataktan yere düşüş ise kalça bölge dışına çıktığı için yine yakalanır.
Duygu, yüz ya da kimlik kullanılmaz.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .poz import PozDizisi

G = 0.3  # eklem güven eşiği


@dataclass
class DusmeAyar:
    YATAY_ACI: float = 60.0        # gövde açısı bundan büyükse yatay (GMDCSA denek 1-3 ile seçildi)
    DIK_ACI: float = 35.0          # bundan küçükse dik
    YATAY_KUTU: float = 1.1       # gövde görünmezse: kutu oranı bundan büyükse yatay
    DIK_KUTU: float = 0.75
    DUSUS_PENCERE: float = 2.0     # sn: yatmadan önce bakılan süre
    DUSUS_ORAN: float = 0.25       # kalça inişi / vücut uzunluğu
    DUSUS_HIZ_SN: float = 0.6      # iniş en çok bu kadar sürede olmalı (yavaşça uzanmak düşme değil)
    YUVARLANMA_ORAN: float = 0.45  # önce dik görülmediyse (yataktan yere yuvarlanma) gereken iniş
    YATMA_KALICI: float = 0.4      # sn
    HAREKET_ESIK: float = 0.06     # vücut uzunluğu / sn
    BAYILMA_SN: float = 10.0
    HAREKETSIZ_SN: float = 30.0
    BOSLUK_SN: float = 3.0         # takip kopukluğu toleransı
    ESLESTIRME: float = 1.0        # yeni iz, kaybolan izin yerinde mi? (boy cinsinden uzaklık)
    yatak_bolgeleri: list = field(default_factory=list)  # [[[x,y],...], ...] 0-1 göreli


def _ic(nokta, cokgen) -> bool:
    x, y = nokta
    ic = False
    n = len(cokgen)
    for i in range(n):
        x1, y1 = cokgen[i]; x2, y2 = cokgen[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            ic = not ic
    return ic


def _durus(kutu, kp, ayar: DusmeAyar, W=None, H=None):
    x1, y1, x2, y2 = kutu
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    oran = w / h
    om, ka = kp[[5, 6]], kp[[11, 12]]
    om_ok, ka_ok = om[:, 2] > G, ka[:, 2] > G
    aci = None
    if om_ok.any() and ka_ok.any():
        o = om[om_ok, :2].mean(0); k = ka[ka_ok, :2].mean(0)
        d = o - k
        if np.linalg.norm(d) > 0.08 * max(w, h):
            aci = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-6)))
    # kadraj kenarına değen kutu kesik kişidir (oturan, yarısı görünen): oranı yataylık sayılmaz
    kenarda = W is not None and (x1 < 0.02 * W or x2 > 0.98 * W or y2 > 0.98 * H or y1 < 0.02 * H)
    kutu_yatay = oran > ayar.YATAY_KUTU and not kenarda
    if aci is not None:
        # kameraya doğru/ters düşüşte gövde kısalır ve açı dik görünür; kutu yine yatay olur
        yatay = aci > ayar.YATAY_ACI or kutu_yatay
        dik = aci < ayar.DIK_ACI and oran < 1.0
    else:
        yatay, dik = kutu_yatay, oran < ayar.DIK_KUTU
    kalca = ka[ka_ok, :2].mean(0) if ka_ok.any() else np.array([(x1 + x2) / 2, y1 + 0.55 * h])
    return dict(aci=aci, oran=oran, yatay=bool(yatay), dik=bool(dik), kalca=kalca, boy=h, uzunluk=max(w, h))


def izleri_cikar(p: PozDizisi, ayar: DusmeAyar):
    izler: dict[int, list] = {}
    for k in p.kareler:
        for j, tid in enumerate(k.ids):
            d = _durus(k.kutular[j], k.noktalar[j], ayar, p.genislik, p.yukseklik)
            d.update(t=k.t, kp=k.noktalar[j], kutu=k.kutular[j])
            izler.setdefault(int(tid), []).append(d)
    return izler


def _birlestir(izler, ayar: DusmeAyar):
    """Düşerken takip numarası değişirse (sık olur) yeni izi eskisine bağlar."""
    sira = sorted(izler, key=lambda i: izler[i][0]["t"])
    bagli = {}
    for i in sira:
        bas = izler[i][0]
        en_iyi, en_d = None, 1e9
        for j in sira:
            if j == i or j in bagli.values():
                continue  # bir iz yalnız bir yeni ize bağlanır
            son = izler[j][-1]
            # eski iz, yeni iz başlamadan önce bitmiş olmalı (aynı anda görülen iki kişi bağlanmaz; döngü olmaz)
            if not (0 < bas["t"] - son["t"] <= 1.5) or izler[j][0]["t"] >= bas["t"]:
                continue
            boy = max(son["boy"], 1)
            d = np.linalg.norm(bas["kalca"] - son["kalca"]) / boy
            if d < ayar.ESLESTIRME and d < en_d:
                en_iyi, en_d = j, d
        if en_iyi is not None:
            bagli[i] = en_iyi
    def kok(i):
        while i in bagli:
            i = bagli[i]
        return i
    sonuc: dict[int, list] = {}
    for i in sira:
        sonuc.setdefault(kok(i), []).extend(izler[i])
    for v in sonuc.values():
        v.sort(key=lambda d: d["t"])
    return sonuc


def olaylari_bul(p: PozDizisi, ayar: DusmeAyar = DusmeAyar()):
    W, H = max(p.genislik, 1), max(p.yukseklik, 1)
    izler = _birlestir(izleri_cikar(p, ayar), ayar)
    olaylar = []
    for tid, iz in izler.items():
        t = np.array([d["t"] for d in iz])
        uzunluk = np.array([d["uzunluk"] for d in iz])
        kalca_y = np.array([d["kalca"][1] for d in iz])
        yatay = np.array([d["yatay"] for d in iz])
        dik = np.array([d["dik"] for d in iz])
        # hareket: ~1 sn önceki kareye göre eklem yer değiştirmesi (kare kare titreşim birbirini götürür)
        hareket = np.full(len(iz), np.nan)
        for i in range(len(iz)):
            j = np.searchsorted(t, t[i] - 1.0)
            if j >= i or t[i] - t[j] < 0.6 or t[i] - t[j] > 1.6:
                continue
            a_, b_ = iz[j]["kp"], iz[i]["kp"]
            ok = (a_[:, 2] > G) & (b_[:, 2] > G)
            if ok.sum() >= 4:
                hareket[i] = np.median(np.linalg.norm(b_[ok, :2] - a_[ok, :2], axis=1)) / uzunluk[i] / (t[i] - t[j])
        yatakta = np.array([any(_ic((d["kalca"][0] / W, d["kalca"][1] / H), c) for c in ayar.yatak_bolgeleri) for d in iz])
        son_dusme = None
        hareketsiz_bas = None
        yatma_bas = None
        bildirilen = set()
        for i in range(len(iz)):
            if i and t[i] - t[i - 1] > ayar.BOSLUK_SN:
                hareketsiz_bas = None; yatma_bas = None
            yatma_bas = (t[i] if yatma_bas is None else yatma_bas) if yatay[i] else None
            # düşme: yatay kalıcı + son DUSUS_PENCERE içinde DUSUS_HIZ_SN'den kısa sürede büyük kalça inişi
            if yatma_bas is not None and t[i] - yatma_bas >= ayar.YATMA_KALICI and (son_dusme is None or yatma_bas > son_dusme["t"] + 0.5):  # yeni düşme = yeni yatma
                pen = np.where((t >= yatma_bas - ayar.DUSUS_PENCERE) & (t <= t[i]))[0]
                ref = float(np.median(uzunluk[pen])) if len(pen) else 1.0
                en_inis = 0.0
                for a1 in pen:
                    b1 = pen[(t[pen] > t[a1]) & (t[pen] <= t[a1] + ayar.DUSUS_HIZ_SN)]
                    if len(b1):
                        en_inis = max(en_inis, float((kalca_y[b1].max() - kalca_y[a1]) / max(ref, 1)))
                once_dik = bool(dik[(t >= yatma_bas - ayar.DUSUS_PENCERE) & (t < yatma_bas)].any())
                if en_inis >= (ayar.DUSUS_ORAN if once_dik else ayar.YUVARLANMA_ORAN):
                    son_dusme = dict(tur="dusme", t=round(float(yatma_bas), 2), iz=tid, inis=round(en_inis, 2),
                                     yatakta=bool(yatakta[i]))
                    olaylar.append(son_dusme)
            # hareketsizlik: yatay ve yavaş; ölçülemeyen kareler süreyi bozmaz
            if yatay[i] and (np.isnan(hareket[i]) or hareket[i] < ayar.HAREKET_ESIK):
                hareketsiz_bas = t[i] if hareketsiz_bas is None else hareketsiz_bas
            else:
                hareketsiz_bas = None
            if hareketsiz_bas is not None:
                sure = t[i] - hareketsiz_bas
                if son_dusme is not None and hareketsiz_bas - son_dusme["t"] <= 5 and sure >= ayar.BAYILMA_SN                         and ("bayilma", son_dusme["t"]) not in bildirilen:
                    bildirilen.add(("bayilma", son_dusme["t"]))
                    olaylar.append(dict(tur="bayilma", t=round(float(t[i]), 2), iz=tid, dusme_t=son_dusme["t"],
                                        hareketsiz_sn=round(float(sure), 1)))
                elif sure >= ayar.HAREKETSIZ_SN and not yatakta[i] and ("hareketsiz", hareketsiz_bas) not in bildirilen                         and (son_dusme is None or ("bayilma", son_dusme["t"]) not in bildirilen):
                    bildirilen.add(("hareketsiz", hareketsiz_bas))
                    olaylar.append(dict(tur="hareketsiz", t=round(float(t[i]), 2), iz=tid,
                                        hareketsiz_sn=round(float(sure), 1)))
    return sorted(olaylar, key=lambda o: o["t"])


OLAY_ADLARI = {"dusme": "Düşme", "bayilma": "Bayılma şüphesi (düşme + hareketsiz)",
               "hareketsiz": "Hareketsiz yatıyor"}


def ayar_sozluk(a: DusmeAyar) -> dict:
    return asdict(a)
