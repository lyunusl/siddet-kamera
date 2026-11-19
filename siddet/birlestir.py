"""Görüntü ve ses puanlarını birleştirip alarm üreten kural.

Alarm iki yoldan çıkar:
  A) "Görüntü + ses": yumuşatılmış görüntü puanı >= V_ESIK ve ±SES_PAYI sn içinde bir ses
     olayı puanı >= A_ESIK.
  B) "Yalnız görüntü (çok yüksek)": yumuşatılmış görüntü puanı >= V_YUKSEK, en az
     SURE_YUKSEK sn boyunca kesintisiz.
Koşul KAPANMA_SN boyunca sağlanmazsa olay kapanır; BEKLEME_SN içinde gelen yeni koşul aynı
olaya eklenir (aynı kavga için tek alarm).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Ayar:
    V_ESIK: float = 0.5
    V_YUKSEK: float = 0.9
    A_ESIK: float = 0.3
    SES_PAYI: float = 2.0
    SURE_YUKSEK: float = 1.5
    SURE_IKISI: float = 0.5
    KAPANMA_SN: float = 3.0
    BEKLEME_SN: float = 10.0
    YUMUSATMA: int = 2      # son kaç görüntü puanının ortalaması


def ses_puani_dizisi(zamanlar, ses_cizelgesi, pay):
    """Her görüntü zamanı için ±pay içindeki en büyük ses olayı puanı ve grubu."""
    if not ses_cizelgesi:
        return np.zeros(len(zamanlar)), [None] * len(zamanlar)
    st = np.array([t for t, _ in ses_cizelgesi])
    puan, grup = [], []
    for t in zamanlar:
        ix = np.where(np.abs(st - t) <= pay)[0]
        en, eg = 0.0, None
        for i in ix:
            for g, p in ses_cizelgesi[i][1].items():
                if p > en:
                    en, eg = p, g
        puan.append(en); grup.append(eg)
    return np.array(puan), grup


def alarmlar(goruntu, ses_cizelgesi, ayar: Ayar = Ayar()):
    """goruntu: liste[(t_orta, puan, kisi_sayisi)]. Dönüş: (alarm listesi, nokta nokta ayrıntı)."""
    if not goruntu:
        return [], []
    t = np.array([g[0] for g in goruntu])
    v = np.array([g[1] for g in goruntu])
    k = ayar.YUMUSATMA
    vs = np.array([v[max(0, i - k + 1):i + 1].mean() for i in range(len(v))])
    a, ag = ses_puani_dizisi(t, ses_cizelgesi, ayar.SES_PAYI)
    adim = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
    ikisi = (vs >= ayar.V_ESIK) & (a >= ayar.A_ESIK)
    yuksek = vs >= ayar.V_YUKSEK
    # süre koşulu: kesintisiz koşu uzunluğu
    def kosu(m):
        n = np.zeros(len(m)); c = 0
        for i, x in enumerate(m):
            c = c + 1 if x else 0
            n[i] = c * adim
        return n
    kosul_b = kosu(yuksek) >= ayar.SURE_YUKSEK
    kosul_a = kosu(ikisi) >= ayar.SURE_IKISI
    tetik = kosul_a | kosul_b
    olaylar = []
    acik = None
    son_dogru = -1e9
    for i in range(len(t)):
        if tetik[i]:
            neden = "goruntu_ve_ses" if kosul_a[i] else "yalniz_goruntu"
            if acik is not None and t[i] - son_dogru <= max(ayar.KAPANMA_SN, ayar.BEKLEME_SN):
                acik["bit"] = float(t[i])
            else:
                if acik is not None:
                    olaylar.append(acik)
                acik = dict(bas=float(t[i]), bit=float(t[i]), nedenler=set(), en_goruntu=0.0, en_ses=0.0,
                            ses_olaylari=set())
            acik["nedenler"].add(neden)
            acik["en_goruntu"] = max(acik["en_goruntu"], float(vs[i]))
            if a[i] >= ayar.A_ESIK:
                acik["en_ses"] = max(acik["en_ses"], float(a[i]))
                acik["ses_olaylari"].add(ag[i])
            son_dogru = t[i]
    if acik is not None:
        olaylar.append(acik)
    for o in olaylar:
        o["nedenler"] = sorted(o["nedenler"]); o["ses_olaylari"] = sorted(x for x in o["ses_olaylari"] if x)
        o["bas"] = round(o["bas"], 2); o["bit"] = round(o["bit"], 2)
        o["en_goruntu"] = round(o["en_goruntu"], 3); o["en_ses"] = round(o["en_ses"], 3)
    ayrinti = [dict(t=round(float(t[i]), 2), goruntu=round(float(v[i]), 3), goruntu_yumusak=round(float(vs[i]), 3),
                    ses=round(float(a[i]), 3), ses_olayi=ag[i] if a[i] >= ayar.A_ESIK else None,
                    kisi=int(goruntu[i][2]), alarm=bool(tetik[i])) for i in range(len(t))]
    return olaylar, ayrinti


def ayar_sozluk(ayar: Ayar) -> dict:
    return asdict(ayar)
