"""İskelet dizisinden pencere tensörü.

Bir pencere PENCERE_SN saniye, T adımdır (15 adım/sn). Pencerede en çok M kişi seçilir:
en çok hareket eden ve birbirine en yakın olan kişiler öne alınır. Her kişi için her adımda:
kendi kalça merkezine ve boyuna göre normalleştirilmiş poz, eklem güvenleri, grup merkezine
göre konum, poz ve konum hızları, en yakın kişiye uzaklık ve yön, görünürlük bayrağı.
Görüntü, yüz ya da kimlik bilgisi yoktur; yalnızca eklem geometrisi.
"""
from __future__ import annotations

import numpy as np

from .poz import PozDizisi, SOL_SAG

PENCERE_SN = 1.6
ADIM_FPS = 15.0
T = int(round(PENCERE_SN * ADIM_FPS))  # 24
M = 4
F = 34 + 17 + 2 + 34 + 2 + 1 + 2 + 1  # 93
GUVEN_ESIK = 0.2


def _izler(p: PozDizisi, t0: float):
    """Pencere adımlarına en yakın kareyi eşleyip iz sözlüğü döndürür: id -> (T,17,3), (T,4)."""
    zaman = np.array([k.t for k in p.kareler]) if p.kareler else np.zeros(0)
    izler: dict[int, dict] = {}
    for s in range(T):
        t = t0 + s / ADIM_FPS
        if len(zaman) == 0:
            break
        i = int(np.argmin(np.abs(zaman - t)))
        if abs(zaman[i] - t) > 1.5 / ADIM_FPS:
            continue
        k = p.kareler[i]
        for j, tid in enumerate(k.ids):
            iz = izler.setdefault(int(tid), {"kp": np.zeros((T, 17, 3), np.float32),
                                             "kutu": np.zeros((T, 4), np.float32),
                                             "var": np.zeros(T, bool)})
            iz["kp"][s] = k.noktalar[j]
            iz["kutu"][s] = k.kutular[j]
            iz["var"][s] = True
    return izler


def _merkez(kp, kutu):
    kalca = kp[:, [11, 12]]
    ok = (kalca[:, :, 2] > GUVEN_ESIK).all(1)
    m = np.stack([(kutu[:, 0] + kutu[:, 2]) / 2, (kutu[:, 1] + kutu[:, 3]) / 2], 1)
    m[ok] = kalca[ok, :, :2].mean(1)
    return m


def pencere_tensoru(p: PozDizisi, t0: float) -> tuple[np.ndarray, int]:
    """(T, M, F) tensör ve pencerede görülen kişi sayısını döndürür."""
    izler = _izler(p, t0)
    adaylar = []
    for tid, iz in izler.items():
        if iz["var"].mean() < 0.3:
            continue
        boy = np.median((iz["kutu"][iz["var"], 3] - iz["kutu"][iz["var"], 1]))
        if boy < 8:
            continue
        iz["boy"] = float(boy)
        iz["merkez"] = _merkez(iz["kp"], iz["kutu"])
        adaylar.append((tid, iz))
    x = np.zeros((T, M, F), np.float32)
    if not adaylar:
        return x, 0
    # hareket puanı ve etkileşim puanı
    puan = []
    for tid, iz in adaylar:
        v = iz["var"]
        kp = iz["kp"][:, 5:, :2] / iz["boy"]
        hiz = np.linalg.norm(np.diff(kp, axis=0), axis=-1)[(v[1:] & v[:-1])]
        hareket = float(hiz.mean()) if hiz.size else 0.0
        yakin = 0.0
        for tid2, iz2 in adaylar:
            if tid2 == tid:
                continue
            ortak = v & iz2["var"]
            if ortak.any():
                d = np.linalg.norm(iz["merkez"][ortak] - iz2["merkez"][ortak], axis=1) / max(iz["boy"], iz2["boy"])
                yakin = max(yakin, float(np.exp(-d.min())))
        puan.append(hareket * 10 + yakin)
    secili = [adaylar[i] for i in np.argsort(puan)[::-1][:M]]
    S = float(np.median([iz["boy"] for _, iz in secili]))
    merkezler = np.stack([iz["merkez"] for _, iz in secili])         # (m, T, 2)
    varlar = np.stack([iz["var"] for _, iz in secili])               # (m, T)
    grup = np.zeros((T, 2), np.float32)
    for s in range(T):
        if varlar[:, s].any():
            grup[s] = merkezler[varlar[:, s], s].mean(0)
    for mi, (tid, iz) in enumerate(secili):
        v = iz["var"]
        kp = iz["kp"]
        guven = kp[:, :, 2] * (kp[:, :, 2] > GUVEN_ESIK)
        yerel = (kp[:, :, :2] - iz["merkez"][:, None]) / iz["boy"]
        yerel = yerel * (guven[..., None] > 0)
        konum = (iz["merkez"] - grup) / S
        yhiz = np.zeros_like(yerel)
        khiz = np.zeros_like(konum)
        iki = v[1:] & v[:-1]
        yhiz[1:][iki] = (yerel[1:] - yerel[:-1])[iki] * ADIM_FPS * 0.1
        yhiz[1:] *= ((guven[1:] > 0) & (guven[:-1] > 0))[..., None]
        khiz[1:][iki] = (konum[1:] - konum[:-1])[iki] * ADIM_FPS * 0.1
        en_yakin = np.full(T, 3.0, np.float32)
        yon = np.zeros((T, 2), np.float32)
        for mj in range(len(secili)):
            if mj == mi:
                continue
            ortak = v & varlar[mj]
            d = (merkezler[mj] - iz["merkez"]) / S
            n = np.linalg.norm(d, axis=1)
            daha = ortak & (n < en_yakin)
            en_yakin[daha] = n[daha]
            yon[daha] = d[daha]
        ozl = np.concatenate([yerel.reshape(T, 34), guven, konum, yhiz.reshape(T, 34), khiz,
                              np.minimum(en_yakin, 3.0)[:, None] / 3.0, np.clip(yon, -3, 3) / 3.0,
                              v[:, None].astype(np.float32)], 1)
        ozl[~v] = 0
        x[:, mi] = np.clip(ozl, -5, 5)
    return x, len(adaylar)


def pencere_baslangiclari(sure: float, adim: float) -> list[float]:
    if sure <= PENCERE_SN:
        return [max(0.0, (sure - PENCERE_SN) / 2)]
    n = int(np.floor((sure - PENCERE_SN) / adim)) + 1
    bas = [i * adim for i in range(n)]
    if sure - PENCERE_SN - bas[-1] > adim * 0.5:
        bas.append(sure - PENCERE_SN)
    return bas


def ayna(x: np.ndarray) -> np.ndarray:
    """Sağ-sol çevirme (veri çoğaltma)."""
    y = x.copy()
    poz = y[..., :34].reshape(*y.shape[:-1], 17, 2)[..., SOL_SAG, :]
    poz[..., 0] *= -1
    y[..., :34] = poz.reshape(*y.shape[:-1], 34)
    y[..., 34:51] = y[..., 34:51][..., SOL_SAG]
    y[..., 51] *= -1
    hz = y[..., 53:87].reshape(*y.shape[:-1], 17, 2)[..., SOL_SAG, :]
    hz[..., 0] *= -1
    y[..., 53:87] = hz.reshape(*y.shape[:-1], 34)
    y[..., 87] *= -1
    y[..., 90] *= -1
    return y
