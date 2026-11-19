"""İskelet kavga sınıflandırıcısını eğitir ve test kümesinde ölçer.

Kullanım:  .venv\\Scripts\\python -m siddet.egit
Çıktı:     modeller/iskelet_kavga.pt, test-sonuclari/iskelet_test.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .model import IskeletKavga, Topluluk
from .ozellik import F, M, T, ayna, pencere_baslangiclari, pencere_tensoru
from .poz import PozDizisi

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "araclar"))
from poz_toplu import poz_yolu  # noqa: E402

ONBELLEK = KOK / "veri" / "pencereler.npz"
EGITIM_ADIM = 0.8


def pencereleri_hazirla(yenile: bool = False):
    if ONBELLEK.exists() and not yenile:
        z = np.load(ONBELLEK, allow_pickle=True)
        return z["x"], z["y"], z["klip"], list(z["bilgi"])
    satir = list(csv.DictReader(open(KOK / "veri" / "liste.csv", encoding="utf-8")))
    xs, ys, klip, bilgi = [], [], [], []
    for ki, r in enumerate(satir):
        yol = poz_yolu(r)
        if not yol.exists():
            continue
        p = PozDizisi.yukle(yol)
        sure = max(p.sure, p.kareler[-1].t if p.kareler else 0)
        for t0 in pencere_baslangiclari(sure, EGITIM_ADIM):
            x, _ = pencere_tensoru(p, t0)
            xs.append(x)
            ys.append(int(r["etiket"]))
            klip.append(len(bilgi))
        bilgi.append(dict(r, sure=sure))
    x, y, klip = np.stack(xs).astype(np.float16), np.array(ys, np.int8), np.array(klip, np.int32)
    np.savez_compressed(ONBELLEK, x=x, y=y, klip=klip, bilgi=np.array(bilgi, dtype=object))
    return x, y, klip, bilgi


def cogalt(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    xn = x.cpu().numpy()
    ayn = np.random.rand(B) < 0.5
    if ayn.any():
        xn[ayn] = ayna(xn[ayn])
    x = torch.from_numpy(xn).to(x.device)
    # kişi sırası karıştır
    perm = torch.stack([torch.randperm(M) for _ in range(B)]).to(x.device)
    x = torch.gather(x, 2, perm[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3]))
    # bir kişiyi düşür
    dus = torch.rand(B, device=x.device) < 0.1
    if dus.any():
        j = torch.randint(0, M, (int(dus.sum()),), device=x.device)
        x[dus.nonzero().squeeze(1), :, j] = 0
    # zaman kaydırma
    k = int(np.random.randint(-2, 3))
    if k:
        x = torch.roll(x, k, 1)
        if k > 0:
            x[:, :k] = 0
        else:
            x[:, k:] = 0
    # ölçek ve gürültü (görünürlük bayrağı hariç)
    olcek = (0.9 + 0.2 * torch.rand(B, 1, 1, 1, device=x.device))
    gor = x[..., -1:].clone()
    x = x * olcek + 0.01 * torch.randn_like(x) * gor
    x[..., -1:] = gor
    return x


def tahmin(model, x: np.ndarray, cihaz, parti=1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), parti):
            xb = torch.from_numpy(x[i:i + parti].astype(np.float32)).to(cihaz)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def klip_puanlari(p: np.ndarray, klip: np.ndarray):
    d = defaultdict(list)
    for s, k in zip(p, klip):
        d[int(k)].append(float(s))
    # klip puanı: art arda iki pencerenin ortalamasının en büyüğü (tek pencere ise kendisi)
    sonuc = {}
    for k, v in d.items():
        v = np.array(v)
        sonuc[k] = float(v.max() if len(v) < 2 else np.convolve(v, [0.5, 0.5], "valid").max())
    return sonuc


def olc(y, s, esik):
    y, s = np.asarray(y), np.asarray(s)
    tp = int(((s >= esik) & (y == 1)).sum()); fp = int(((s >= esik) & (y == 0)).sum())
    fn = int(((s < esik) & (y == 1)).sum()); tn = int(((s < esik) & (y == 0)).sum())
    kes = tp / max(tp + fp, 1); duy = tp / max(tp + fn, 1)
    r = dict(n=len(y), pozitif=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn, kesinlik=round(kes, 3), duyarlilik=round(duy, 3),
             f1=round(2 * kes * duy / max(kes + duy, 1e-9), 3))
    if 0 < y.sum() < len(y):
        r["auc"] = round(float(roc_auc_score(y, s)), 3)
        r["ap"] = round(float(average_precision_score(y, s)), 3)
    return r


def main():
    torch.manual_seed(0); np.random.seed(0)
    x, y, klip, bilgi = pencereleri_hazirla("--yenile" in sys.argv)
    bolum = np.array([bilgi[k]["bolum"] for k in klip])
    print("pencere:", {b: int((bolum == b).sum()) for b in ("egitim", "dogrulama", "test")}, "F", x.shape)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    egt = np.where(bolum == "egitim")[0]
    dog = np.where(bolum == "dogrulama")[0]
    xt = torch.from_numpy(x[egt].astype(np.float32)).to(cihaz)
    yt = torch.from_numpy(y[egt].astype(np.float32)).to(cihaz)
    DEVIR, TOHUM = 60, 3
    oran = float((y[egt] == 0).sum() / max((y[egt] == 1).sum(), 1))
    kayip = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(oran, device=cihaz))
    durumlar = []
    bas = time.time()
    for tohum in range(TOHUM):
        torch.manual_seed(tohum); np.random.seed(tohum)
        model = IskeletKavga(F).to(cihaz)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=DEVIR * (len(egt) // 256 + 1))
        en_iyi, en_iyi_durum = -1, None
        for devir in range(DEVIR):
            model.train()
            sira = torch.randperm(len(egt), device=cihaz)
            toplam = 0.0
            for i in range(0, len(egt), 256):
                j = sira[i:i + 256]
                loss = kayip(model(cogalt(xt[j].clone())), yt[j])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                toplam += float(loss.detach()) * len(j)
            pd = tahmin(model, x[dog], cihaz)
            ks = klip_puanlari(pd, klip[dog])
            ap = average_precision_score([int(bilgi[k]["etiket"]) for k in ks], list(ks.values()))
            if ap > en_iyi:
                en_iyi, en_iyi_durum = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"tohum {tohum}: son kayıp {toplam / len(egt):.4f}, en iyi doğrulama klip AP {en_iyi:.3f} "
              f"({time.time() - bas:.0f} sn)", flush=True)
        durumlar.append(en_iyi_durum)
    uyeler = []
    for d in durumlar:
        m = IskeletKavga(F).to(cihaz); m.load_state_dict(d); uyeler.append(m)
    model = Topluluk(uyeler).to(cihaz)
    print("parametre (tek model):", sum(p.numel() for p in uyeler[0].parameters()))

    # eşik: doğrulamada kesinlik >= 0.90 olan en düşük eşik (yoksa F1 en iyi)
    pd = tahmin(model, x[dog], cihaz)
    ks = klip_puanlari(pd, klip[dog])
    yk = np.array([int(bilgi[k]["etiket"]) for k in ks]); sk = np.array(list(ks.values()))
    adaylar = np.linspace(0.05, 0.99, 95)
    sonuc = [(e, olc(yk, sk, e)) for e in adaylar]
    uygun = [e for e, r in sonuc if r["kesinlik"] >= 0.90 and r["tp"] > 0]
    esik = float(min(uygun)) if uygun else float(max(sonuc, key=lambda a: a[1]["f1"])[0])
    yuksek = float(min([e for e, r in sonuc if r["kesinlik"] >= 0.98 and r["tp"] > 0] or [0.97]))
    print("eşik", round(esik, 3), "yüksek eşik", round(yuksek, 3))

    tst = np.where(bolum == "test")[0]
    pt = tahmin(model, x[tst], cihaz)
    ks = klip_puanlari(pt, klip[tst])
    rapor = {"esik": esik, "yuksek_esik": yuksek, "dogrulama_klip": olc(yk, sk, esik), "test": {}}
    gruplar = defaultdict(lambda: ([], []))
    for k, s in ks.items():
        b = bilgi[k]
        for anahtar in ("hepsi", b["veri"], "kategori:" + b["kategori"]):
            gruplar[anahtar][0].append(int(b["etiket"])); gruplar[anahtar][1].append(s)
    for anahtar, (yy, ss) in sorted(gruplar.items()):
        rapor["test"][anahtar] = olc(yy, ss, esik)
        rapor["test"][anahtar]["yuksek_esikte"] = olc(yy, ss, yuksek)
    rapor["test_klip_puanlari"] = {bilgi[k]["video"]: round(s, 4) for k, s in ks.items()}
    (KOK / "test-sonuclari").mkdir(exist_ok=True)
    json.dump(rapor, open(KOK / "test-sonuclari" / "iskelet_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    torch.save({"durumlar": durumlar, "F": F, "T": T, "M": M, "esik": esik, "yuksek_esik": yuksek},
               KOK / "modeller" / "iskelet_kavga.pt")
    for anahtar, r in rapor["test"].items():
        print(f"{anahtar:28s} n={r['n']:4d} kes={r['kesinlik']:.3f} duy={r['duyarlilik']:.3f} "
              f"fp={r['fp']:3d} auc={r.get('auc', '-')}")


if __name__ == "__main__":
    main()
