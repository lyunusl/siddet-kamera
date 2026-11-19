"""Düşme / hareketsizlik modülünü ölçer.

1) GMDCSA-24: denek 1-3 ile eşik taraması (ayar), denek 4 ile test. Düşme klibinde DUSME olayı çıkarsa
   yakalandı; günlük hareket (ADL) klibinde çıkarsa yanlış alarm.
2) Diğer test klipleri (otobüs, AIRTLab, sokak; kavga olmayanlar) ve üretilmiş klipler: yanlış düşme.
3) Hareketsizlik: GMDCSA klipleri kısa (5-8 sn), 10/30 sn'lik süre sınanamaz; bunun yerine düşme
   kliplerinin sonunda yerde hareketsiz kalma ve ADL'de yatakta hareketsiz yatma 1,5 sn'lik vekil
   eşikle sayılır.
Çıktı: test-sonuclari/dusme_olcumu.json
"""
import csv
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "araclar"))
from poz_toplu import poz_yolu  # noqa: E402
from siddet.dusme import DusmeAyar, olaylari_bul  # noqa: E402
from siddet.poz import PozDizisi  # noqa: E402


def main():
    satir = list(csv.DictReader(open(KOK / "veri" / "liste.csv", encoding="utf-8")))
    g = [(r, PozDizisi.yukle(poz_yolu(r))) for r in satir if r["veri"] == "gmdcsa"]
    for r, _ in g:
        r["denek"] = int(r["grup"].split(":")[1])

    def degerlendir(ayar, kume):
        tp = fn = fp = tn = 0
        for r, p in kume:
            var = any(o["tur"] == "dusme" for o in olaylari_bul(p, ayar))
            if r["kategori"] == "hasta_dusmesi":
                tp += var; fn += not var
            else:
                fp += var; tn += not var
        kes = tp / max(tp + fp, 1); duy = tp / max(tp + fn, 1)
        return dict(tp=tp, fn=fn, fp=fp, tn=tn, kesinlik=round(kes, 3), duyarlilik=round(duy, 3),
                    f1=round(2 * kes * duy / max(kes + duy, 1e-9), 3))

    ayar_kume = [x for x in g if x[0]["denek"] <= 3]
    test_kume = [x for x in g if x[0]["denek"] == 4]
    en_iyi = None
    for aci, kutu, oran, hiz, kalici in itertools.product((60, 65), (1.1, 1.3), (0.2, 0.25, 0.3), (0.6, 0.8), (0.4, 0.8)):
        a = DusmeAyar(YATAY_ACI=aci, YATAY_KUTU=kutu, DUSUS_ORAN=oran, DUSUS_HIZ_SN=hiz, YATMA_KALICI=kalici)
        s = degerlendir(a, ayar_kume)
        # yanlış alarmı daha pahalı say: F1 eşitse az yanlış alarm
        anahtar = (s["f1"] - 0.02 * s["fp"], -s["fp"])
        if en_iyi is None or anahtar > en_iyi[0]:
            en_iyi = (anahtar, a, s)
    _, ayar, s_ayar = en_iyi
    s_test = degerlendir(ayar, test_kume)
    print("seçilen:", {k: v for k, v in asdict(ayar).items() if k in ("YATAY_ACI", "YATAY_KUTU", "DUSUS_ORAN", "DUSUS_HIZ_SN", "YATMA_KALICI")})
    print("denek 1-3 (ayar):", s_ayar)
    print("denek 4 (test):  ", s_test)

    # diğer olumsuzlar: test bölümündeki kavga olmayan klipler + üretilmiş klipler
    diger = {}
    for r in satir:
        if r["bolum"] != "test" or r["veri"] == "gmdcsa":
            continue
        p = PozDizisi.yukle(poz_yolu(r))
        n = sum(1 for o in olaylari_bul(p, ayar) if o["tur"] == "dusme")
        k = f"{r['veri']}:{'kavga' if r['etiket'] == '1' else 'normal'}"
        d = diger.setdefault(k, dict(klip=0, dusme_cikan=0))
        d["klip"] += 1; d["dusme_cikan"] += int(n > 0)
    uretilen = {}
    for f in sorted((KOK / "veri" / "poz" / "uretilen").glob("*.npz")):
        p = PozDizisi.yukle(f)
        uretilen[f.stem] = [dict(o, kp=None) if False else o for o in olaylari_bul(p, replace(ayar, BAYILMA_SN=3.0))]
    print("diğer test klipleri:", diger)
    print("üretilmiş klipler:", {k: [o["tur"] for o in v] for k, v in uretilen.items()})

    # hareketsizlik vekil ölçüm (1,5 sn)
    vekil = replace(ayar, BAYILMA_SN=1.5, HAREKETSIZ_SN=1.5)
    say = {"dusme_sonrasi_bayilma": [0, 0], "adl_hareketsiz_yatma": [0, 0]}
    for r, p in g:
        o = olaylari_bul(p, vekil)
        if r["kategori"] == "hasta_dusmesi":
            say["dusme_sonrasi_bayilma"][0] += 1
            say["dusme_sonrasi_bayilma"][1] += any(x["tur"] == "bayilma" for x in o)
        else:
            say["adl_hareketsiz_yatma"][0] += 1
            say["adl_hareketsiz_yatma"][1] += any(x["tur"] in ("hareketsiz", "bayilma") for x in o)
    print("vekil hareketsizlik (1,5 sn):", say)
    json.dump(dict(ayar=asdict(ayar), gmdcsa_ayar_denek_1_3=s_ayar, gmdcsa_test_denek_4=s_test, diger_test=diger,
                   uretilen=uretilen, vekil_hareketsizlik_1_5sn=say),
              open(KOK / "test-sonuclari" / "dusme_olcumu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
