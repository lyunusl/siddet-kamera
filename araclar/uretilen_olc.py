"""Yapay zekâ ile üretilmiş zor olumsuz klipleri (veri/ham/uretilen) analiz eder.

Her klip için: en yüksek görüntü puanı, seçili ayarda alarm sayısı; ayrıca hepsi uç uca
eklenmiş tek akışta farklı V_YUKSEK eşiklerinde yanlış alarm sayısı.
Klipler eğitime girmez; yalnız ölçüm içindir.
Çıktı: test-sonuclari/uretilen_olcum.json
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from siddet.analiz import Analizci  # noqa: E402
from siddet.birlestir import alarmlar  # noqa: E402
from siddet.olc import akis_kur  # noqa: E402
from siddet.poz import PozDizisi  # noqa: E402

KLASOR = KOK / "veri" / "ham" / "uretilen"


def main():
    an = Analizci(ses=False)
    kategoriler = {s.split("\t")[0]: s.split("\t")[1] for s in (KLASOR / "isler.tsv").read_text(encoding="utf-8").splitlines() if s}
    klipler, sonuc = [], {"klipler": {}, "akis": {}}
    for v in sorted(KLASOR.glob("u*.mp4")):
        poz_yol = KOK / "veri" / "poz" / "uretilen" / (v.stem + ".npz")
        if poz_yol.exists():
            p = PozDizisi.yukle(poz_yol)
            r, _ = an.analiz_et(v, p)
        else:
            r, p = an.analiz_et(v)
            p.kaydet(poz_yol)
        klipler.append(p)
        en = max((x["goruntu_yumusak"] for x in r["puanlar"]), default=0)
        sonuc["klipler"][v.stem] = dict(kategori=kategoriler.get(v.stem), sure=r["sure_sn"], en_yuksek_goruntu=round(en, 3),
                                        ort_kisi=round(sum(x["kisi"] for x in r["puanlar"]) / max(len(r["puanlar"]), 1), 1),
                                        alarm=len(r["alarmlar"]), alarmlar=r["alarmlar"])
        print(f"{v.stem:24s} kişi≈{sonuc['klipler'][v.stem]['ort_kisi']:4.1f} en yüksek {en:.2f} alarm {len(r['alarmlar'])}")
    akis, _ = akis_kur([(p, None) for p in klipler])
    gc = an.goruntu_cizelgesi(akis)
    for vy in (0.8, 0.85, 0.9, 0.95, 0.98):
        ay = replace(an.ayar, V_YUKSEK=vy)
        al, _ = alarmlar(gc, [], ay)
        sonuc["akis"][f"{vy:.2f}"] = dict(dakika=round(akis.sure / 60, 2), yanlis_alarm=len(al),
                                          saatte=round(len(al) / (akis.sure / 3600), 1))
        print(f"V_YUKSEK {vy:.2f}: {len(al)} yanlış alarm / {akis.sure / 60:.1f} dk")
    sonuc["ayar"] = asdict(an.ayar)
    json.dump(sonuc, open(KOK / "test-sonuclari" / "uretilen_olcum.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
