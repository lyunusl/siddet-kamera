"""Yapay zekâ ile üretilmiş iki bayılma klibinde (15 sn) düşme / bayılma sonuçlarını kaydeder.

Klipler: veri/ham/uretilen/b1_hastane_bayilma.mp4, b2_okul_bayilma.mp4 (depoda yok; README'deki promptlarla
üretildi). Okul klibinde kişi çok küçük olduğu için iki çözünürlük denenir (640, 1280).
Çıktı: test-sonuclari/bayilma_olcumu.json
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from siddet.dusme import DusmeAyar, olaylari_bul  # noqa: E402
from siddet.poz import PozDizisi, model_yukle, poz_cikar  # noqa: E402

KLIPLER = ["b1_hastane_bayilma", "b2_okul_bayilma"]


def main():
    model = model_yukle()
    temel = DusmeAyar()
    sonuc = {"ayar": asdict(temel), "klipler": {}}
    for ad in KLIPLER:
        video = KOK / "veri" / "ham" / "uretilen" / (ad + ".mp4")
        for imgsz in (640, 1280):
            onbellek = KOK / "veri" / "poz" / "uretilen" / f"{ad}_{imgsz}.npz"
            if onbellek.exists():
                p = PozDizisi.yukle(onbellek)
            else:
                p = poz_cikar(video, model, imgsz=imgsz)
                p.kaydet(onbellek)
            kayit = {"kisi_gorulen_kare": sum(1 for k in p.kareler if len(k.ids)), "kare": len(p.kareler)}
            for bayilma in (10.0, 4.0):
                kayit[f"bayilma_{bayilma:g}sn"] = olaylari_bul(p, replace(temel, BAYILMA_SN=bayilma))
            sonuc["klipler"][f"{ad}@{imgsz}"] = kayit
            print(ad, imgsz, {k: ([(o["tur"], o["t"], o.get("hareketsiz_sn")) for o in v] if k.startswith("bayilma") else v)
                              for k, v in kayit.items()})
    (KOK / "test-sonuclari").mkdir(exist_ok=True)
    json.dump(sonuc, open(KOK / "test-sonuclari" / "bayilma_olcumu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
