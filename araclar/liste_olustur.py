"""Veri setlerinden tek bir liste (veri/liste.csv) çıkarır: video, etiket, grup, bölüm, kategori.

bolum: egitim / dogrulama / test. Gruplar sızıntıyı önler: aynı çekimin kopyaları aynı bölümde.
- bus: resmî train.txt / test.txt; eğitimin %15'i doğrulama
- airtlab: klip numarası grup (cam1/cam2 aynı olay); numaraya göre %70/%10/%20
- survfight: kaynak YouTube videosu grup; %60/%15/%25
- gmdcsa (düşme + günlük hareket, hepsi olumsuz): denek 1-2 eğitim, denek 3 doğrulama, denek 4 test
- commons (indirildiyse): yalnız test (uzun olumsuz videolar)
"""
import csv
import hashlib
import re
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
HAM = KOK / "veri" / "ham"


def bol(anahtar: str, oranlar=(0.7, 0.1, 0.2)) -> str:
    h = int(hashlib.md5(anahtar.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < oranlar[0]:
        return "egitim"
    if h < oranlar[0] + oranlar[1]:
        return "dogrulama"
    return "test"


def main():
    satir = []
    # Bus Violence
    b = HAM / "bus"
    test = set((b / "test.txt").read_text().split())
    for etiket, klasor in ((1, "Violence"), (0, "NoViolence")):
        for v in sorted((b / klasor).glob("*.mp4")):
            if v.name in test:
                bolum = "test"
            else:
                bolum = "dogrulama" if bol("bus" + v.name, (0.85, 0.15, 0)) == "dogrulama" else "egitim"
            satir.append(dict(veri="bus", video=v.relative_to(KOK).as_posix(), etiket=etiket, grup="bus:" + v.stem,
                              bolum=bolum, kategori="kavga" if etiket else "otobus_normal"))
    # AIRTLab
    a = HAM / "airtlab" / "violence-detection-dataset"
    for etiket, klasor in ((1, "violent"), (0, "non-violent")):
        for kam in ("cam1", "cam2"):
            for v in sorted((a / klasor / kam).glob("*.mp4")):
                grup = f"airtlab:{klasor}:{v.stem}"
                satir.append(dict(veri="airtlab", video=v.relative_to(KOK).as_posix(), etiket=etiket, grup=grup,
                                  bolum=bol(grup), kategori="kavga" if etiket else "sarilma_alkis_selam"))
    # Surveillance Camera Fight
    s = HAM / "survfight"
    url = None
    kaynak = {}
    for ln in (s / "videos.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln.startswith("http"):
            url = ln
        m = re.match(r"(n?o?fi\d+)\s*:", ln)
        if m and url:
            kaynak[m.group(1)] = url
    for etiket, klasor in ((1, "fight"), (0, "noFight")):
        for v in sorted((s / klasor).glob("*.mp4")):
            grup = "survfight:" + kaynak.get(v.stem, v.stem)
            satir.append(dict(veri="survfight", video=v.relative_to(KOK).as_posix(), etiket=etiket, grup=grup,
                              bolum=bol(grup, (0.6, 0.15, 0.25)), kategori="kavga" if etiket else "sokak_normal"))
    # GMDCSA-24
    g = next((HAM / "gmdcsa").glob("*GMDCSA24*"))
    for denek in range(1, 5):
        for tur in ("Fall", "ADL"):
            for v in sorted((g / f"Subject {denek}" / tur).glob("*.mp4")):
                bolum = "egitim" if denek <= 2 else ("dogrulama" if denek == 3 else "test")
                satir.append(dict(veri="gmdcsa", video=v.relative_to(KOK).as_posix(), etiket=0,
                                  grup=f"gmdcsa:{denek}", bolum=bolum,
                                  kategori="hasta_dusmesi" if tur == "Fall" else "gunluk_hareket"))
    # Commons
    c = HAM / "commons" / "kaynaklar.csv"
    if c.exists():
        for r in csv.DictReader(open(c, encoding="utf-8")):
            v = HAM / "commons" / (r["kod"] + ".mp4")
            if v.exists():
                satir.append(dict(veri="commons", video=v.relative_to(KOK).as_posix(), etiket=0,
                                  grup="commons:" + r["kod"], bolum="test", kategori=r["kategori"]))
    with open(KOK / "veri" / "liste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satir[0]))
        w.writeheader()
        w.writerows(satir)
    from collections import Counter
    print(Counter((r["veri"], r["bolum"], r["etiket"]) for r in satir))


if __name__ == "__main__":
    main()
