"""liste.csv'deki her video için iskelet dosyası (veri/poz/<veri>/<ad>.npz) üretir. Var olanı atlar."""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siddet.poz import model_yukle, poz_cikar  # noqa: E402

KOK = Path(__file__).resolve().parents[1]


def poz_yolu(r) -> Path:
    v = Path(r["video"])
    ad = v.stem if r["veri"] != "airtlab" else f"{v.parent.parent.name}_{v.parent.name}_{v.stem}"
    if r["veri"] == "gmdcsa":
        ad = f"{v.parent.parent.name.replace(' ', '')}_{v.parent.name}_{v.stem}"
    return KOK / "veri" / "poz" / r["veri"] / (ad + ".npz")


def main():
    satir = list(csv.DictReader(open(KOK / "veri" / "liste.csv", encoding="utf-8")))
    argv = sys.argv[1:]
    parca = next((a for a in argv if "/" in a), "0/1")
    pi, pn = map(int, parca.split("/"))
    sadece = {a for a in argv if "/" not in a}
    model = model_yukle()
    bas = time.time()
    yapilan = 0
    for i, r in enumerate(satir):
        if (sadece and r["veri"] not in sadece) or i % pn != pi:
            continue
        hedef = poz_yolu(r)
        if hedef.exists():
            continue
        try:
            p = poz_cikar(KOK / r["video"], model)
        except Exception as e:  # bozuk video ya da bellek: atla, sonra yeniden denenir
            print("HATA", r["video"], type(e).__name__, e, flush=True)
            continue
        p.kaydet(hedef)
        yapilan += 1
        if yapilan % 50 == 0:
            print(f"{i + 1}/{len(satir)} {r['veri']} {time.time() - bas:.0f} sn", flush=True)
    print("bitti", yapilan, f"{time.time() - bas:.0f} sn")


if __name__ == "__main__":
    main()
