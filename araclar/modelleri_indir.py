"""Hazır modelleri resmî kaynaklarından modeller/ klasörüne indirir ve SHA-256 özetini doğrular.

    python araclar/modelleri_indir.py

İskelet kavga sınıflandırıcısı (modeller/iskelet_kavga.pt) burada yoktur: araştırma lisanslı verilerle
eğitildiği için dağıtılmaz. Kendiniz eğitin (README → "Yeniden üretmek").
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
MODELLER = [
    ("yolo11m-pose.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
     "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"),
    ("Cnn14_mAP=0.431.pth", "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
     "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"),
    ("audioset_etiketler.csv", "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
     "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429"),
    ("yunet_yuz.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
]


def ozet(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def main() -> int:
    hedef_klasor = KOK / "modeller"
    hedef_klasor.mkdir(exist_ok=True)
    hata = 0
    for ad, url, beklenen in MODELLER:
        hedef = hedef_klasor / ad
        if hedef.exists() and ozet(hedef) == beklenen:
            print("var:", ad)
            continue
        print("indiriliyor:", ad)
        gecici = hedef.with_suffix(hedef.suffix + ".indir")
        urllib.request.urlretrieve(url, gecici)
        if ozet(gecici) != beklenen:
            print("  ÖZET UYMADI, silindi:", ad)
            gecici.unlink()
            hata += 1
            continue
        gecici.replace(hedef)
        print("  tamam")
    return 1 if hata else 0


if __name__ == "__main__":
    sys.exit(main())
