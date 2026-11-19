"""Kişi iskeletleri: YOLO11-pose + ByteTrack.

Videodan yaklaşık HEDEF_FPS kare/sn örnekler, her karede kişi kutusu, takip numarası ve
17 COCO eklem noktası üretir. Yüz bulanıklaştırma da burada: görüntü diske ya da ekrana
gitmeden önce `yuzleri_bulanik` çağrılır.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

KOK = Path(__file__).resolve().parents[1]
POZ_MODELI = KOK / "modeller" / "yolo11m-pose.pt"
HEDEF_FPS = 15.0

# COCO eklemleri: 0 burun, 1-2 göz, 3-4 kulak, 5-6 omuz, 7-8 dirsek, 9-10 bilek,
# 11-12 kalça, 13-14 diz, 15-16 ayak bileği
SOL_SAG = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass
class Kare:
    t: float                       # saniye
    ids: np.ndarray                # (N,) takip numarası
    kutular: np.ndarray            # (N, 4) xyxy piksel
    noktalar: np.ndarray           # (N, 17, 3) x, y, güven


@dataclass
class PozDizisi:
    kareler: list[Kare] = field(default_factory=list)
    genislik: int = 0
    yukseklik: int = 0
    fps: float = 0.0
    sure: float = 0.0

    def kaydet(self, yol: Path) -> None:
        satir = []
        for i, k in enumerate(self.kareler):
            for j in range(len(k.ids)):
                satir.append(np.concatenate([[i, k.t, k.ids[j]], k.kutular[j], k.noktalar[j].ravel()]))
        dizi = np.array(satir, dtype=np.float32).reshape(-1, 3 + 4 + 51)
        zamanlar = np.array([k.t for k in self.kareler], dtype=np.float32)
        yol.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(yol, satir=dizi, zaman=zamanlar,
                            meta=np.array([self.genislik, self.yukseklik, self.fps, self.sure], dtype=np.float32))

    @classmethod
    def yukle(cls, yol: Path) -> "PozDizisi":
        z = np.load(yol)
        satir, zaman, meta = z["satir"], z["zaman"], z["meta"]
        p = cls(genislik=int(meta[0]), yukseklik=int(meta[1]), fps=float(meta[2]), sure=float(meta[3]))
        for i, t in enumerate(zaman):
            s = satir[satir[:, 0] == i]
            p.kareler.append(Kare(float(t), s[:, 2].astype(int), s[:, 3:7], s[:, 7:].reshape(-1, 17, 3)))
        return p


def model_yukle(yol: Path = POZ_MODELI):
    from ultralytics import YOLO
    return YOLO(str(yol))


def poz_cikar(video: Path, model=None, hedef_fps: float = HEDEF_FPS, kare_isle=None,
              imgsz: int = 640, conf: float = 0.25) -> PozDizisi:
    """Videodaki kişileri takip eder. `kare_isle(goruntu, kare)` verilirse her örneklenen
    karede çağrılır (demo çıktısı için)."""
    model = model or model_yukle()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    p = PozDizisi(genislik=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), yukseklik=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  fps=fps, sure=n / fps if n else 0.0)
    adim = 1.0 / hedef_fps
    sonraki = 0.0
    i = 0
    ilk = True
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / fps
        i += 1
        if t + 1e-6 < sonraki:
            continue
        sonraki += adim
        if sonraki < t:  # gecikme birikmesin
            sonraki = t + adim
        ok, goruntu = cap.retrieve()
        if not ok:
            break
        r = model.track(goruntu, persist=not ilk, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf,
                        classes=[0], verbose=False)[0]
        ilk = False
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            kutular = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.arange(1, len(kutular) + 1)
            kp = r.keypoints.data.cpu().numpy()
            if kp.shape[-1] == 2:
                kp = np.concatenate([kp, np.ones(kp.shape[:-1] + (1,))], -1)
        else:
            kutular, ids, kp = np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0, 17, 3))
        kare = Kare(t, ids, kutular.astype(np.float32), kp.astype(np.float32))
        p.kareler.append(kare)
        if kare_isle is not None:
            kare_isle(goruntu, kare)
    cap.release()
    if not p.sure and p.kareler:
        p.sure = p.kareler[-1].t + 1.0 / fps
    return p


def yuzleri_bulanik(goruntu: np.ndarray, kutular: np.ndarray, noktalar: np.ndarray) -> np.ndarray:
    """Verilen kişilerin baş bölgesini pikselleştirir (yerinde). Tek başına yeterli değildir;
    diske/ekrana giden kareler için `YuzGizleyici` kullanılır."""
    for b in _bas_bolgeleri(kutular, noktalar, goruntu.shape):
        _pikselle(goruntu, b)
    return goruntu


class YuzGizleyici:
    """Diske ya da ekrana giden her karede yüzleri gizler. Üç katman:
    1) takipteki kişilerin baş bölgesi, 2) ayrı, düşük eşikli (0,10) ve yüksek çözünürlüklü bir
    kişi taraması (takipte kaçan, yerde yatan, bulanık kişiler için), 3) YuNet yüz algılayıcı
    (Copyright (c) 2020 Shiqi Yu, MIT; lisans metni ucuncu-taraf/YuNet-LICENSE.txt). Son HAFIZA_SN içinde gizlenen bölgeler de gizlenmeye devam eder."""

    HAFIZA_SN = 0.6

    def __init__(self, poz_modeli=None):
        self.poz = poz_modeli or model_yukle()
        self.yunet = None
        yol = KOK / "modeller" / "yunet_yuz.onnx"
        if yol.exists():
            self.yunet = cv2.FaceDetectorYN.create(str(yol), "", (320, 320), 0.6, 0.3, 5000)
        self.gecmis: list[tuple[float, tuple]] = []

    def uygula(self, img: np.ndarray, t: float, kare: Kare | None = None) -> np.ndarray:
        bolgeler = []
        if kare is not None and len(kare.ids):
            bolgeler += _bas_bolgeleri(kare.kutular, kare.noktalar, img.shape)
        r = self.poz(img, conf=0.10, imgsz=960, classes=[0], verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(r.boxes), 17, 3))
            bolgeler += _bas_bolgeleri(r.boxes.xyxy.cpu().numpy(), kp, img.shape)
        if self.yunet is not None:
            h, w = img.shape[:2]
            self.yunet.setInputSize((w, h))
            _, yuzler = self.yunet.detect(img)
            for f in (yuzler if yuzler is not None else []):
                x, y, fw, fh = f[:4]
                bolgeler.append((x - 0.3 * fw, y - 0.4 * fh, x + 1.3 * fw, y + 1.3 * fh))
        self.gecmis = [(gt, b) for gt, b in self.gecmis if t - gt <= self.HAFIZA_SN] + [(t, b) for b in bolgeler]
        for _, b in self.gecmis:
            _pikselle(img, b)
        return img


def _bas_bolgeleri(kutular, noktalar, sekil):
    sonuc = []
    for kutu, kp in zip(kutular, noktalar):
        x1, y1, x2, y2 = kutu
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        bas = kp[:5]
        gorunur = bas[bas[:, 2] > 0.3]
        if len(gorunur) >= 2:
            cx, cy = gorunur[:, 0].mean(), gorunur[:, 1].mean()
            r = max(np.ptp(gorunur[:, 0]), np.ptp(gorunur[:, 1]), 0.12 * bh, 0.25 * bw) * 1.1
            omuz = kp[5:7]
            if (omuz[:, 2] > 0.3).all():
                r = max(r, 0.45 * np.linalg.norm(omuz[0, :2] - omuz[1, :2]))
            sonuc.append((cx - r, cy - r * 1.2, cx + r, cy + r))
        else:
            # baş görünmüyorsa (yerde yatan, sırtı dönük, bulanık) kutunun üst kısmı ve
            # yatay kişilerde iki yan uç gizlenir
            sonuc.append((x1, y1, x2, y1 + 0.25 * bh))
            if bw > 1.2 * bh:
                sonuc.append((x1, y1, x1 + 0.3 * bw, y2))
                sonuc.append((x2 - 0.3 * bw, y1, x2, y2))
    return sonuc


def _pikselle(img, b):
    h, w = img.shape[:2]
    ax1, ay1, ax2, ay2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
    if ax2 - ax1 < 2 or ay2 - ay1 < 2:
        return
    parca = img[ay1:ay2, ax1:ax2]
    blok = max(4, (ax2 - ax1) // 8)
    kucuk = cv2.resize(parca, (max(1, (ax2 - ax1) // blok), max(1, (ay2 - ay1) // blok)), interpolation=cv2.INTER_AREA)
    img[ay1:ay2, ax1:ax2] = cv2.resize(kucuk, (ax2 - ax1, ay2 - ay1), interpolation=cv2.INTER_NEAREST)


ISKELET = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def iskelet_ciz(goruntu: np.ndarray, noktalar: np.ndarray, renk=(80, 220, 255)) -> None:
    for kp in noktalar:
        for a, b in ISKELET:
            if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                cv2.line(goruntu, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), renk, 2, cv2.LINE_AA)
        for x, y, c in kp[5:]:
            if c > 0.3:
                cv2.circle(goruntu, (int(x), int(y)), 3, renk, -1, cv2.LINE_AA)
