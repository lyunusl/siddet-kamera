"""Ses olayı algılama: hazır PANNs Cnn14 (AudioSet, 527 sınıf), eğitim yok.

Ses yalnızca bellekte işlenir: ffmpeg sesi doğrudan bu sürece akıtır, diske yazılmaz,
saklanmaz. Çıktı yalnızca olaylardır (zaman, olay türü, puan). Konuşma içeriği, kimlik
ya da duygu çıkarılmaz; yalnızca ses olayı sınıfları kullanılır.

Model: Kong ve ark., "PANNs" (2020), ağırlıklar Zenodo 3987831 (CC BY 4.0).
Aşağıdaki `Cnn14` ve `_Blok` sınıfları qiuqiangkong/audioset_tagging_cnn (pytorch/models.py) kodundan
uyarlanmıştır: Copyright (c) 2018-2020 Qiuqiang Kong, MIT Lisansı; lisans metni
ucuncu-taraf/PANNs-LICENSE.MIT.txt dosyasındadır.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

KOK = Path(__file__).resolve().parents[1]
CNN14 = KOK / "modeller" / "Cnn14_mAP=0.431.pth"
ETIKETLER = KOK / "modeller" / "audioset_etiketler.csv"
ORNEK_HIZI = 32000
PENCERE_SN = 2.0
ADIM_SN = 0.5

# Olay grupları -> AudioSet sınıf adları. Duygu değil, fiziksel ses olayı.
OLAY_GRUPLARI = {
    "cigilk_bagirma": ["Screaming", "Yell", "Shout", "Battle cry"],
    "cam_kirilmasi": ["Shatter", "Glass", "Breaking", "Smash, crash"],
    "darbe": ["Slap, smack", "Whack, thwack", "Thump, thud", "Bang"],
}
OLAY_ADLARI = {"cigilk_bagirma": "Çığlık / bağırma", "cam_kirilmasi": "Cam kırılması / kırılma",
               "darbe": "Darbe / tokat sesi"}


class _Blok(nn.Module):
    def __init__(self, gir, cik):
        super().__init__()
        self.conv1 = nn.Conv2d(gir, cik, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(cik, cik, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cik)
        self.bn2 = nn.BatchNorm2d(cik)

    def forward(self, x, havuz=(2, 2)):
        x = Fn.relu_(self.bn1(self.conv1(x)))
        x = Fn.relu_(self.bn2(self.conv2(x)))
        return Fn.avg_pool2d(x, havuz)


class Cnn14(nn.Module):
    """audioset_tagging_cnn/pytorch/models.py içindeki Cnn14'ün çıkarım için sadeleştirilmiş hali
    (aynı katman adları, aynı ağırlık dosyası)."""

    def __init__(self, sinif=527):
        super().__init__()
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann",
                                                 center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=ORNEK_HIZI, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
                                                 ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        kanallar = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _Blok(kanallar[i], kanallar[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, sinif)

    def forward(self, dalga):
        x = self.logmel_extractor(self.spectrogram_extractor(dalga))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(6):
            x = getattr(self, f"conv_block{i + 1}")(x, (2, 2) if i < 5 else (1, 1))
        x = x.mean(3)
        x = x.amax(2) + x.mean(2)
        return torch.sigmoid(self.fc_audioset(Fn.relu_(self.fc1(x))))


class SesOlayi:
    def __init__(self, cihaz=None):
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Cnn14()
        durum = torch.load(CNN14, map_location="cpu", weights_only=False)["model"]
        eksik, fazla = self.model.load_state_dict(durum, strict=False)
        fazla = [k for k in fazla if not k.startswith("spec_augmenter")]
        if eksik or fazla:
            raise RuntimeError(f"Cnn14 ağırlıkları uyuşmadı: {eksik[:3]} {fazla[:3]}")
        self.model.eval().to(self.cihaz)
        adlar = [r["display_name"] for r in csv.DictReader(open(ETIKETLER, encoding="utf-8"))]
        self.indeks = {g: [adlar.index(a) for a in s] for g, s in OLAY_GRUPLARI.items()}

    @staticmethod
    def ses_oku(video: Path) -> np.ndarray | None:
        """Videonun sesini bellekte 32 kHz mono olarak okur. Ses yoksa None."""
        komut = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(ORNEK_HIZI),
                 "-f", "f32le", "pipe:1"]
        r = subprocess.run(komut, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            return None
        return np.frombuffer(r.stdout, np.float32).copy()

    @torch.no_grad()
    def zaman_cizelgesi(self, dalga: np.ndarray, parti: int = 32):
        """Her ADIM_SN'de bir PENCERE_SN'lik ses için grup puanları: liste[(t_orta, {grup: puan})]."""
        n = int(PENCERE_SN * ORNEK_HIZI)
        adim = int(ADIM_SN * ORNEK_HIZI)
        if len(dalga) < n:
            dalga = np.pad(dalga, (0, n - len(dalga)))
        baslar = list(range(0, len(dalga) - n + 1, adim))
        sonuc = []
        for i in range(0, len(baslar), parti):
            b = baslar[i:i + parti]
            x = torch.from_numpy(np.stack([dalga[s:s + n] for s in b])).to(self.cihaz)
            olas = self.model(x).cpu().numpy()
            for s, o in zip(b, olas):
                sonuc.append(((s + n / 2) / ORNEK_HIZI, {g: float(o[ix].max()) for g, ix in self.indeks.items()}))
        return sonuc

    def video_olaylari(self, video: Path):
        dalga = self.ses_oku(video)
        if dalga is None:
            return None
        cizelge = self.zaman_cizelgesi(dalga)
        del dalga  # ham ses tutulmaz
        return cizelge


def olaylari_cikar(cizelge, esik: float):
    """Zaman çizelgesinden olay listesi: art arda eşiği geçen pencereler tek olay."""
    olaylar = []
    if not cizelge:
        return olaylar
    for grup in OLAY_GRUPLARI:
        acik = None
        for t, p in cizelge:
            if p[grup] >= esik:
                if acik is None:
                    acik = dict(tur=grup, ad=OLAY_ADLARI[grup], bas=round(t - PENCERE_SN / 2, 2), puan=0.0)
                acik["bit"] = round(t + PENCERE_SN / 2, 2)
                acik["puan"] = round(max(acik["puan"], p[grup]), 3)
            elif acik is not None:
                olaylar.append(acik); acik = None
        if acik is not None:
            olaylar.append(acik)
    return sorted(olaylar, key=lambda o: o["bas"])
