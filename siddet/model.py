"""Küçük iskelet sınıflandırıcı: kişi kodlayıcı + kişiler üstünde havuzlama + zamansal evrişim + GRU.

Girdi (B, T, M, F), çıktı kavga logiti (B,). Parametre sayısı ~150 bin; işlemcide de hızlı.
"""
import torch
from torch import nn


class IskeletKavga(nn.Module):
    def __init__(self, F: int = 93, gizli: int = 96, birak: float = 0.3):
        super().__init__()
        self.kisi = nn.Sequential(nn.Linear(F, gizli), nn.GELU(), nn.LayerNorm(gizli),
                                  nn.Linear(gizli, gizli), nn.GELU())
        self.zaman = nn.Sequential(
            nn.Conv1d(2 * gizli, gizli, 3, padding=1), nn.GELU(), nn.Dropout(birak),
            nn.Conv1d(gizli, gizli, 3, padding=2, dilation=2), nn.GELU())
        self.gru = nn.GRU(gizli, gizli // 2, batch_first=True, bidirectional=True)
        self.cikis = nn.Sequential(nn.Dropout(birak), nn.Linear(2 * gizli, 1))

    def forward(self, x):
        B, T, M, F = x.shape
        maske = x[..., -1:] > 0.5                       # görünürlük bayrağı
        h = self.kisi(x)                                # (B,T,M,H)
        enbuyuk = h.masked_fill(~maske, -1e4).amax(2)
        enbuyuk = torch.where(maske.any(2), enbuyuk, torch.zeros_like(enbuyuk))
        ort = (h * maske).sum(2) / maske.sum(2).clamp(min=1)
        z = torch.cat([enbuyuk, ort], -1).transpose(1, 2)  # (B,2H,T)
        z = self.zaman(z).transpose(1, 2)
        z, _ = self.gru(z)
        z = torch.cat([z.amax(1), z.mean(1)], -1)
        return self.cikis(z).squeeze(-1)


class Topluluk(nn.Module):
    """Farklı tohumlarla eğitilmiş modellerin olasılık ortalaması; çıktı yine logit."""

    def __init__(self, modeller):
        super().__init__()
        self.modeller = nn.ModuleList(modeller)

    def forward(self, x):
        p = torch.stack([torch.sigmoid(m(x)) for m in self.modeller]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))
