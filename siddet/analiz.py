"""Bir video dosyasından kavga / şiddet alarm zaman çizelgesi üretir.

Kullanım:
    .venv\\Scripts\\python -m siddet.analiz VIDEO.mp4 [--cikti KLASOR] [--video-yaz] [--sessiz]

Çıktı klasörü (varsayılan cikti/<video adı>/):
    zaman_cizelgesi.json   alarmlar, ses olayları, 0,5 sn aralıklı puanlar
    zaman_cizelgesi.html   puan grafiği ve alarm tablosu
    isaretli.mp4           (--video-yaz) yüzler pikselli, iskeletli, puan çubuklu, SESSİZ video

Yüzler her zaman pikselli. Ses diske yazılmaz, çıktı videoya konmaz; yalnız olay üretilir.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .birlestir import Ayar, alarmlar, ayar_sozluk
from .dusme import OLAY_ADLARI as DUSME_ADLARI, DusmeAyar, olaylari_bul
from .model import IskeletKavga, Topluluk
from .ozellik import PENCERE_SN, pencere_baslangiclari, pencere_tensoru
from .poz import PozDizisi, YuzGizleyici, iskelet_ciz, model_yukle, poz_cikar
from .ses import OLAY_ADLARI, SesOlayi, olaylari_cikar

KOK = Path(__file__).resolve().parents[1]
ISKELET_MODELI = KOK / "modeller" / "iskelet_kavga.pt"
ADIM = 0.5


class Analizci:
    def __init__(self, ses: bool = True, ayar: Ayar | None = None, dusme_ayar: DusmeAyar | None = None,
                 imgsz: int = 640):
        self.imgsz = imgsz
        self.dusme_ayar = dusme_ayar or DusmeAyar()
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = torch.load(ISKELET_MODELI, map_location="cpu", weights_only=False)
        uyeler = []
        for durum in d["durumlar"]:
            m = IskeletKavga(d["F"]); m.load_state_dict(durum); uyeler.append(m)
        self.kavga = Topluluk(uyeler).to(self.cihaz).eval()
        # V_ESIK: doğrulamada klip kesinliği >= 0,90 olan eşik. V_YUKSEK: test-sonuclari/alarm_olcumu.json
        # taramasında seçilen çalışma noktası (0,95: olumsuz akışta saatte ~17 yanlış alarm, yakalama ~%45).
        self.ayar = ayar or Ayar(V_ESIK=float(d["esik"]), V_YUKSEK=0.95)
        self.poz_modeli = None
        self.ses = SesOlayi(self.cihaz) if ses else None

    @torch.no_grad()
    def goruntu_cizelgesi(self, p: PozDizisi):
        sure = max(p.sure, p.kareler[-1].t if p.kareler else 0.0)
        baslar = pencere_baslangiclari(sure, ADIM)
        xs, kisi = [], []
        for t0 in baslar:
            x, n = pencere_tensoru(p, t0)
            xs.append(x); kisi.append(n)
        if not xs:
            return []
        x = torch.from_numpy(np.stack(xs)).to(self.cihaz)
        puan = torch.sigmoid(self.kavga(x)).cpu().numpy()
        # iki kişiden az görünen pencerede kavga sayılmaz
        puan = np.where(np.array(kisi) >= 2, puan, puan * 0.2)
        return [(round(t0 + PENCERE_SN / 2, 2), float(s), int(n)) for t0, s, n in zip(baslar, puan, kisi)]

    def analiz_et(self, video: Path, poz: PozDizisi | None = None):
        bas = time.time()
        if poz is None:
            self.poz_modeli = self.poz_modeli or model_yukle()
            poz = poz_cikar(video, self.poz_modeli, imgsz=self.imgsz)
        gc = self.goruntu_cizelgesi(poz)
        sc = self.ses.video_olaylari(video) if self.ses else None
        olay, ayrinti = alarmlar(gc, sc or [], self.ayar)
        dusme = olaylari_bul(poz, self.dusme_ayar)
        # düşme anına denk gelen kavga alarmı: yere düşen ve yardım edilen kişi de "temas + ani hareket"
        # gibi görünür. Alarm silinmez (kavgada da düşülür) ama işaretlenir.
        for a in olay:
            a["dusme_ile_cakisiyor"] = any(d["tur"] == "dusme" and a["bas"] - 3 <= d["t"] <= a["bit"] + 3 for d in dusme)
        return dict(video=Path(video).name, sure_sn=round(poz.sure, 2), analiz_sn=round(time.time() - bas, 1),
                    ses_var=sc is not None, ayar=ayar_sozluk(self.ayar), alarmlar=olay,
                    ses_olaylari=olaylari_cikar(sc, self.ayar.A_ESIK) if sc else [],
                    dusme_olaylari=dusme,
                    dusme_ayar={k: v for k, v in self.dusme_ayar.__dict__.items()},
                    puanlar=ayrinti), poz


def isaretli_video_yaz(video: Path, poz: PozDizisi, sonuc: dict, hedef: Path, fps: float = 15.0, poz_modeli=None):
    gizle = YuzGizleyici(poz_modeli)
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(3)), int(cap.get(4))
    olcek = min(1.0, 960 / max(W, 1))
    w, h = int(W * olcek) // 2 * 2, int(H * olcek) // 2 * 2
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           str(hedef)], stdin=subprocess.PIPE)
    zaman = np.array([k.t for k in poz.kareler])
    puan_t = np.array([p["t"] for p in sonuc["puanlar"]]) if sonuc["puanlar"] else np.zeros(1)
    alarm_araliklari = [(a["bas"] - PENCERE_SN / 2, a["bit"] + 0.5) for a in sonuc["alarmlar"]]
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, sonraki = 0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / vfps; i += 1
        if t + 1e-6 < sonraki:
            continue
        sonraki += 1.0 / fps
        _, img = cap.retrieve()
        if len(zaman):
            k = poz.kareler[int(np.argmin(np.abs(zaman - t)))]
            gizle.uygula(img, t, k if abs(k.t - t) < 0.1 else None)
            img = sahne_pikselle(cv2.resize(img, (w, h)))
            iskelet_ciz(img, k.noktalar * np.array([olcek, olcek, 1], np.float32))
        else:
            gizle.uygula(img, t)
            img = sahne_pikselle(cv2.resize(img, (w, h)))
        if sonuc["puanlar"]:
            j = int(np.argmin(np.abs(puan_t - t)))
            p = sonuc["puanlar"][j]
            _cubuk(img, 12, h - 46, "GORUNTU", p["goruntu_yumusak"], sonuc["ayar"]["V_ESIK"])
            _cubuk(img, 12, h - 24, "SES", p["ses"], sonuc["ayar"]["A_ESIK"])
        yazi = []
        if any(a <= t <= b for a, b in alarm_araliklari):
            yazi.append("ALARM: KAVGA SUPHESI")
        for o in sonuc.get("dusme_olaylari", []):
            if o["tur"] == "dusme" and o["t"] - 0.3 <= t <= o["t"] + 3:
                yazi.append("ALARM: DUSME")
            elif o["tur"] == "bayilma" and o["t"] <= t <= o["t"] + 4:
                yazi.append("ALARM: BAYILMA SUPHESI")
            elif o["tur"] == "hareketsiz" and o["t"] <= t <= o["t"] + 4:
                yazi.append("ALARM: HAREKETSIZ YATIYOR")
        if yazi:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 230), 6)
            for n, y_ in enumerate(dict.fromkeys(yazi)):
                cv2.putText(img, y_, (14, 34 + 30 * n), cv2.FONT_HERSHEY_DUPLEX, 0.9, (40, 40, 230), 2, cv2.LINE_AA)
        cv2.putText(img, f"{t:5.1f} sn", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    cap.release()
    ff.stdin.close(); ff.wait()


def sahne_pikselle(img, blok: int = 12):
    """Bütün sahneyi kaba piksellere indirir: yüz algılayıcının kaçırdığı kişiler de tanınmaz.
    İskelet ve puanlar bunun üstüne çizilir."""
    h, w = img.shape[:2]
    k = cv2.resize(img, (max(1, w // blok), max(1, h // blok)), interpolation=cv2.INTER_AREA)
    return cv2.resize(k, (w, h), interpolation=cv2.INTER_NEAREST)


def _cubuk(img, x, y, ad, deger, esik):
    cv2.putText(img, ad, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    x0, gen = x + 80, 200
    cv2.rectangle(img, (x0, y), (x0 + gen, y + 14), (60, 60, 60), -1)
    renk = (40, 40, 230) if deger >= esik else (80, 200, 255)
    cv2.rectangle(img, (x0, y), (x0 + int(gen * min(deger, 1)), y + 14), renk, -1)
    cv2.line(img, (x0 + int(gen * esik), y - 2), (x0 + int(gen * esik), y + 16), (255, 255, 255), 1)


def html_yaz(sonuc: dict, hedef: Path):
    p = sonuc["puanlar"]
    sure = max(sonuc["sure_sn"], p[-1]["t"] if p else 1)
    W, H = 1000, 220
    def X(t): return 50 + (W - 70) * t / max(sure, 1e-6)
    def Y(v, ust, alt): return alt - (alt - ust) * v
    g = " ".join(f"{X(q['t']):.1f},{Y(q['goruntu_yumusak'], 20, 110):.1f}" for q in p)
    s = " ".join(f"{X(q['t']):.1f},{Y(q['ses'], 130, 200):.1f}" for q in p)
    alarm = "".join(f'<rect x="{X(a["bas"] - 0.8):.1f}" y="14" width="{max(X(a["bit"] + 0.8) - X(a["bas"] - 0.8), 3):.1f}" '
                    f'height="192" fill="#ff3b3b" opacity=".18"/>' for a in sonuc["alarmlar"])
    ay = sonuc["ayar"]
    satirlar = "".join(
        f"<tr><td>{a['bas']:.1f} – {a['bit']:.1f} sn</td><td>{'Görüntü + ses' if 'goruntu_ve_ses' in a['nedenler'] else 'Yalnız görüntü (çok yüksek)'}{' · düşmeyle çakışıyor' if a.get('dusme_ile_cakisiyor') else ''}</td>"
        f"<td>{a['en_goruntu']:.2f}</td><td>{a['en_ses']:.2f}</td><td>{', '.join(OLAY_ADLARI.get(x, x) for x in a['ses_olaylari']) or '—'}</td></tr>"
        for a in sonuc["alarmlar"]) or '<tr><td colspan="5">Alarm yok</td></tr>'
    olaylar = "".join(f"<li>{o['bas']:.1f}–{o['bit']:.1f} sn · {o['ad']} · {o['puan']:.2f}</li>" for o in sonuc["ses_olaylari"]) \
        or "<li>Ses olayı yok" + ("" if sonuc["ses_var"] else " (videoda ses kanalı yok)") + "</li>"
    dusmeler = "".join(f"<li>{o['t']:.1f} sn · {DUSME_ADLARI[o['tur']]}" + (f" · {o['hareketsiz_sn']:.0f} sn hareketsiz" if 'hareketsiz_sn' in o else "")
                       + "</li>" for o in sonuc.get("dusme_olaylari", [])) or "<li>Yok</li>"
    dusme_isaret = "".join(f'<line x1="{X(o["t"]):.1f}" x2="{X(o["t"]):.1f}" y1="14" y2="206" stroke="#ffb547" stroke-width="2"/>'
                           f'<text x="{X(o["t"]) + 3:.1f}" y="12" class="k" fill="#ffb547">{DUSME_ADLARI[o["tur"]].split(" (")[0]}</text>'
                           for o in sonuc.get("dusme_olaylari", []))
    alarm += dusme_isaret
    tik = "".join(f'<text x="{X(t):.0f}" y="216" class="k">{t:.0f}</text>' for t in np.arange(0, sure + 0.01, max(1, round(sure / 10))))
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Şiddet Kamera · {sonuc['video']}</title><style>
body{{background:#0b0e13;color:#e8edf3;font:15px/1.5 system-ui,Segoe UI,sans-serif;margin:0;padding:24px}}
h1{{font-size:20px;margin:0 0 4px;letter-spacing:.04em}} .alt{{color:#8a96a8;margin-bottom:18px}}
.kart{{background:#121822;border:1px solid #233044;border-radius:10px;padding:16px;margin-bottom:16px;overflow-x:auto}}
svg{{width:100%;max-width:{W}px;height:auto}} .k{{fill:#8a96a8;font-size:11px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #233044;padding:6px 8px;text-align:left}}
th{{color:#ffb547;font-weight:600}} .rozet{{display:inline-block;background:#ff3b3b;color:#fff;border-radius:4px;padding:1px 8px;font-weight:700}}
</style></head><body>
<h1>ŞİDDET VE DÜŞME KAMERASI · OLAY ZAMAN ÇİZELGESİ</h1>
<div class="alt">{sonuc['video']} · {sonuc['sure_sn']:.1f} sn · <span class="rozet">{len(sonuc['alarmlar'])} kavga alarmı</span> <span class="rozet" style="background:#ffb547;color:#111">{len(sonuc.get('dusme_olaylari', []))} düşme/hareketsizlik</span> ·
yüzler gizli · ses kaydedilmedi, yalnız olay üretildi · prototip, araştırma amaçlı</div>
<div class="kart"><svg viewBox="0 0 {W} {H + 10}">{alarm}
<text x="4" y="24" class="k">görüntü</text><text x="4" y="140" class="k">ses</text>
<line x1="50" x2="{W - 20}" y1="{Y(ay['V_ESIK'], 20, 110):.1f}" y2="{Y(ay['V_ESIK'], 20, 110):.1f}" stroke="#ffb547" stroke-dasharray="4 4"/>
<line x1="50" x2="{W - 20}" y1="{Y(ay['V_YUKSEK'], 20, 110):.1f}" y2="{Y(ay['V_YUKSEK'], 20, 110):.1f}" stroke="#ff3b3b" stroke-dasharray="2 4"/>
<line x1="50" x2="{W - 20}" y1="{Y(ay['A_ESIK'], 130, 200):.1f}" y2="{Y(ay['A_ESIK'], 130, 200):.1f}" stroke="#ffb547" stroke-dasharray="4 4"/>
<polyline fill="none" stroke="#5ec8ff" stroke-width="2" points="{g}"/>
<polyline fill="none" stroke="#9cff7a" stroke-width="2" points="{s}"/>{tik}</svg>
<div class="alt">Mavi: görüntü kavga puanı (yumuşatılmış) · yeşil: en güçlü ses olayı puanı · kesikli: eşikler · kırmızı alan: alarm</div></div>
<div class="kart"><table><tr><th>Zaman</th><th>Neden</th><th>Görüntü</th><th>Ses</th><th>Ses olayı</th></tr>{satirlar}</table></div>
<div class="kart"><b>Düşme / bayılma / hareketsizlik</b><ul>{dusmeler}</ul>
<div class="alt">Bayılma şüphesi: düşmeden sonra {sonuc.get('dusme_ayar', {}).get('BAYILMA_SN', 10):.0f} sn hareketsiz · Hareketsiz yatıyor: düşme görülmeden {sonuc.get('dusme_ayar', {}).get('HAREKETSIZ_SN', 30):.0f} sn (yatak bölgeleri hariç)</div></div>
<div class="kart"><b>Ses olayları</b><ul>{olaylar}</ul></div>
</body></html>"""
    hedef.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Videodan kavga/şiddet alarm zaman çizelgesi")
    ap.add_argument("video", type=Path)
    ap.add_argument("--cikti", type=Path)
    ap.add_argument("--video-yaz", action="store_true", help="yüzleri pikselli işaretli video da yaz")
    ap.add_argument("--sessiz", action="store_true", help="ses olayı algılamayı kapat")
    ap.add_argument("--bayilma-sn", type=float, default=10.0, help="düşmeden sonra kaç sn hareketsizlikte bayılma şüphesi")
    ap.add_argument("--hareketsiz-sn", type=float, default=30.0, help="düşme görülmeden kaç sn yatay hareketsizlikte uyarı")
    ap.add_argument("--imgsz", type=int, default=640, help="poz modeli çözünürlüğü; uzak/küçük kişiler için 1280")
    ap.add_argument("--yatak", type=Path, help="yatak/sedye bölgeleri JSON: [[[x,y],...],...] 0-1 göreli")
    a = ap.parse_args()
    cikti = a.cikti or KOK / "cikti" / a.video.stem
    cikti.mkdir(parents=True, exist_ok=True)
    yataklar = json.load(open(a.yatak, encoding="utf-8")) if a.yatak else []
    an = Analizci(ses=not a.sessiz, dusme_ayar=DusmeAyar(BAYILMA_SN=a.bayilma_sn, HAREKETSIZ_SN=a.hareketsiz_sn,
                                                            yatak_bolgeleri=yataklar), imgsz=a.imgsz)
    sonuc, poz = an.analiz_et(a.video)
    json.dump(sonuc, open(cikti / "zaman_cizelgesi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    html_yaz(sonuc, cikti / "zaman_cizelgesi.html")
    if a.video_yaz:
        isaretli_video_yaz(a.video, poz, sonuc, cikti / "isaretli.mp4", poz_modeli=an.poz_modeli)
    print(f"{sonuc['video']}: {sonuc['sure_sn']:.1f} sn, {len(sonuc['alarmlar'])} alarm, analiz {sonuc['analiz_sn']} sn")
    for al in sonuc["alarmlar"]:
        print(f"  ALARM {al['bas']:.1f}-{al['bit']:.1f} sn  {'+'.join(al['nedenler'])}  görüntü {al['en_goruntu']:.2f} ses {al['en_ses']:.2f}")
    for o in sonuc["dusme_olaylari"]:
        print(f"  {DUSME_ADLARI[o['tur']].replace('i', 'İ').upper()} {o['t']:.1f} sn" + (f"  ({o['hareketsiz_sn']:.0f} sn hareketsiz)" if "hareketsiz_sn" in o else ""))
    print("çıktı:", cikti)


if __name__ == "__main__":
    main()
