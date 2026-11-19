"""Alarm düzeyinde ölçüm: yanlış alarm sıklığı ve olay yakalama oranı.

Yöntem
- Test kümesindeki klipler iskelet düzeyinde uç uca eklenerek "kamera akışları" kurulur
  (klipler arasında 0,5 sn boşluk; takip numaraları klipten klibe ayrılır).
- Olumsuz akış: test kümesindeki bütün kavga olmayan klipler (otobüs, sokak, sarılma/alkış/selam,
  hasta düşmesi, günlük hareket). Buradaki her alarm yanlış alarmdır. Kategori kategori ayrı akış.
- Olaylar: kısa kliplerden (otobüs 1-1,6 sn, sokak 2 sn) aynı veri setinden art arda 3 klip tek
  kavga olayı sayılır (~5-6 sn); AIRTLab klibi tek başına olaydır. Her olay ayrı akışta çalıştırılır;
  en az bir alarm çıkarsa "yakalandı".
- Ses: test kliplerinin çoğu sessiz (otobüs, AIRTLab) ya da sesi sonradan eklenmiş müzik (sokak).
  Bu yüzden ses üç senaryoda ölçülür:
    kapali      : yalnız görüntü yolu (B)
    gunluk_ses  : olumsuz kliplere ESC-50'den rastgele gündelik sesler (bebek ağlaması, gülme,
                  alkış, ayak sesi, kapı, öksürük...) bindirilir; kavga olaylarında ses yok.
                  Sesin yanlış alarm ekleyip eklemediğini ölçer.
    ust_sinir   : gunluk_ses + her kavga olayında güçlü bir ses olayı olduğu varsayılır.
                  Ses yolunun en çok ne kadar yakalama ekleyebileceğini gösterir (iyimser sınır).

Kullanım: .venv\\Scripts\\python -m siddet.olc
Çıktı:    test-sonuclari/alarm_olcumu.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .analiz import Analizci
from .birlestir import Ayar, alarmlar
from .poz import Kare, PozDizisi

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "araclar"))
from poz_toplu import poz_yolu  # noqa: E402

SES_ONBELLEK = KOK / "veri" / "ses_olay_puanlari.json"   # yalnız olay puanları, ses değil
ESC_ONBELLEK = KOK / "veri" / "esc50_olay_puanlari.json"
BOSLUK = 0.5
GUNLUK_SINIFLAR = ["crying_baby", "laughing", "clapping", "footsteps", "door_wood_knock", "door_wood_creaks",
                   "coughing", "sneezing", "breathing", "snoring", "drinking_sipping", "keyboard_typing",
                   "mouse_click", "can_opening", "clock_alarm", "vacuum_cleaner", "washing_machine",
                   "toilet_flush", "brushing_teeth", "dog", "cat", "rain", "church_bells", "siren", "car_horn"]


def poisson_ga(k: int, saat: float):
    alt = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    ust = chi2.ppf(0.975, 2 * k + 2) / 2
    return round(alt / saat, 1), round(ust / saat, 1)


def akis_kur(parcalar):
    """parcalar: liste[(PozDizisi, ses_cizelgesi|None)] -> (PozDizisi, ses)"""
    akis = PozDizisi(genislik=0, yukseklik=0, fps=15.0)
    ses = []
    t0, id0 = 0.0, 0
    for p, sc in parcalar:
        for k in p.kareler:
            akis.kareler.append(Kare(k.t + t0, k.ids + id0 if len(k.ids) else k.ids, k.kutular, k.noktalar))
        sure = max(p.sure, p.kareler[-1].t + 1 / 15 if p.kareler else 0)
        if sc:
            ses.extend((t + t0, d) for t, d in sc if t <= sure + 1)
        id0 += 10000
        t0 += sure + BOSLUK
    akis.sure = t0
    return akis, ses


def esc50_cizelgeleri(an: Analizci):
    if ESC_ONBELLEK.exists():
        return json.load(open(ESC_ONBELLEK, encoding="utf-8"))
    kok = KOK / "veri" / "ham" / "ESC-50-master"
    sonuc = {}
    for r in csv.DictReader(open(kok / "meta" / "esc50.csv", encoding="utf-8")):
        dalga = an.ses.ses_oku(kok / "audio" / r["filename"])
        sonuc[r["filename"]] = dict(sinif=r["category"], cizelge=[[t, d] for t, d in an.ses.zaman_cizelgesi(dalga)])
    json.dump(sonuc, open(ESC_ONBELLEK, "w", encoding="utf-8"))
    return sonuc


def ses_esc50_raporu(esc, esik):
    """ESC-50 üzerinde ses olayı algılayıcısının tek başına davranışı (sınıf başına olay çıkma oranı)."""
    sinif = defaultdict(lambda: [0, 0])
    grup_sayac = defaultdict(lambda: defaultdict(int))
    for v in esc.values():
        en = max((max(d.values()) for _, d in v["cizelge"]), default=0)
        sinif[v["sinif"]][0] += 1
        if en >= esik:
            sinif[v["sinif"]][1] += 1
            g = max(v["cizelge"], key=lambda x: max(x[1].values()))[1]
            grup_sayac[v["sinif"]][max(g, key=g.get)] += 1
    oran = {s: dict(klip=n, olay_cikan=k, oran=round(k / n, 3), gruplar=dict(grup_sayac[s])) for s, (n, k) in sinif.items()}
    olumsuz = [s for s in sinif if s not in ("glass_breaking",)]
    toplam_n = sum(sinif[s][0] for s in olumsuz); toplam_k = sum(sinif[s][1] for s in olumsuz)
    return dict(esik=esik, cam_kirilmasi_yakalama=oran.get("glass_breaking"),
                diger_siniflarda_olay_orani=round(toplam_k / toplam_n, 4),
                diger_siniflar_saatte_olay=round(toplam_k / (toplam_n * 5 / 3600), 1),
                en_cok_yanlis=sorted(((s, o["oran"]) for s, o in oran.items() if s != "glass_breaking"),
                                     key=lambda a: -a[1])[:8], sinif=oran)


def main():
    rnd = random.Random(7)
    satir = [r for r in csv.DictReader(open(KOK / "veri" / "liste.csv", encoding="utf-8")) if r["bolum"] == "test"]
    an = Analizci(ses=True)
    ses_puan = json.load(open(SES_ONBELLEK, encoding="utf-8")) if SES_ONBELLEK.exists() else {}
    klipler = []
    for r in satir:
        yol = poz_yolu(r)
        if not yol.exists():
            continue
        if r["video"] not in ses_puan:
            sc = an.ses.video_olaylari(KOK / r["video"])
            ses_puan[r["video"]] = [[t, d] for t, d in sc] if sc else None
        klipler.append((r, PozDizisi.yukle(yol), ses_puan[r["video"]]))
    json.dump(ses_puan, open(SES_ONBELLEK, "w", encoding="utf-8"))
    esc = esc50_cizelgeleri(an)
    gunluk = [v["cizelge"] for v in esc.values() if v["sinif"] in GUNLUK_SINIFLAR]

    olumsuz = [k for k in klipler if k[0]["etiket"] == "0"]
    olumlu = defaultdict(list)
    for k in klipler:
        if k[0]["etiket"] == "1":
            olumlu[k[0]["veri"]].append(k)
    olaylar = []
    for veri, ks in sorted(olumlu.items()):
        rnd.shuffle(ks)
        n = 1 if veri == "airtlab" else 3
        for i in range(0, len(ks) - n + 1, n):
            olaylar.append((veri, ks[i:i + n]))

    # ses bindirme: her olumsuz klibe sabit (tohumlu) bir gündelik ses
    bindirme = {k[0]["video"]: gunluk[random.Random(k[0]["video"]).randrange(len(gunluk))] for k in olumsuz}
    guclu = lambda sure: [[t, {"cigilk_bagirma": 0.9, "cam_kirilmasi": 0.0, "darbe": 0.0}]
                          for t in np.arange(1.0, max(sure, 1.0) + 0.01, 0.5)]

    def olumsuz_ses(r, sc, senaryo):
        if senaryo == "kapali":
            return None
        ek = bindirme[r["video"]]
        return (sc or []) + ek

    # görüntü puanlarını bir kez hesapla (ayar değişse de aynı)
    onbellek = {}

    def gc_al(anahtar, akis):
        if anahtar not in onbellek:
            onbellek[anahtar] = an.goruntu_cizelgesi(akis)
        return onbellek[anahtar]

    def calistir(ayar: Ayar, senaryo: str):
        sonuc = {"kategori": {}}
        toplam_k, toplam_saat = 0, 0.0
        for kat in sorted({k[0]["kategori"] for k in olumsuz}):
            ks = [k for k in olumsuz if k[0]["kategori"] == kat]
            random.Random(1).shuffle(ks)
            akis, ses = akis_kur([(p, olumsuz_ses(r, sc, senaryo)) for r, p, sc in ks])
            al, _ = alarmlar(gc_al(("neg", kat), akis), ses, ayar)
            saat = akis.sure / 3600
            toplam_k += len(al); toplam_saat += saat
            sonuc["kategori"][kat] = dict(klip=len(ks), dakika=round(akis.sure / 60, 2), yanlis_alarm=len(al),
                                          saatte=round(len(al) / saat, 1),
                                          ses_yolundan=sum(1 for a in al if "goruntu_ve_ses" in a["nedenler"]))
        sonuc["olumsuz_toplam"] = dict(dakika=round(toplam_saat * 60, 1), yanlis_alarm=toplam_k,
                                       saatte=round(toplam_k / toplam_saat, 1),
                                       saatte_guven_araligi_95=poisson_ga(toplam_k, toplam_saat))
        yak = defaultdict(lambda: [0, 0])
        for i, (veri, parca) in enumerate(olaylar):
            akis, _ = akis_kur([(p, None) for r, p, sc in parca])
            ses = guclu(akis.sure) if senaryo == "ust_sinir" else []
            al, _ = alarmlar(gc_al(("olay", i), akis), ses, ayar)
            yak[veri][0] += 1
            yak[veri][1] += int(bool(al))
        n = sum(a for a, _ in yak.values()); k = sum(b for _, b in yak.values())
        sonuc["olaylar"] = dict(olay=n, yakalanan=k, duyarlilik=round(k / n, 3),
                                veri_bazli={v: f"{b}/{a}" for v, (a, b) in sorted(yak.items())})
        return sonuc

    temel = an.ayar
    denemeler = []
    for vy in (0.8, 0.85, 0.9, 0.95, 0.98):
        ay = replace(temel, V_YUKSEK=vy)
        for senaryo in ("kapali", "gunluk_ses", "ust_sinir"):
            d = dict(ayar=asdict(ay), senaryo=senaryo, **calistir(ay, senaryo))
            denemeler.append(d)
            print(f"V_YUKSEK={vy:.2f} {senaryo:10s}: olumsuz akışta {d['olumsuz_toplam']['yanlis_alarm']} yanlış alarm / "
                  f"{d['olumsuz_toplam']['dakika']} dk (saatte {d['olumsuz_toplam']['saatte']}), "
                  f"olay yakalama {d['olaylar']['yakalanan']}/{d['olaylar']['olay']} ({d['olaylar']['duyarlilik']})",
                  flush=True)
    rapor = dict(temel_ayar=asdict(temel), olay_sayisi=len(olaylar), ses_esc50=ses_esc50_raporu(esc, temel.A_ESIK),
                 denemeler=denemeler)
    json.dump(rapor, open(KOK / "test-sonuclari" / "alarm_olcumu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    e = rapor["ses_esc50"]
    print("ESC-50: cam kırılması yakalama", e["cam_kirilmasi_yakalama"], "| diğer sınıflarda olay oranı",
          e["diger_siniflarda_olay_orani"], "| en çok yanlış:", e["en_cok_yanlis"])


if __name__ == "__main__":
    main()
