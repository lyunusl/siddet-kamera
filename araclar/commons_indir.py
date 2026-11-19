"""Wikimedia Commons'tan zor olumsuz (siddet olmayan) videolari indirir.

Her dosyanin lisansi, yazari ve sayfa adresi veri/ham/commons/kaynaklar.csv dosyasina yazilir.
Videolar en cok 120 sn olacak sekilde kirpilir (ffmpeg), ses korunur (ses olayi testi icin).
"""
import csv, json, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

UA = {"User-Agent": "siddet-kamera-prototip/0.1 (arastirma)"}
KOK = Path(__file__).resolve().parents[1] / "veri" / "ham" / "commons"
DOSYALAR = {
    "cocuk": ["Two children in the playground at school.webm", "Kids in the Playground.webm",
              "Children Playing in Cameroon.webm", "Pre-schoolers playing in Khayelitsha.webm",
              "Kids Running from DRONE!.webm", "Margaret Mahy Playground 002.webm",
              "Playgroundslideinuse-japan-2016-2-14.webm", "24 Hour Make a Wish Fun Run (976331).webm"],
    "sarilma": ["Diti hugs Natalie.ogv"],
    "spor": ["Sasebo Bo Boys Youth Basketball Game (999422).webm", "NSF Diego Garcia hosts basketball game (989713).webm",
             "Goud bezig! (26-10-2013) - RTV Noord.webm"],
    "kalabalik": ["Ratina Mall Fire Alarm Announcement, Tampere, Finland.webm",
                  "Interior View of Chelsea and Westminster Hospital, London.webm",
                  "Starbucks en la aĉetcentro Kristall (Tjumeno).webm", "Crowd in the subway station.webm"],
}

def api(**p):
    p["format"] = "json"
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))

def indir(url, yol, deneme=6):
    for i in range(deneme):
        try:
            veri = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
            open(yol, "wb").write(veri); time.sleep(3); return
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == deneme - 1:
                raise
            time.sleep(15 * (i + 1))

def temizle(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()

def main():
    KOK.mkdir(parents=True, exist_ok=True)
    satirlar = []
    for kategori, adlar in DOSYALAR.items():
        for ad in adlar:
            r = api(action="query", titles="File:" + ad, prop="videoinfo",
                    viprop="url|size|extmetadata|derivatives")
            pg = next(iter(r["query"]["pages"].values()))
            if "videoinfo" not in pg:
                print("YOK:", ad); continue
            vi = pg["videoinfo"][0]; em = vi.get("extmetadata", {})
            lis = em.get("LicenseShortName", {}).get("value", "?")
            # 480p ya da 360p turevi varsa onu al (daha kucuk)
            url = vi["url"]
            for d in vi.get("derivatives", []):
                if d.get("height") in (480, 360) and "webm" in d.get("type", ""):
                    url = d["src"]; break
            kod = f"{kategori}_{len([s for s in satirlar if s['kategori']==kategori])+1:02d}"
            ham = KOK / (kod + ".indir")
            hedef = KOK / (kod + ".mp4")
            if not hedef.exists():
                urllib.request.urlretrieve(urllib.request.Request(url, headers=UA).full_url, ham) if False else \
                    open(ham, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read())
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ham), "-t", "120", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(hedef)], check=True)
                ham.unlink()
            satirlar.append(dict(kod=kod, kategori=kategori, dosya=ad, lisans=lis,
                                 yazar=temizle(em.get("Artist", {}).get("value")),
                                 sayfa=vi.get("descriptionurl", ""), indirilen=url))
            print(kod, lis, ad)
    with open(KOK / "kaynaklar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0])); w.writeheader(); w.writerows(satirlar)

if __name__ == "__main__":
    main()
