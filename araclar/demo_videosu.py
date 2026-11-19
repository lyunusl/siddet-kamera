"""Test kümesinden (eğitimde görülmemiş) kliplerle tek bir demo videosu kurar: ornekler/demo_test_karisik.mp4

Sıra: normal ve kavga bölümleri sırayla. Her parçanın başlangıç saniyesi ve etiketi
ornekler/demo_test_karisik.json dosyasına yazılır (doğru cevap anahtarı). Bölümler arasında 1,5 sn siyah
ara vardır (farklı kayıtlar arasındaki kesme takipte sahte bir "kişi atlaması" yaratmasın); bir bölümün
içindeki aynı veri setinden kısa klipler, ölçümdeki gibi art arda eklenir. Sonda yapay zekâ ile üretilmiş
bir hasta düşmesi klibi vardır (veri/ham/uretilen/u06_hasta_dusmesi.mp4).
Sesi olmayan kliplere sessiz ses kanalı eklenir; sesli kliplerin (Surveillance Fight, GMDCSA) kendi sesi kalır.
"""
import csv
import json
import random
import subprocess
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "ornekler"


def sure(v):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                                capture_output=True, text=True).stdout.strip())


def ses_var(v):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(v)], capture_output=True, text=True).stdout.strip()
    return bool(r)


def main():
    rnd = random.Random(11)
    test = [r for r in csv.DictReader(open(KOK / "veri" / "liste.csv", encoding="utf-8")) if r["bolum"] == "test"]

    def sec(veri, etiket, n, kat=None, filtre=None):
        a = [r for r in test if r["veri"] == veri and r["etiket"] == str(etiket) and (kat is None or r["kategori"] == kat)
             and (filtre is None or filtre(r))]
        rnd.shuffle(a)
        return a[:n]

    cam1 = lambda r: "/cam1/" in r["video"]
    parcalar = [
        ("normal", sec("survfight", 0, 3)),
        ("kavga", sec("survfight", 1, 3)),
        ("normal", sec("airtlab", 0, 2, filtre=cam1)),        # sarılma / çak / alkış
        ("kavga", sec("airtlab", 1, 1, filtre=cam1)),
        ("normal", sec("gmdcsa", 0, 2, kat="hasta_dusmesi")),
        ("normal", sec("bus", 0, 4)),
        ("kavga", sec("bus", 1, 4)),
        ("normal", sec("gmdcsa", 0, 1, kat="gunluk_hareket")),
        ("dusme", [dict(video="veri/ham/uretilen/u06_hasta_dusmesi.mp4", kategori="hasta_dusmesi_yapay_zeka",
                        veri="uretilen")]),
    ]
    ARA = 1.5
    girdiler, filtre, anahtar = [], [], []
    t = 0.0
    i = 0   # birleştirilecek parça numarası
    g = 0   # ffmpeg girdi numarası
    for etiket, klipler in parcalar:
        for r in klipler:
            v = KOK / r["video"]
            s = sure(v)
            girdiler += ["-i", str(v)]
            sesli = ses_var(v)
            filtre.append(f"[{g}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
                          f"fps=30,setsar=1,trim=duration={s:.3f},setpts=PTS-STARTPTS[v{i}]")
            if sesli:
                filtre.append(f"[{g}:a]aresample=44100,aformat=channel_layouts=mono,atrim=duration={s:.3f},"
                              f"apad=whole_dur={s:.3f},asetpts=PTS-STARTPTS[a{i}]")
            else:
                filtre.append(f"anullsrc=r=44100:cl=mono,atrim=duration={s:.3f}[a{i}]")
            anahtar.append(dict(bas=round(t, 2), bit=round(t + s, 2), etiket=etiket, kategori=r["kategori"],
                                kaynak=r["veri"], ses=sesli, video=r["video"]))
            t += s
            i += 1
            g += 1
        filtre.append(f"color=c=black:s=1280x720:r=30:d={ARA}[v{i}]")
        filtre.append(f"anullsrc=r=44100:cl=mono,atrim=duration={ARA}[a{i}]")
        t += ARA
        i += 1
    filtre.append("".join(f"[v{j}][a{j}]" for j in range(i)) + f"concat=n={i}:v=1:a=1[v][a]")
    CIKTI.mkdir(exist_ok=True)
    hedef = CIKTI / "demo_test_karisik.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *girdiler, "-filter_complex", ";".join(filtre), "-map", "[v]",
                    "-map", "[a]", "-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-c:a", "aac", str(hedef)],
                   check=True)
    json.dump(dict(video=hedef.name, sure=round(t, 2), parcalar=anahtar), open(CIKTI / "demo_test_karisik.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    print(hedef, round(t, 1), "sn,", i, "klip")


if __name__ == "__main__":
    main()
