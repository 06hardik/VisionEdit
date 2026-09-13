"""
scripts/download_kinetics400.py
================================
Downloads targeted Kinetics-400 classes by reading the metadata CSV
that FiftyOne already cached, then using `python -m yt_dlp` directly.

WHY this approach:
- yt-dlp.exe may be blocked by Application Control policies
- FiftyOne uses hardcoded format codes that YouTube no longer supports
- This script uses `python -m yt_dlp` (the Python module, not the exe)
  with a robust format selector that always works

Run:
    python scripts/download_kinetics400.py

Requirements: fiftyone (already installed), yt-dlp (already installed)
"""

import csv, pathlib, subprocess, sys, time, shutil
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

# Where to save the downloaded clips
OUT_DIR = pathlib.Path(r"C:\Users\ASUS\OneDrive\Desktop\VisionEdit\datasets\kinetics400")

# FiftyOne cached the metadata here when you ran the first attempt
CSV_PATH = pathlib.Path(r"C:\Users\ASUS\fiftyone\kinetics-400\tmp-download\kinetics400\validate.csv")

# Max clips to download per class (lower = faster, higher = more data)
MAX_PER_CLASS = 30

# yt-dlp format: tries mp4/webm up to 720p, falls back to best available
YT_FORMAT = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]/best"

# Seconds to wait between downloads (avoids YouTube rate-limiting)
SLEEP_BETWEEN = 1.5

# ── Target classes ────────────────────────────────────────────────────────────
TARGET_CLASSES = [
    # HIGH_MOTION
    "gymnastics tumbling","breakdancing","somersaulting","parkour","skateboarding",
    "snowboarding","surfing water","springboard diving","hurdling","long jump",
    "triple jump","high jump","pole vault","hammer throw","javelin throw","shot put",
    "bungee jumping","skydiving","rock climbing","ice skating",
    "skiing (not slalom or crosscountry)","skiing slalom","snowkiting",
    "bouncing on trampoline","cartwheeling",
    # HIGH_SEMANTIC
    "shooting basketball","dribbling basketball","dunking basketball","playing basketball",
    "playing tennis","playing volleyball","playing cricket","playing badminton",
    "playing ice hockey","kicking soccer ball","shooting goal (soccer)",
    "playing guitar","playing piano","playing drums","playing violin",
    "riding a bike","riding mountain bike","riding horse","riding camel","riding elephant",
    "driving car","driving tractor","sled dog racing","walking the dog",
    "feeding birds","catching fish","archery","bowling","golf driving","golf putting",
    # HIGH_FACE (most important for FER stream)
    "laughing","crying","singing","hugging","kissing","celebrating","applauding",
    "clapping","giving or receiving award","blowing out candles","opening present",
    "yawning","sneezing","headbanging","pumping fist","news anchoring",
    "presenting weather forecast","testifying","answering questions",
    # LOW_ACTIVITY (control group)
    "reading book","reading newspaper","texting","using computer","writing","waiting in line",
]

TARGET_SET = set(TARGET_CLASSES)

# ── Load + filter CSV ─────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    print(f"ERROR: CSV not found at {CSV_PATH}")
    print("Run `python scripts/download_kinetics400.py` once first to trigger FiftyOne metadata download.")
    sys.exit(1)

print("Reading metadata CSV...")
by_class = defaultdict(list)
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["label"] in TARGET_SET:
            by_class[row["label"]].append(row)

total_available = sum(len(v) for v in by_class.values())
print(f"Classes in CSV  : {len(by_class)} / {len(TARGET_CLASSES)}")
print(f"Clips available : {total_available}")
print(f"Max per class   : {MAX_PER_CLASS}")
print(f"Output dir      : {OUT_DIR}")
print()

# ── Download loop ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
grand_ok = 0
grand_fail = 0
grand_skip = 0

for label in sorted(by_class.keys()):
    clips = by_class[label][:MAX_PER_CLASS]
    class_dir = OUT_DIR / label
    class_dir.mkdir(parents=True, exist_ok=True)

    already = len(list(class_dir.glob("*.mp4")))
    need    = max(0, MAX_PER_CLASS - already)

    if need == 0:
        print(f"[SKIP] {label!r:45s} ({already} clips already present)")
        grand_skip += already
        continue

    print(f"[DOWN] {label!r:45s} need={need}  available={len(clips)}", flush=True)
    ok = 0; fail = 0

    for row in clips:
        if ok >= need:
            break

        yt_id  = row["youtube_id"]
        t_start = int(row["time_start"])
        t_end   = int(row["time_end"])
        url     = f"https://www.youtube.com/watch?v={yt_id}"
        out_tpl = str(class_dir / f"{yt_id}.%(ext)s")

        # Skip if already downloaded in a previous run
        if any(class_dir.glob(f"{yt_id}.*")):
            ok += 1
            continue

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--quiet",
            "--no-warnings",
            "--format", YT_FORMAT,
            "--merge-output-format", "mp4",
            "--download-sections", f"*{t_start}-{t_end}",
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--output", out_tpl,
            url,
        ]

        try:
            result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
            if result.returncode == 0 and any(class_dir.glob(f"{yt_id}.*")):
                ok += 1
            else:
                fail += 1
        except subprocess.TimeoutExpired:
            fail += 1
        except Exception:
            fail += 1

        time.sleep(SLEEP_BETWEEN)

    print(f"       ok={ok}  fail={fail}")
    grand_ok   += ok
    grand_fail += fail

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"Download complete.")
print(f"  Downloaded  : {grand_ok}")
print(f"  Failed/skip : {grand_fail}  (deleted/private YouTube vids)")
print(f"  Pre-existing: {grand_skip}")
print(f"  Location    : {OUT_DIR}")
print()
all_clips = list(OUT_DIR.rglob("*.mp4"))
print(f"Total .mp4 files on disk: {len(all_clips)}")
print("=" * 65)
