# Dataset Guide

## Kinetics-400

**Source:** DeepMind / Google  
**Size:** 400 classes, ~240k training clips, ~20k validation clips  
**Clip length:** 10 seconds each  
**Format:** MP4, typically 720p at 25fps

### Classes Used

We target 80 of the 400 classes, grouped by which pipeline stream they best exercise:

| Group | Classes | Stream validated |
|-------|---------|-----------------|
| HIGH_MOTION (25) | gymnastics tumbling, breakdancing, somersaulting, parkour, skateboarding, snowboarding, springboard diving, hurdling, long jump, triple jump, high jump, pole vault, hammer throw, javelin throw, shot put, bungee jumping, skydiving, rock climbing, ice skating, skiing slalom, skiing (not slalom or crosscountry), snowkiting, bouncing on trampoline, cartwheeling, surfing water | M_i (motion score) |
| HIGH_SEMANTIC (30) | shooting basketball, dribbling basketball, dunking basketball, playing basketball, playing tennis, playing volleyball, playing cricket, playing badminton, playing ice hockey, kicking soccer ball, shooting goal (soccer), playing guitar, playing piano, playing drums, playing violin, riding a bike, riding mountain bike, riding horse, riding camel, riding elephant, driving car, driving tractor, sled dog racing, walking the dog, feeding birds, catching fish, archery, bowling, golf driving, golf putting | O_i (YOLO object score) |
| HIGH_FACE (19) | laughing, crying, singing, hugging, kissing, celebrating, applauding, clapping, giving or receiving award, blowing out candles, opening present, yawning, sneezing, headbanging, pumping fist, news anchoring, presenting weather forecast, testifying, answering questions | E_i (FER emotion score) |
| LOW_ACTIVITY (6) | reading book, reading newspaper, texting, using computer, writing, waiting in line | Control group (low S_i) |

### Download
`ash
# FiftyOne metadata is cached to ~/fiftyone/kinetics-400/
# The script reads the CSV and downloads via python -m yt_dlp
python scripts/download_kinetics400.py
`
Saves to: datasets/kinetics400/<class name>/<clip>.mp4

---

## HMDB51

**Source:** Serre Lab, Brown University  
**Size:** 51 action categories, 6,849 clips  
**Clip length:** 2–5 seconds  
**Format:** AVI, 320×240, 30fps

### Download
1. Download from: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
2. Extract all inner .rar files
3. Set path in config.yaml:

`yaml
datasets:
  hmdb51_root: "D:/datasets/hmdb51"
`

### Categories Used

| Group | Categories |
|-------|-----------|
| HIGH_FACE | laugh, cry, talk, smile, kiss |
| HIGH_MOTION | cartwheel, somersault, jump, dive, flic_flac |
| HIGH_OBJECT | shoot_ball, ride_bike, play_guitar, dribble, golf |
| LOW_MOTION (control) | sit, stand, smoke, eat, drink |
