"""Test the detector on short samples of every recording before a full rerun.

  python test_runs\\run_tests.py

For each recording, cuts two 30 s clips -- busy (the minute the old detector counted most)
and noon (13:00 clock time) -- and writes under test_runs\\:
  clips\\<key>_<label>.mp4       raw clip
  annotated\\<key>_<label>.mp4   detector's view: red = counted bird, yellow = candidate not counted
  summary.csv                   old vs new count per clip
"""
import csv
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
sys.path.insert(0, str(BATCH.parent / "backend"))
import detection

FFMPEG = (r"C:\Users\DriveLab\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")
VIDEOS = Path(r"C:\Farm_videos")
RECOUNT_1080 = {"2026-09-30_A": "farm_video_compressed_1080.mp4", "2026-10-09_B": "merged_1080.mp4"}
CLIP_S = 30


def jobs():
    for r in csv.DictReader(open(BATCH / "results.csv")):
        key = f"{r['date']}_{r['location']}"
        video = VIDEOS / r["folder"] / RECOUNT_1080.get(key, r["video"])
        log = BATCH / "logs" / (f"{key}_1080" if key in RECOUNT_1080 else key)
        minutes = [int(m["new_birds"]) for m in csv.DictReader(open(f"{log}_per_minute.csv"))]
        fps = float(r["fps"])
        minute_s = 60 * int(fps) / fps            # the old detector's "minute" in real seconds
        busy = max(range(len(minutes)), key=minutes.__getitem__)
        noon = (datetime.fromisoformat(f"{r['date']} 13:00") - datetime.fromisoformat(r["recording_start"])).total_seconds()
        noon = min(max(noon, 0), float(r["duration_hours"]) * 3600 - CLIP_S)
        for label, offset in (("busy", busy * minute_s), ("noon", noon)):
            yield key, label, str(video), offset, minutes[min(int(offset / minute_s), len(minutes) - 1)]


def run(job):
    key, label, video, offset, old_per_min = job
    clip = HERE / "clips" / f"{key}_{label}.mp4"
    if not clip.exists():
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{offset:.0f}", "-i", video,
                        "-t", str(CLIP_S), "-c", "copy", "-an", str(clip)], check=True)
    detection._FRAME_SEND_INTERVAL = detection.FRAME_SKIP     # annotate every processed frame
    out, writer, t0, d = HERE / "annotated" / f"{key}_{label}.mp4", None, time.time(), None
    for ev in detection.process_video_streaming(str(clip)):
        if ev["kind"] == "frame":
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, ev["frame"].shape[1::-1])
            writer.write(ev["frame"])
        elif ev["kind"] == "result":
            d = ev["data"]
    if writer:
        writer.release()
    return {"recording": key, "clip": label, "video_offset_s": round(offset),
            "old_count_that_minute": old_per_min, "new_count_30s": d["unique_flying_birds"],
            "new_max_in_view": d["max_concurrent_birds"], "processing_s": round(time.time() - t0)}


if __name__ == "__main__":
    (HERE / "clips").mkdir(exist_ok=True)
    (HERE / "annotated").mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run, jobs()))
    with open(HERE / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
