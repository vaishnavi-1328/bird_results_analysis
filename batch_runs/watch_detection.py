"""Watch the bird detector work on a short piece of a recording.

  python watch_detection.py <video> <start hh:mm:ss into the video> [minutes, default 3] [--save out.mp4]

Red box    = a track the detector confirmed and COUNTED as a bird.
Yellow box = a moving candidate that has not been counted.
Keys (window mode only): space = pause/resume, q = quit.
With --save it writes the annotated video instead of opening a window (needed with headless OpenCV).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import detection

FFMPEG = (r"C:\Users\DriveLab\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")

video, start = sys.argv[1], sys.argv[2]
minutes = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 3
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None

# 1. Cut the piece out without re-encoding (takes a few seconds).
clip = Path(tempfile.gettempdir()) / "watch_clip.mp4"
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", start, "-i", video,
                "-t", str(minutes * 60), "-c", "copy", "-an", str(clip)], check=True)

# 2. Run the unchanged detector on it, drawing every processed frame instead of every 15th.
detection._FRAME_SEND_INTERVAL = detection.FRAME_SKIP
writer = None
show = not save            # with --save, write the file only (headless OpenCV has no window)
if show:
    cv2.namedWindow("detector", cv2.WINDOW_NORMAL)
for ev in detection.process_video_streaming(str(clip)):
    if ev["kind"] == "result":
        d = ev["data"]
        print(f"{minutes:g} min from {start}: {d['unique_flying_birds']:,} birds counted, "
              f"max {d['max_concurrent_birds']} in view at once")
    if ev["kind"] != "frame":
        continue
    frame = ev["frame"]
    if save:
        if writer is None:
            writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), 15, frame.shape[1::-1])
        writer.write(frame)
    if show:
        cv2.imshow("detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
if writer:
    writer.release()
if show:
    cv2.destroyAllWindows()
