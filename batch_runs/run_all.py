"""Batch-run detection on every C:\\Farm_videos\\<mmddyyyy>\\<location>\\ video, one child process per video.

Usage:  python run_all.py              (resumes; interrupted/crashed videos continue from their checkpoint)
        python run_all.py --retry      (also re-runs safety_stop/error videos)
        python run_all.py --threads 2  (OpenCV threads per run, default 4; 0 = one per core)
        python run_all.py --workers 3  (videos processed at the same time, default 4)
        python run_all.py --skip 2026-10-09_B  (leave these videos out, comma-separated)
Children run at below-normal priority. A video is given up on after MAX_ATTEMPTS starts.
Where a folder has a *_1080* re-encode next to the 4K original, the 1080p file is used,
with the original's start time from starttimes.csv.
Outputs, one folder per location under OUT (results_v2\\location_A, results_v2\\location_B):
  results.csv                       one row per video
  <date>_<loc>_per_minute.csv       new birds per minute of footage, with wall-clock time
  logs/<date>_<loc>.jsonl           heartbeat every HEARTBEAT frames (flushed + fsynced)
  logs/<date>_<loc>.fault.txt       native-crash traceback from faulthandler
  logs/<date>_<loc>.stderr.txt      child stderr
  logs/<date>_<loc>.ckpt            tracker checkpoint (every 5 min), deleted once the video is done
Recording start times come from starttimes.csv (camera timecode HH:MM:SS;FF, ';' = drop-frame).
"""
import concurrent.futures, csv, ctypes, datetime, faulthandler, json, os, subprocess, sys, threading, time

ROOT       = r"C:\Farm_videos"
BACKEND    = r"C:\Bird_influenza_analysis\backend"
HERE       = os.path.dirname(os.path.abspath(__file__))
OUT        = r"C:\Bird_influenza_analysis\results_v2"
STARTTIMES = os.path.join(HERE, "starttimes.csv")
THRESHOLD  = 127          # same value the frontend sends
HEARTBEAT  = 2000         # frames between heartbeats
RSS_LIMIT  = 8e9
AVAIL_MIN  = 3e9
MAX_ATTEMPTS = 10
VIDEO_EXT  = (".mp4", ".mov", ".avi", ".mkv")
FIELDS = ["date", "location", "start_timecode", "recording_start", "recording_end",
          "folder", "video", "size_gb", "width", "height", "fps", "total_frames",
          "duration_hours", "status", "attempts", "cv_threads", "exit_code", "last_frame", "last_video_time", "last_frame_clock_time",
          "progress_pct",
          "peak_rss_mb", "min_sys_avail_gb", "unique_flying_birds", "max_concurrent_birds", "total_tracks",
          "processing_time_s", "wall_start", "wall_end", "python", "opencv", "note"]


def loc_dir(loc):
    return os.path.join(OUT, f"location_{loc}")


def logs_dir(loc):
    return os.path.join(loc_dir(loc), "logs")


def find_videos():
    jobs = []
    for d in sorted(os.listdir(ROOT)):
        dpath = os.path.join(ROOT, d)
        if not os.path.isdir(dpath):
            continue
        date = datetime.datetime.strptime(d, "%m%d%Y").date().isoformat()
        for loc in sorted(os.listdir(dpath)):
            lpath = os.path.join(dpath, loc)
            if not os.path.isdir(lpath):
                continue
            vids = sorted(f for f in os.listdir(lpath)
                          if f.lower().endswith(VIDEO_EXT) and not f.upper().startswith("GX"))
            vids.sort(key=lambda f: ("_1080" not in f, f))      # prefer the 1080p re-encode
            if not vids:
                print(f"skip {d}/{loc}: no video")
                continue
            if len(vids) > 1:
                print(f"WARNING {d}/{loc}: several videos {vids}, using {vids[0]}")
            jobs.append({"date": date, "location": loc[-1].upper(), "folder": f"{d}/{loc}",
                         "path": os.path.join(lpath, vids[0])})
    return jobs


def key(job):
    return f"{job['date']}_{job['location']}"


def read_starttimes():
    """Map lower-cased video path -> timecode string."""
    if not os.path.exists(STARTTIMES):
        return {}
    with open(STARTTIMES, newline="") as f:
        return {os.path.normcase(r["file"]): r["timecode"].strip() for r in csv.DictReader(f)}


def recording_start(date, timecode, fps):
    """date 'YYYY-MM-DD' + 'HH:MM:SS;FF' -> datetime (FF frames converted with the video fps)."""
    hms, _, ff = timecode.replace(";", ":").rpartition(":")
    h, m, s = (int(x) for x in hms.split(":"))
    start = datetime.datetime.fromisoformat(date).replace(hour=h, minute=m, second=s)
    return start + datetime.timedelta(seconds=int(ff) / (fps or 30))


def add_clock_times(row, job):
    """Fill recording_start / recording_end / last_frame_clock_time from the timecode."""
    tc = job.get("timecode")
    row["start_timecode"] = tc or ""
    if not tc:
        row["note"] = ((row.get("note") or "") + " no start timecode in starttimes.csv").strip()
        return
    fps = float(row.get("fps") or 30)
    start = recording_start(job["date"], tc, fps)
    row["recording_start"] = start.isoformat(sep=" ", timespec="seconds")
    if row.get("duration_hours") not in (None, ""):
        end = start + datetime.timedelta(hours=float(row["duration_hours"]))
        row["recording_end"] = end.isoformat(sep=" ", timespec="seconds")
    if row.get("last_frame") not in (None, ""):
        at = start + datetime.timedelta(seconds=int(row["last_frame"]) / fps)
        row["last_frame_clock_time"] = at.isoformat(sep=" ", timespec="seconds")


def write_per_minute(job, log_path):
    """Per-minute new-bird counts from the final result, labelled with wall-clock time."""
    result = None
    with open(log_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "result":
                result = e["result"]
    if not result or not job.get("timecode"):
        return
    start = recording_start(job["date"], job["timecode"], result.get("fps") or 30)
    with open(os.path.join(loc_dir(job["location"]), key(job) + "_per_minute.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "clock_time", "new_birds"])
        for i, n in enumerate(json.loads(result["birds_per_minute"])):
            w.writerow([i, (start + datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M"), n])


def read_results():
    rows = {}
    for loc in ("A", "B"):
        path = os.path.join(loc_dir(loc), "results.csv")
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows.update({f"{r['date']}_{r['location']}": r for r in csv.DictReader(f)})
    return rows


def write_results(rows):
    """One results.csv per location folder."""
    for loc in sorted({r["location"] for r in rows.values()}):
        path = os.path.join(loc_dir(loc), "results.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            for k in sorted(rows):
                if rows[k]["location"] == loc:
                    w.writerow({c: rows[k].get(c, "") for c in FIELDS})
        os.replace(tmp, path)


def summarize_log(log_path):
    """Fold a heartbeat log into result fields (works for finished and cut-off runs)."""
    out = {"peak_rss_mb": 0, "min_sys_avail_gb": 99.0}
    if not os.path.exists(log_path):
        return out
    with open(log_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line may be torn by a hard crash
            if "rss_mb" in e:
                out["peak_rss_mb"] = max(out["peak_rss_mb"], e["rss_mb"])
            if "sys_avail_gb" in e:
                out["min_sys_avail_gb"] = min(out["min_sys_avail_gb"], e["sys_avail_gb"])
            for k in ("last_frame", "last_video_time", "progress_pct", "width", "height", "fps",
                      "total_frames", "duration_hours", "python", "opencv", "status", "note",
                      "unique_flying_birds", "max_concurrent_birds", "total_tracks", "processing_time_s"):
                if k in e:
                    out[k] = e[k]
    if out["min_sys_avail_gb"] == 99.0:
        out["min_sys_avail_gb"] = ""
    return out


def probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"width": int(cap.get(3)), "height": int(cap.get(4)), "fps": round(fps, 3),
            "total_frames": total, "duration_hours": round(total / fps / 3600, 2)}
    cap.release()
    return info


def base_row(job):
    row = {"date": job["date"], "location": job["location"], "folder": job["folder"],
           "video": os.path.basename(job["path"]), "size_gb": round(os.path.getsize(job["path"]) / 1e9, 2)}
    row.update(job.get("info") or probe(job["path"]))
    return row


def run_job(job, rows, lock, threads):
    """Run (or resume) one video in a child process; rows/results.csv are only touched under lock."""
    k = key(job)
    log_path = os.path.join(logs_dir(job["location"]), k + ".jsonl")
    ckpt_path = os.path.join(logs_dir(job["location"]), k + ".ckpt")
    with lock:
        prev = rows.get(k)
        attempts = int((prev or {}).get("attempts") or (1 if prev else 0))
        resuming = os.path.exists(ckpt_path)
        if os.path.exists(log_path) and not resuming:
            os.replace(log_path, log_path + f".prev{int(time.time())}")
        row = base_row(job)
        row.update(status="running", attempts=attempts + 1, cv_threads=threads,
                   wall_start=(prev or {}).get("wall_start") if resuming else None)
        row["wall_start"] = row["wall_start"] or datetime.datetime.now().isoformat(timespec="seconds")
        add_clock_times(row, job)
        rows[k] = row
        write_results(rows)
        print(f"[{k}] {'resume' if resuming else 'start'} (attempt {attempts + 1}) {job['path']}", flush=True)

    env = dict(os.environ, BIRD_CV_THREADS=str(threads))
    with open(os.path.join(logs_dir(job["location"]), k + ".stderr.txt"), "a") as err:
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--child", job["path"], log_path, ckpt_path],
                             stdout=err, stderr=subprocess.STDOUT, env=env,
                             creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    row.update(summarize_log(log_path))
    row["exit_code"] = rc
    row["wall_end"] = datetime.datetime.now().isoformat(timespec="seconds")
    if rc == 0 and row.get("status") == "done":
        pass
    elif rc == 2:
        row["status"] = "safety_stop"
    elif row.get("status") != "error":
        row["status"] = "crashed"
        row["note"] = f"child exit code {rc} ({rc & 0xFFFFFFFF:#010x})"
    add_clock_times(row, job)
    if row["status"] == "done":
        write_per_minute(job, log_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    with lock:
        rows[k] = row
        write_results(rows)
        print(f"[{k}] {row['status']} rc={rc} last_frame={row.get('last_frame')} "
              f"birds={row.get('unique_flying_birds')}", flush=True)


def run_parent(retry, threads, workers, skip):
    # Keep Windows from idle-sleeping while the batch runs (it slept for 4 days on Oct 29).
    # A lid close or the power button can still put it to sleep.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    for loc in ("A", "B"):
        os.makedirs(logs_dir(loc), exist_ok=True)
    jobs = find_videos()
    rows = read_results()
    starts = read_starttimes()
    for job in jobs:
        # a *_1080 re-encode uses the start time of the original it was made from
        job["timecode"] = (starts.get(os.path.normcase(job["path"]))
                           or starts.get(os.path.normcase(job["path"].replace("_1080", ""))))
        if not job["timecode"]:
            print(f"WARNING {job['folder']}: no start timecode in starttimes.csv", flush=True)
        job["info"] = probe(job["path"])
    # Lowest resolution first, then by date/location
    jobs.sort(key=lambda j: (j["info"]["width"] * j["info"]["height"], j["date"], j["location"]))
    print(f"{len(jobs)} videos found, order:", flush=True)
    for job in jobs:
        print(f"  {key(job)} {job['info']['width']}x{job['info']['height']} {job['info']['duration_hours']}h", flush=True)
    todo = []
    for job in jobs:
        k = key(job)
        if k in skip:
            print(f"[{k}] skipped (--skip)", flush=True)
            continue
        prev = rows.get(k)
        if prev and prev["status"] == "running":
            # Parent died mid-video last time: most likely a system crash/reboot.
            prev.update(summarize_log(os.path.join(logs_dir(job["location"]), k + ".jsonl")))
            prev["status"] = "interrupted"
            prev["note"] = "run was cut off (system crash/reboot?) - see last_frame / last heartbeat"
            add_clock_times(prev, job)
            write_results(rows)
        if prev:
            attempts = int(prev.get("attempts") or 1)
            status = prev["status"]
            resumable = status in ("interrupted", "crashed") and attempts < MAX_ATTEMPTS
            if status == "done" or not (resumable or retry):
                print(f"[{k}] already {status} (attempts={attempts}), skipping", flush=True)
                continue
        todo.append(job)
    print(f"{len(todo)} videos to run, {workers} at a time", flush=True)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job, rows, lock, threads): key(job) for job in todo}
        for f in concurrent.futures.as_completed(futures):
            if f.exception():
                print(f"[{futures[f]}] parent error: {f.exception()!r}", flush=True)
    print("all done", flush=True)


def run_child(path, log_path, ckpt_path):
    import logging
    import cv2
    import psutil
    sys.path.insert(0, BACKEND)
    logging.disable(logging.INFO)
    import detection

    fault = open(log_path.replace(".jsonl", ".fault.txt"), "w")
    faulthandler.enable(fault)
    log = open(log_path, "a")

    def emit(**e):
        e["t"] = datetime.datetime.now().isoformat(timespec="seconds")
        log.write(json.dumps(e) + "\n")
        log.flush()
        os.fsync(log.fileno())

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = detection.load_checkpoint(ckpt_path, path) if os.path.exists(ckpt_path) else None
    emit(event="start", video=path, width=int(cap.get(3)), height=int(cap.get(4)), fps=round(fps, 3),
         total_frames=total, duration_hours=round(total / fps / 3600, 2),
         python=sys.version.split()[0], opencv=cv2.__version__, cpu_count=os.cpu_count(),
         cv_threads=cv2.getNumThreads(), resume_from_frame=state["frame_count"] if state else 0)
    cap.release()

    proc = psutil.Process()
    psutil.cpu_percent(None)
    t0 = time.time()
    next_hb = 0
    for item in detection.process_video_streaming(path, threshold=THRESHOLD, checkpoint_path=ckpt_path):
        if item["kind"] == "progress":
            frame = int(round(item["progress"] * total)) if total else 0
            rss, avail = proc.memory_info().rss, psutil.virtual_memory().available
            if frame >= next_hb:
                next_hb = frame + HEARTBEAT
                emit(event="hb", last_frame=frame, last_video_time=str(datetime.timedelta(seconds=int(frame / fps))),
                     progress_pct=round(item["progress"] * 100, 2), rss_mb=round(rss / 1e6),
                     sys_avail_gb=round(avail / 1e9, 2), cpu_pct=psutil.cpu_percent(None),
                     unique_flying_birds=item["unique_birds"], elapsed_s=round(time.time() - t0))
            if rss > RSS_LIMIT or avail < AVAIL_MIN:
                emit(event="safety_stop", status="safety_stop", last_frame=frame, rss_mb=round(rss / 1e6),
                     sys_avail_gb=round(avail / 1e9, 2), note="memory limit hit")
                os._exit(2)
        elif item["kind"] == "result":
            r = item["data"]
            emit(event="result", status="done", progress_pct=100, last_frame=r["frames_processed"],
                 last_video_time=str(datetime.timedelta(seconds=int(r["frames_processed"] / fps))),
                 unique_flying_birds=r["unique_flying_birds"], max_concurrent_birds=r["max_concurrent_birds"],
                 total_tracks=r["total_tracks"], processing_time_s=r["processing_time"], result=r)
            return 0
    # process_video_streaming catches its own exceptions and just stops, so record that here
    emit(event="no_result", status="error",
         note="generator ended without a result - detection.py logged an exception (see stderr file)")
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        sys.exit(run_child(sys.argv[2], sys.argv[3], sys.argv[4]))
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    skip = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
    run_parent(retry="--retry" in sys.argv, threads=threads, workers=workers, skip=skip)
