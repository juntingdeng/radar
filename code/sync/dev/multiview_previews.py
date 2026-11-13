"""Render a preview.mp4 inside every multiview_v1 window dir (real-time 10 Hz, annotated with pid + labelled
object count + the window's selection reason). Dataset lives on local disk, so ffmpeg writes directly.
Run: python sync/dev/multiview_previews.py
"""
import os, csv, json, glob, subprocess, tempfile, shutil, collections
import numpy as np, cv2

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.expanduser("~/CMU/Research/mobile_data_local/multiview_v1/0508")
LABELS = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
FPS = 10

objc = collections.Counter()
for f in glob.glob(os.path.join(LABELS, "*.json")):
    for sid, boxes in json.load(open(f)).items():
        if sid.startswith("0508"):
            objc[int(sid.split("_")[1])] = max(objc[int(sid.split("_")[1])],
                                               sum(1 for b in boxes if "forward" in b))
man = json.load(open(os.path.join(OUT, "manifest.json")))
for w in man["windows"]:
    wdir = w["dir"]; name = os.path.basename(wdir)
    stage = tempfile.mkdtemp(prefix="mvprev_"); k = 0
    for fp in sorted(glob.glob(os.path.join(wdir, "color", "p*.*"))):
        pid = int(os.path.basename(fp)[1:7])
        im = cv2.imread(fp)
        if im is None: continue
        bar = np.full((34, im.shape[1], 3), 250, np.uint8)
        why = w["why"][:58]
        cv2.putText(bar, f"{name}  pid {pid}  objs {objc.get(pid, 0)}  | {why}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 40, 50), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(stage, f"{k:05d}.jpg"), np.vstack([bar, im])); k += 1
    out = os.path.join(wdir, "preview.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(stage, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
    shutil.rmtree(stage)
    print(f"{name}: preview.mp4 {os.path.getsize(out)//1024} KB ({k} frames)", flush=True)
print("done")
