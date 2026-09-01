# make_cam_video.py
import os, cv2, re
from pathlib import Path

IN_ROOT = Path("gradcam_out")     # where *_cam.jpg were saved
OUT_ROOT = Path("cam_videos"); OUT_ROOT.mkdir(exist_ok=True, parents=True)
FPS = 20

_num = re.compile(r"(\d+)")
def sort_key(name): 
    m = _num.search(name); return int(m.group(1)) if m else name

for dp,_,files in os.walk(IN_ROOT):
    imgs = [f for f in files if f.endswith("_cam.jpg")]
    if not imgs: continue
    imgs = sorted(imgs, key=sort_key)
    sess = Path(dp).relative_to(IN_ROOT)
    out = OUT_ROOT / (str(sess).replace(os.sep,"_") + ".mp4")
    first = cv2.imread(str(Path(dp)/imgs[0]))
    h,w = first.shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w,h))
    for fn in imgs:
        vw.write(cv2.imread(str(Path(dp)/fn)))
    vw.release()
    print("wrote", out)
