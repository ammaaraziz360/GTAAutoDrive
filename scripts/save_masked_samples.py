import sys, pathlib
repo_root = str(pathlib.Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dataset import FramesControls, IMNET_MEAN, IMNET_STD
import torch, os
from PIL import Image

OUT_DIR = "debug_masked"
os.makedirs(OUT_DIR, exist_ok=True)

# Adjust paths here if your dataset paths differ
ds = FramesControls("data/frames", "data/labels_teacher", train=False, input_size=224, use_prev=False)

n = min(8, len(ds))
print(f"Saving {n} sample masked images to {OUT_DIR}/")
for i in range(n):
    x, y, t = ds[i]
    # x may be 3xHxxW or 6xHxxW (if use_prev); show the current frame (last 3 channels)
    if x.shape[0] == 6:
        img_t = x[-3:]
    else:
        img_t = x[:3]

    mean = torch.tensor(IMNET_MEAN).view(3,1,1)
    std = torch.tensor(IMNET_STD).view(3,1,1)
    img = (img_t * std + mean).clamp(0,1)
    np_img = (img * 255).byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(np_img).save(os.path.join(OUT_DIR, f"masked_{i}.jpg"))

print("Done.")
