import os, argparse, glob, re, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----- your model -----
from model import StudentPolicy  # expects the class from your repo


IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="data/frames", help="Root with frames (supports subfolders)")
    ap.add_argument("--labels_root", default=None, help="Optional; if provided, reads JSON to print true vs pred")
    ap.add_argument("--out_dir", default="gradcam_out", help="Where to save overlays")
    ap.add_argument("--ckpt", default="checkpoints/best.pth", help="Trained checkpoint (.pth)")
    ap.add_argument("--size", type=int, default=260, help="Model input size")
    ap.add_argument("--frames_T", type=int, default=3, help="Temporal stack for the student: 1 or 3")
    ap.add_argument("--limit", type=int, default=50, help="Max frames to visualize per folder (avoid huge dumps)")
    ap.add_argument("--every", type=int, default=2, help="Stride when sampling frames")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    return ap.parse_args()

_num = re.compile(r"(\d+)")

def frame_sort_key(name: str):
    m = _num.search(name)
    return int(m.group(1)) if m else name

def list_sessions(root: Path):
    sessions = []
    for dp, _, files in os.walk(root):
        if any(f.lower().endswith((".jpg",".png")) for f in files):
            sessions.append(Path(dp))
    return sorted(sessions)

def pil_crop_road(img_pil):
    w, h = img_pil.size
    top = int(0.10 * h)  # same as training sample
    return img_pil.crop((0, top, w, h))

def to_tensor(img_pil, size):
    if img_pil is None:
        raise ValueError("to_tensor received None image")
    img = img_pil.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    arr = arr.transpose(2,0,1)  # C,H,W
    return torch.from_numpy(arr)

def load_student(ckpt_path, in_ch, aux_tlight=False, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    # trust current in_ch / aux flag over ckpt if needed
    model = StudentPolicy(in_ch=in_ch, meta_dim=0, aux_tlight=aux_tlight).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def main():
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_ch = 3 * max(1, args.frames_T)

    model = load_student(args.ckpt, in_ch=in_ch, aux_tlight=False, device=dev)

    # pick a deep conv layer for CAM
    # MobileNetV3-small last block conv:
    try:
        target_layers = [model.backbone[-1][0]]
    except Exception:
        # fallback: find any Conv2d in last block
        last = model.backbone[-1]
        conv = None
        for m in reversed(list(last.modules())):
            if isinstance(m, torch.nn.Conv2d):
                conv = m; break
        if conv is None:
            raise RuntimeError("Could not find a Conv2d layer for Grad-CAM target.")
        target_layers = [conv]

    cam = GradCAM(model=model, target_layers=target_layers)

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    labels_root = Path(args.labels_root) if args.labels_root else None

    sessions = list_sessions(frames_root)
    if not sessions:
        print(f"No frames found under {frames_root}")
        return

    for sess in sessions:
        rel = sess.relative_to(frames_root)
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        names = sorted([n for n in os.listdir(sess) if n.lower().endswith((".jpg",".png"))], key=frame_sort_key)
        if not names: 
            continue

        # sample frames
        picked = names[::max(1,args.every)][:args.limit]

        # keep a small rolling buffer for T=3 causal stack [t-2,t-1,t]
        buffer = []
        min_history = max(0, args.frames_T - 1)  # need this many previous frames

        for i, fname in enumerate(names):
            if i < min_history:
                continue  # skip early frames that don't have enough history
            
            if args.frames_T == 1:
                pass  # single frame, no buffer needed
            else:
                # maintain buffer of T preprocessed tensors from same session
                if len(buffer) == 0:
                    # initialize with previous frames
                    for prev_idx in range(i - min_history, i + 1):
                        prev_img = pil_crop_road(Image.open(sess / names[prev_idx]).convert("RGB"))
                        prev_ten = to_tensor(prev_img, args.size)
                        buffer.append(prev_ten)
                else:
                    # push new frame, pop oldest
                    img_cur = pil_crop_road(Image.open(sess / fname).convert("RGB"))
                    ten_cur = to_tensor(img_cur, args.size)
                    buffer = buffer[1:] + [ten_cur]

            if fname not in picked:
                continue

            # build input tensor
            if args.frames_T == 1:
                img_c = pil_crop_road(Image.open(sess / fname).convert("RGB"))
                ten = to_tensor(img_c, args.size)
                t_in = ten.unsqueeze(0).to(dev)  # [1,3,H,W]
                # background for overlay
                bg_rgb = np.asarray(img_c.resize((args.size,args.size))).astype(np.float32)/255.0
            else:
                # channel-stacked [t-2,t-1,t]
                t_in = torch.cat(buffer, dim=0).unsqueeze(0).to(dev)  # [1,9,H,W]
                # Use the LAST frame (t) as the background
                bg_rgb = (buffer[-1].cpu().numpy().transpose(1,2,0) * IMNET_STD + IMNET_MEAN)
                bg_rgb = np.clip(bg_rgb, 0, 1)

            # forward once to get prediction (optional, not needed by GradCAM)
            with torch.no_grad():
                pred = model(t_in)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                steer_pred = float(pred[0,0].item())
                thr_pred   = float(pred[0,1].item())
                brk_pred   = float(pred[0,2].item())

            # Build CAM for the steer output neuron (index 0)
            # pytorch-grad-cam needs targets; use RawScoresOutputTarget-like behavior:
            class SteerTarget:
                def __call__(self, model_out):
                    # model_out may be [B,3] or [3] (no batch). Return the
                    # steer score in a batch-compatible way: model_out[...,0]
                    # yields shape (B,) for batched outputs or a scalar for 1D.
                    return model_out[..., 0]

            grayscale_cam = cam(input_tensor=t_in, targets=[SteerTarget()])[0]  # [H,W] in [0,1]
            cam_overlay = show_cam_on_image(bg_rgb, grayscale_cam, use_rgb=True)

            # annotate with numbers
            disp = cam_overlay.copy()
            if args.frames_T > 1:
                frame_nums = [frame_sort_key(names[max(0, i-j)]) for j in range(args.frames_T-1, -1, -1)]
                frames_txt = f"frames: {frame_nums}"
                cv2.putText(disp, frames_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(disp, frames_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            # Put predictions on separate lines for readability
            y0 = 46 if args.frames_T > 1 else 22
            line_h = 18
            txt1 = f"steer_pred={steer_pred:+.3f}"
            txt2 = f"thr={thr_pred:.3f}"
            txt3 = f"brk={brk_pred:.3f}"
            cv2.putText(disp, txt1, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt1, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, txt2, (10, y0+line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt2, (10, y0+line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, txt3, (10, y0+line_h*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt3, (10, y0+line_h*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # if label json exists, print target too
            if labels_root:
                jpath = labels_root / rel / (Path(fname).stem + ".json")
                if jpath.exists():
                    try:
                        data = json.load(open(jpath, "r", encoding="utf-8"))
                        c = data.get("controls", data)
                        s_true = float(c["steer"])
                        t_true = float(c["throttle"])
                        b_true = float(c["brake"])
                        # Put label values on separate lines below the predictions
                        label_y0 = (46 if args.frames_T > 1 else 22) + line_h*3
                        ltxt1 = f"label steer={s_true:+.3f}"
                        ltxt2 = f"thr={t_true:.3f}"
                        ltxt3 = f"brk={b_true:.3f}"
                        cv2.putText(disp, ltxt1, (10, label_y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(disp, ltxt1, (10, label_y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(disp, ltxt2, (10, label_y0+line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(disp, ltxt2, (10, label_y0+line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(disp, ltxt3, (10, label_y0+line_h*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(disp, ltxt3, (10, label_y0+line_h*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                    except Exception:
                        pass

            # save
            out_path = out_dir / f"{Path(fname).stem}_cam.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

        print(f"[session] {rel} → saved Grad-CAM overlays to {out_dir}")

if __name__ == "__main__":
    main()
