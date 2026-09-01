# dataset.py
import json, os, random
from PIL import Image, ImageDraw
import torch
import re
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

# -------- Minimap masking (bottom-left HUD) --------
# Set these ratios to match your capture. The code will mask the box
# starting at (x_ratio*W, y_ratio*H) with size (w_ratio*W, h_ratio*H).
MINIMAP_ENABLED = True
MINIMAP_X_RATIO = 0.00
MINIMAP_Y_RATIO = 0.78
MINIMAP_W_RATIO = 0.22
MINIMAP_H_RATIO = 0.22

def build_transforms(train=True, input_size=260):
    aug = [T.Resize((input_size, input_size)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)]
    return T.Compose(aug)


def _mask_minimap(img: Image.Image) -> Image.Image:
    """Mask out the bottom-left minimap by filling the rectangle with a
    nearby sample color. This avoids changing image size while removing
    the HUD content the network could memorize.
    """
    if not MINIMAP_ENABLED:
        return img
    w, h = img.size
    x1 = int(MINIMAP_X_RATIO * w)
    y1 = int(MINIMAP_Y_RATIO * h)
    x2 = int(x1 + MINIMAP_W_RATIO * w)
    y2 = int(y1 + MINIMAP_H_RATIO * h)
    x2 = min(w, max(x1 + 1, x2))
    y2 = min(h, max(y1 + 1, y2))

    # choose a sample pixel just to the right of the minimap box if possible
    sample_x = min(w - 1, x2 + 5)
    sample_y = min(h - 1, y2 - 5)
    try:
        fill = img.getpixel((sample_x, sample_y))
    except Exception:
        # fallback to a mid-gray if sampling fails
        fill = (128, 128, 128)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=fill)
    return img


# local numeric filename sort used for session ordering (e.g. frame_000012.jpg)
_num = re.compile(r"(\d+)")
def frame_sort_key(name: str):
    m = _num.search(name)
    return int(m.group(1)) if m else name

class FramesControls(Dataset):
    def __init__(self, frames_dir, labels_dir, split_file=None, train=True, input_size=260, use_prev=0, oversample_turns=False, turn_threshold=0.15):
        """
        frames_dir: root folder with session subfolders containing frames
        labels_dir: mirrored folder with JSON labels
        use_prev: number of previous frames to stack (0,1,2,...)
        oversample_turns: if True, duplicate turn samples to balance the dataset
        turn_threshold: steering magnitude above which a sample is considered a "turn"
        """
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.train = train
        self.tf = build_transforms(train, input_size)
        # shared augmentations (sampled once per stacked sequence)
        # INCREASED to fight overfitting from oversampling
        if train:
            self._color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)
            self._rand_affine = T.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR
            )
        else:
            self._color_jitter = None
            self._rand_affine = None
        # number of previous frames to include (0 => single frame)
        self.use_prev = int(use_prev)
        self.oversample_turns = oversample_turns and train
        self.turn_threshold = turn_threshold

        # Recursively list frames under frames_dir and keep only those that have
        # a corresponding label JSON in labels_dir. We'll organize frames by
        # session so we can only include examples that have the required
        # previous-frame context available within the same session.
        exts = {'.jpg', '.jpeg', '.png'}
        sessions = {}
        for root, _, files in os.walk(frames_dir):
            rel_root = os.path.relpath(root, frames_dir).replace('\\', '/')
            # rel_root '.' means frames are at root; treat session as ''
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    rel = (rel_root + '/' + f).lstrip('./').lstrip('/') if rel_root != '.' else f
                    sessions.setdefault(rel.split('/')[0], []).append(rel)

        # sort frames within each session using the numeric sort key
        for s, flist in sessions.items():
            sessions[s] = sorted(flist, key=frame_sort_key)

        # keep sessions accessible for __getitem__ (needed to find previous frames)
        self._sessions = sessions

        # Build the final item list: include only frames that have a label and
        # have enough previous frames in the same session.
        self.items = []
        # map rel -> (session, idx_in_session) for fast lookup
        self._session_index = {}
        for s, flist in sessions.items():
            for idx, rel in enumerate(flist):
                stem = os.path.splitext(rel)[0]
                label_path = os.path.join(labels_dir, stem + '.json')
                if os.path.exists(label_path) and idx - self.use_prev >= 0:
                    self.items.append(rel)
                    self._session_index[rel] = (s, idx)

        # If a split_file is provided it may contain basenames or relative paths
        # (without extension). Accept either form when filtering.
        if split_file:
            keep_lines = [l.strip() for l in open(split_file) if l.strip()]
            keep_basenames = set()
            keep_relpaths = set()
            for k in keep_lines:
                kp = k.replace('\\', '/')
                keep_basenames.add(os.path.splitext(os.path.basename(kp))[0])
                keep_relpaths.add(os.path.splitext(kp)[0])

            def keep_item(rel):
                base = os.path.splitext(os.path.basename(rel))[0]
                relstem = os.path.splitext(rel)[0]
                return (base in keep_basenames) or (relstem in keep_relpaths)

            self.items = [n for n in self.items if keep_item(n)]

        # Oversample turns: duplicate items where |steer| > threshold
        if self.oversample_turns:
            turn_items = []
            straight_items = []
            for item in self.items:
                try:
                    steer, _, _, _ = self._load_label(item)
                    if abs(steer) > self.turn_threshold:
                        turn_items.append(item)
                    else:
                        straight_items.append(item)
                except:
                    straight_items.append(item)
            
            # Calculate how many times to duplicate turns to balance
            if turn_items and straight_items:
                ratio = len(straight_items) / len(turn_items)
                # Cap at 2x to avoid overfitting on duplicated samples
                repeat = min(2, max(1, int(ratio)))
                print(f"[Dataset] Oversampling turns: {len(turn_items)} turn samples x{repeat}, {len(straight_items)} straight samples")
                self.items = straight_items + turn_items * repeat
            else:
                print(f"[Dataset] No oversampling: {len(turn_items)} turns, {len(straight_items)} straight")

    def __len__(self): return len(self.items)

    def _sample_aug_params(self, img_size):
        color_params = None
        affine_params = None
        if self._color_jitter is not None:
            color_params = T.ColorJitter.get_params(
                self._color_jitter.brightness,
                self._color_jitter.contrast,
                self._color_jitter.saturation,
                self._color_jitter.hue,
            )
        if self._rand_affine is not None:
            affine_params = T.RandomAffine.get_params(
                self._rand_affine.degrees,
                self._rand_affine.translate,
                self._rand_affine.scale,
                self._rand_affine.shear,
                img_size[::-1] if isinstance(img_size, tuple) else img_size,
            )
        return color_params, affine_params

    def _apply_color_jitter(self, img, params):
        if params is None:
            return img
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        # fn_idx is a permutation tensor describing the order of operations
        for fn_id in fn_idx.tolist():
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        return img

    def _apply_transforms(self, img, color_params=None, affine_params=None):
        if color_params is not None:
            img = self._apply_color_jitter(img, color_params)
        if affine_params is not None and self._rand_affine is not None:
            angle, translations, scale, shear = affine_params
            img = TF.affine(
                img,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=getattr(self._rand_affine, 'interpolation', InterpolationMode.BILINEAR),
                fill=getattr(self._rand_affine, 'fill', None),
                center=getattr(self._rand_affine, 'center', None),
            )
        return self.tf(img)

    def _load_label(self, name):
        stem = os.path.splitext(name)[0]
        # support nested session folders in labels_dir as well
        jpath = os.path.join(self.labels_dir, *stem.split('/')) + ".json"
        try:
            with open(jpath, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
        except Exception as e:
            raise RuntimeError(f"Failed to load label JSON '{jpath}': {e}")

        if j is None:
            raise RuntimeError(f"Label JSON '{jpath}' contains null / empty content")

        # Some label files wrap controls under a top-level 'controls' key.
        # Accept either form and provide safe defaults if keys are missing.
        data = j.get('controls', j) if isinstance(j, dict) else j

        try:
            steer = float(data.get('steer', 0.0))
            thr = float(data.get('throttle', 0.0))
            brk = float(data.get('brake', 0.0))
        except Exception as e:
            raise RuntimeError(f"Invalid label contents in '{jpath}': {e}")

        return steer, thr, brk, data

    def __getitem__(self, idx):
        name = self.items[idx]
        img = Image.open(os.path.join(self.frames_dir, *name.split('/'))).convert("RGB")

        # Optional: simple road crop (hide HUD). Tune these for your capture.
        w, h = img.size
        top_crop = int(0.10*h)       # remove a bit of sky/hud
        bottom_crop = h              # keep road fully
        img = img.crop((0, top_crop, w, bottom_crop))

        # Mask out the bottom-left minimap/HUD region so the model can't
        # trivially learn to read it. This preserves image size and aspect
        # while removing the shortcut.
        img = _mask_minimap(img)

        color_fn = None
        affine_params = None
        if self.train:
            color_fn, affine_params = self._sample_aug_params(img.size)

        x = self._apply_transforms(img, color_fn, affine_params)             # [3,H,W]
        if self.use_prev:
            # stack previous N frames from the same session in chronological order
            s, pos = self._session_index[name]
            prev_tensors = []
            for p in range(self.use_prev, 0, -1):
                prev_rel = self._sessions[s][pos - p]
                prev_img = Image.open(os.path.join(self.frames_dir, *prev_rel.split('/'))).convert("RGB")
                prev_img = prev_img.crop((0, top_crop, w, bottom_crop))
                prev_img = _mask_minimap(prev_img)
                prev_tensors.append(self._apply_transforms(prev_img, color_fn, affine_params))
            # concatenate [t-n, ..., t-1, t]
            x = torch.cat(prev_tensors + [x], dim=0)

        steer, thr, brk, raw = self._load_label(name)
        y = torch.tensor([steer, thr, brk], dtype=torch.float32)

        # Optional aux label: traffic light state → int class
        tlight_map = {"red":0,"yellow":1,"green":2,"none":3}
        tlight = raw.get("perception", {}).get("traffic_light_state", "none")
        tcls = torch.tensor(tlight_map.get(tlight, 3), dtype=torch.long)

        return x, y, tcls
