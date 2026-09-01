import os, base64, json, time, random, re
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI

# -------- Paths (edit these) --------
FRAMES_ROOT = Path("data/frames")   # per-session folders of JPGs/PNGs you send to teacher
LABELS_ROOT = Path("data/labels_teacher")   # mirrored output tree for JSON labels
LABELS_ROOT.mkdir(parents=True, exist_ok=True)

# -------- Temporal config --------
WINDOW = 3       # use [t-1, t, t+1]; set 5 for [t-2..t+2]
STRIDE = 2       # slide by this many frames between windows (2 cuts API calls ~in half)
FPS_NOTE = "Frames are ~0.2s apart; label the MIDDLE frame."

# -------- Model + rate limiting --------
MODEL = "gpt-5-mini"  # economical; switch to "gpt-4o" for tricky scenes
MAX_RETRIES = 6

# -------- Prompt with perspective clarification --------
TEACHER_PROMPT = (
    "You're an expert human driver inside a realistic driving simulator. "
    "You will be shown one or more consecutive frames from the FRONT camera. "
    "Based only on what you can see, output a JSON object describing what a safe "
    "and skilled human driver would do RIGHT NOW for the MIDDLE frame. "
    "Respond with ONLY valid JSON (no explanations outside the JSON). "
    "IMPORTANT CLARIFICATION: interpret steering relative to the car’s forward motion "
    "(NOT the camera/view perspective). "
    "In your JSON: 'steer' ∈ [-1,1] (−1=full LEFT, 0=straight, +1=full RIGHT); "
    "'throttle' ∈ [0,1] (0=no gas, 1=full acceleration); "
    "'brake' ∈ [0,1] (0=no brake, 1=full brake); "
    "'reason_short' is one concise sentence explaining your choice. "
    f"{FPS_NOTE} "
    "OUTPUT ONLY JSON, e.g. "
    "{\"steer\": -0.2, \"throttle\": 0.4, \"brake\": 0.0, \"reason_short\": \"Turning left through intersection on wet road.\"}"
)

# -------- Helpers --------
client = OpenAI(api_key="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")

def to_b64(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")

_num = re.compile(r"(\d+)")

def frame_sort_key(name: str):
    m = _num.search(name)
    return int(m.group(1)) if m else name

def list_sessions(root: Path) -> List[Path]:
    sessions = []
    for dp, _, files in os.walk(root):
        if any(f.lower().endswith((".jpg", ".png")) for f in files):
            sessions.append(Path(dp))
    return sorted(sessions)

def sliding_windows(names: List[str], window: int, stride: int) -> List[Tuple[int, List[str]]]:
    """Return [(mid_index, [filenames in window])...] within one session."""
    if window % 2 == 0:  # ensure odd
        window += 1
    half = window // 2
    out = []
    i = half
    while i < len(names) - half:
        chunk = names[i - half : i + half + 1]
        out.append((i, chunk))
        i += stride
    return out

def robust_call(msg_content) -> dict:
    delay = 1.0
    for _ in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": msg_content}],
                text={"format": {"type": "json_object"}}
            )
            data = json.loads(resp.output_text)
            # clamp sanity (keeps labels clean)
            data["steer"] = float(max(-1.0, min(1.0, data.get("steer", 0.0))))
            data["throttle"] = float(max(0.0, min(1.0, data.get("throttle", 0.0))))
            data["brake"] = float(max(0.0, min(1.0, data.get("brake", 0.0))))
            return data
        except Exception as e:
            # backoff on 429/5xx/timeouts
            ra = None
            try:
                ra = getattr(e, "response").headers.get("retry-after")
            except Exception:
                pass
            emsg = str(e).lower()
            if "rate limit" in emsg or "429" in emsg or ra or any(x in emsg for x in ["502","503","504","timeout"]):
                sleep_s = float(ra) if ra else delay + random.uniform(0, 0.5 * delay)
                time.sleep(sleep_s)
                delay = min(delay * 2, 30)
                continue
            raise

def label_session(sess_dir: Path):
    rel = sess_dir.relative_to(FRAMES_ROOT)
    out_dir = LABELS_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get all frame paths and their corresponding label paths
    names = sorted([n for n in os.listdir(sess_dir) if n.lower().endswith((".jpg",".png"))], key=frame_sort_key)
    if not names:
        print(f"[skip] no frames: {rel}")
        return
    
    # Check which frames need labels
    need_labels = []
    for name in names:
        label_path = out_dir / (Path(name).stem + ".json")
        if not label_path.exists():
            need_labels.append(name)
    
    if not need_labels:
        print(f"[skip] fully labeled: {rel}")
        return
    
    print(f"[session] {rel} -> {len(need_labels)} frames need labels")

    windows = sliding_windows(names, WINDOW, STRIDE)
    unlabeled_windows = []
    
    # Only process windows where the middle frame needs a label
    for mid_idx, chunk in windows:
        mid_name = names[mid_idx]
        label_path = out_dir / (Path(mid_name).stem + ".json")
        if not label_path.exists():
            unlabeled_windows.append((mid_idx, chunk))
    
    print(f"[processing] {len(unlabeled_windows)} windows to generate labels")
    
    for mid_idx, chunk in unlabeled_windows:
        # Build multi-image prompt (label the middle)
        content = [{"type": "input_text", "text": TEACHER_PROMPT}]
        for fname in chunk:
            content.append({"type": "input_image", "image_url": to_b64(sess_dir / fname)})

        data = robust_call(content)

        mid_name = names[mid_idx]
        # save one JSON per middle frame
        (out_dir / (Path(mid_name).stem + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")

def main():
    sessions = list_sessions(FRAMES_ROOT)
    if not sessions:
        print(f"No sessions found under {FRAMES_ROOT}")
        return
    for s in sessions:
        label_session(s)
    print("✓ Done.")

if __name__ == "__main__":
    main()
