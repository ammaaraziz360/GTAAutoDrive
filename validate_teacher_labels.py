# validate_labels.py
import os, json, re, csv, math
from glob import glob

LABEL_DIR = "data/labels"          # folder with *.json per frame
OUT_CSV   = "label_qc_report.csv"

# --- Heuristic thresholds (tune as you like) ---
STEER_FLIP_THRESH   = 0.6   # flag if sign flips and |Δsteer| > this
STEER_JUMP_THRESH   = 0.35  # large |Δsteer| between consecutive frames
THR_JUMP_THRESH     = 0.40
BRK_JUMP_THRESH     = 0.40
DUAL_PEDAL_THRESH   = 0.25  # throttle AND brake both above this
RED_BRAKE_MIN       = 0.5   # want brake > this when light is red
RED_THR_MAX         = 0.1   # want throttle < this when light is red

# --- Helpers ---
def frame_key(path):
    # supports names like frame_000123.json or 000123.json
    b = os.path.basename(path)
    m = re.search(r'(\d+)', b)
    return int(m.group(1)) if m else b

def load_labels():
    files = sorted(glob(os.path.join(LABEL_DIR, "*.json")), key=frame_key)
    rows = []
    for f in files:
        try:
            j = json.load(open(f, "r", encoding="utf-8"))
            c = j.get("controls", j)  # allow flat or nested
            steer = float(c["steer"])
            thr   = float(c["throttle"])
            brk   = float(c["brake"])
            tlight = (j.get("perception") or {}).get("traffic_light_state")
            rows.append({
                "file": os.path.basename(f),
                "steer": steer, "throttle": thr, "brake": brk,
                "traffic_light_state": tlight
            })
        except Exception as e:
            rows.append({"file": os.path.basename(f), "error": str(e)})
    return rows

def sign(x): 
    return 0 if abs(x) < 1e-6 else (1 if x>0 else -1)

rows = load_labels()

# compute diffs
issues = []
for i, r in enumerate(rows):
    if "error" in r:
        issues.append({**r, "issue": "parse_error"})
        continue
    # dual pedal
    if r["throttle"] > DUAL_PEDAL_THRESH and r["brake"] > DUAL_PEDAL_THRESH:
        issues.append({**r, "issue": "dual_pedal"})
    # red-light compliance (optional)
    if r.get("traffic_light_state") == "red":
        if not (r["brake"] >= RED_BRAKE_MIN and r["throttle"] <= RED_THR_MAX):
            issues.append({**r, "issue": "red_light_noncompliant"})
    # temporal checks
    if i>0 and "error" not in rows[i-1]:
        p = rows[i-1]
        ds = r["steer"] - p["steer"]
        dt = r["throttle"] - p["throttle"]
        db = r["brake"] - p["brake"]
        if abs(ds) > STEER_JUMP_THRESH:
            issues.append({**r, "issue": f"steer_jump_{ds:+.2f}"})
        if sign(r["steer"]) != 0 and sign(p["steer"]) != 0:
            if sign(r["steer"]) != sign(p["steer"]) and abs(ds) > STEER_FLIP_THRESH:
                issues.append({**r, "issue": f"steer_sign_flip_{ds:+.2f}"})
        if abs(dt) > THR_JUMP_THRESH:
            issues.append({**r, "issue": f"throttle_jump_{dt:+.2f}"})
        if abs(db) > BRK_JUMP_THRESH:
            issues.append({**r, "issue": f"brake_jump_{db:+.2f}"})

# write CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file","steer","throttle","brake","traffic_light_state","issue"])
    for e in issues:
        w.writerow([e.get("file"), e.get("steer"), e.get("throttle"),
                    e.get("brake"), e.get("traffic_light_state"), e.get("issue")])

# simple summary
total = len([r for r in rows if "error" not in r])
print(f"Checked {total} labeled frames.")
print(f"Flagged {len(issues)} potential issues → see {OUT_CSV}")
by_type = {}
for e in issues:
    by_type[e["issue"].split("_")[0]] = by_type.get(e["issue"].split("_")[0], 0) + 1
print("Issue counts:", by_type)
