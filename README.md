GTAAutoDrive

This repository contains a small end-to-end pipeline for training and running a driving agent on GTA V.

Key scripts
- `train.py` - training loop for StudentPolicy
- `model.py` - model definition
- `dataset.py` - dataset + minimap masking and frame stacking
- `grad_cam_steer.py` - Grad-CAM visualization for steer output
- `run_agent.py` - runtime agent that captures screen, runs the model and sends controls to GTA (supports vgamepad/keyboard)

Requirements
Run the helper to install Python deps:

  .\scripts\install_deps.ps1

Helper scripts (Windows PowerShell)
- `scripts/install_deps.ps1` - install Python dependencies listed in `requirements.txt`
- `scripts/run_train.ps1` - example wrapper to launch `train.py`
- `scripts/run_agent.ps1` - wrapper to launch the live agent with default GTA window
- `scripts/run_gradcam.ps1` - wrapper to run `grad_cam_steer.py`
- `scripts/start_tensorboard.ps1` - launch TensorBoard for the `runs/` folder

Examples

1) Generate frames from a video:

	ffmpeg -i clips/your_clip.mp4 -vf "fps=5:-1" -qscale:v 3 data/frames/session-000X/frame_%06d.jpg

2) Run training with 2 previous frames:

	.\scripts\run_train.ps1 -prev_frames 2 -bs 32 -epochs 50

3) Run the live agent (ensure ViGEmBus installed for XInput support or use keyboard fallback):

	.\scripts\run_agent.ps1 -window "Grand Theft Auto V" -ckpt checkpoints\best.pth -steer_multiplier 0.8 -throttle_multiplier 1.2

4) Run Grad-CAM visualization:

	.\scripts\run_gradcam.ps1 -frames_T 3 -ckpt checkpoints\best.pth -out_dir gradcam_out

 scp -i C:\Users\ammaa\Downloads\gtaselfdrive.pem ubuntu@104.171.202.135:~/gtaselfdrive/GTAAutoDrive/checkpoints\best.pth C:\Users\ammaa\OneDrive\Documents\GTAAutoDrive\checkpoints

```
 python run_agent.py --window "Grand Theft Auto V" --ckpt checkpoints\best.pth `
  --alpha-steer 0.2 --max-steer-change 0.15 `
  --alpha-throttle 0.5 --max-throttle-change 0.1 `
  --steer-multiplier 2.5 --fps 15
```
## Temporal teacher — important compatibility note

The repository includes a small LLM-based "teacher" for labeling frames (`teacher_label_temporal.py`). Be aware of an important temporal-context mismatch between the teacher script and the student dataset:

- `teacher_label_temporal.py` (default behavior)
	- Uses a sliding window of images to provide context to the model. By default `WINDOW = 3` and the script builds windows like `[t-1, t, t+1]` and asks the teacher to label the MIDDLE frame (`t`). This means the teacher can (and often will) use *future* visual information (frame `t+1`) when creating the label for `t`.

- `dataset.py` / `FramesControls` (student input)
	- The dataset supports stacking previous frames via the `use_prev` argument. For example `use_prev=2` produces an input of `[t-2, t-1, t]` (only past frames + current). At inference the student will only ever see past frames.

Why this matters
- If the teacher uses future frames to create labels but the student is trained only on past frames, the teacher labels may contain information the student cannot observe at inference time. That can reduce the usefulness of those labels or produce a student that cannot reproduce teacher behavior in live runs.

Recommended approaches
- Preferred: generate teacher labels from past frames only so the visual context matches the student. There are two ways to do this:
	1. Edit `teacher_label_temporal.py` to use a "past" window mode (make windows `[t-(W-1), ..., t]` and instruct the prompt to "label the LAST frame"). This keeps the teacher from peeking at future frames and aligns labels with `use_prev` in `FramesControls`.
	2. If you already have labels produced with the centered window and you intentionally want those stronger labels, be aware they may rely on future context and may not be fully reproducible by the student.

- Quick practical workaround (no code edit): copy teacher labels into the labels folder your training run expects, then train with `use_prev` configured to match the teacher mode you used.

Example PowerShell commands
- Copy labels produced by the temporal teacher into the default training labels folder:

```powershell
# copy all teacher labels into training labels folder (recursively)
Copy-Item -Path data\labels_teacher\* -Destination data\labels\ -Recurse -Force
```

- Quick check that the dataset can load stacked inputs (run from workspace root):

```powershell
python - <<'PY'
from dataset import FramesControls
ds = FramesControls('data/frames', 'data/labels', use_prev=2)
print('items:', len(ds))
if len(ds):
		x, y, tcls = ds[0]
		print('x.shape', x.shape, 'y', y)
PY
```

If you'd like I can also (optionally) add a small `WINDOW_MODE = "past"` flag to `teacher_label_temporal.py` and a short CLI to choose output folder; tell me if you want that implemented and I'll add it and run a quick smoke test that writes a few past-mode labels into `data/labels`.

### Run the teacher

Quick commands to run the teacher scripts from the repository root (PowerShell examples):

- Run the temporal teacher (uses the constants at the top of `teacher_label_temporal.py`; default writes to `data/labels_teacher`):

```powershell
python teacher_label_temporal.py
```

- Run the single-frame teacher script (writes JSON labels to `./data/labels` by default). Use `--max` to limit labels produced or `--frame` to target a single frame:

```powershell
python teacher.py --max 100
# or label a single frame (basename or filename):
python teacher.py --frame 000080
```

Notes:
- `teacher_label_temporal.py` currently builds windows according to the `WINDOW` and `STRIDE` constants near the top of the file and labels the MIDDLE frame of each window. If you want teacher labels that match a student trained on past-only inputs, edit `teacher_label_temporal.py` to use a past-mode window (e.g., `[t-2,t-1,t]`) and update the prompt to "label the LAST frame". I can add this option if you'd like.


