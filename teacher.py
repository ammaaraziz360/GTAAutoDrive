import base64, json, os, time, logging, argparse, sys
from openai import OpenAI

client = OpenAI(api_key="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")

frames_dir = "./data/frames_teacher"
labels_dir = "./data/labels"
os.makedirs(labels_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Prepare list of frames and counters
frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# CLI: allow processing a single frame or limiting runs
ap = argparse.ArgumentParser(description="Teacher labeling script")
ap.add_argument("--frame", type=str, default=None, help="process only this frame (e.g. 000080 or frame_000080.jpg)")
ap.add_argument("--max", type=int, default=50, help="maximum number of labels to generate (default 50)")
args = ap.parse_args()

if args.frame:
    requested = args.frame
    # normalize requested to possible filenames
    possible = set()
    if requested.lower().endswith('.jpg'):
        possible.add(requested)
        possible.add(requested.replace('.jpg',''))
    else:
        possible.add(requested)
        possible.add(requested if requested.startswith('frame_') else f'frame_{requested}')
        possible.add(f"{requested}.jpg")
        possible.add(f"frame_{requested}.jpg")

    matched = [f for f in frames if os.path.splitext(f)[0] in {os.path.splitext(p)[0] for p in possible}]
    if not matched:
        logging.error("Requested frame '%s' not found in %s", requested, frames_dir)
        logging.info("Available frames: %s", ", ".join(frames[:20]) + ("..." if len(frames) > 20 else ""))
        sys.exit(1)
    frames = matched
total = len(frames)
logging.info("Found %d frames in %s", total, frames_dir)

processed = 0
skipped = 0
bad_json = 0
errors = 0
start_time = time.time()

run = 0
# Delay between requests (seconds). Can be overridden via environment variable REQUEST_DELAY
request_delay = float(os.getenv("REQUEST_DELAY", "10.0"))

for idx, fname in enumerate(frames, start=1):
    # Skip if label already exists
    label_path = os.path.join(labels_dir, fname.replace(".jpg", ".json"))
    if os.path.exists(label_path):
        skipped += 1
        logging.info("Label exists, skipping %s (%d/%d)", fname, idx, total)
        continue

    logging.info("Processing %s (%d/%d)...", fname, idx, total)

    # Load & encode image
    with open(os.path.join(frames_dir, fname), "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    teacher_prompt = (
    "You're an expert human driver inside a realistic driving simulator. "
    "You’ll be shown a frame from the front camera of your car. "
    "Based only on what you can see, output a JSON object describing what "
    "a safe and skilled human driver would do right now. "
    "Respond with ONLY valid JSON (no explanations outside the JSON). "
    "IMPORTANT CLARIFICATION: interpret steering direction relative to the car’s forward motion "
    "(not the camera view or road perspective). "
    "In your JSON: "
    "'steer' ranges from -1 to 1 (-1 = full LEFT turn, 0 = straight, +1 = full RIGHT turn); "
    "'throttle' ranges 0–1 (0 = no gas, 1 = full acceleration); "
    "'brake' ranges 0–1 (0 = no brake, 1 = full brake); "
    "'reason_short' is a single concise sentence explaining why you chose those actions. "
    "OUTPUT ONLY JSON NO CODE BLOCKS, like: "
    "{\"steer\": -0.2, \"throttle\": 0.4, \"brake\": 0.0, \"reason_short\": \"Turning left through intersection with clear road ahead.\"}"
)


    # Query the teacher (GPT-4-Vision or GPT-5)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o, gpt-5, etc.
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", 
                     "text": teacher_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }]
        )

        # Parse JSON output
        text = resp.choices[0].message.content
        try:
            data = json.loads(text)
        except Exception:
            bad_json += 1
            logging.warning("Bad JSON for %s: %s", fname, text)
            continue

        # Save label
        with open(label_path, "w") as out:
            json.dump(data, out, indent=2)

        processed += 1
        run += 1
        logging.info("Saved label for %s (%d/%d) — processed=%d skipped=%d bad_json=%d", fname, idx, total, processed, skipped, bad_json)

        # Respect configured delay between requests to avoid rate limits
        if request_delay > 0:
            logging.debug("Sleeping %.2fs before next request", request_delay)
            time.sleep(request_delay)

        # periodic progress summary
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / idx if idx else 0
            remaining = total - idx
            eta = remaining * avg
            logging.info("Progress: %d/%d, elapsed=%.1fs, avg=%.2fs/frame, ETA=%.1fs", idx, total, elapsed, avg, eta)
    
        if run == args.max:
            break

    except Exception as e:
        errors += 1
        logging.exception("Error processing %s: %s", fname, e)
        # Backoff: sleep a bit longer on errors to reduce subsequent immediate failures
        backoff = max(request_delay * 2, 1.0)
        logging.info("Sleeping %.1fs after error before continuing", backoff)
        time.sleep(backoff)
        continue

# Final summary
elapsed = time.time() - start_time
logging.info("Done. processed=%d skipped=%d bad_json=%d errors=%d total_frames=%d elapsed=%.1fs", processed, skipped, bad_json, errors, total, elapsed)
