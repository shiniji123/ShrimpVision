from pathlib import Path

BASE = Path(r"c:\Users\anuch\AntiProject\shrime_project")

trained  = BASE / "runs" / "shrimp_yolo26_gpu" / "weights" / "best.pt"
fallback = BASE / "yolo26n.pt"

print("=== Model File Check ===")
print(f"Trained  : {trained}")
print(f"  Exists : {trained.exists()}")
print(f"Fallback : {fallback}")
print(f"  Exists : {fallback.exists()}")

webapp = BASE / "webapp"
checks = [
    "app.py",
    "config.py",
    "inference_engine.py",
    "behavior_analyzer.py",
    "templates/index.html",
    "static/css/index.css",
    "static/js/app.js",
    "static/js/upload.js",
    "static/js/dashboard.js",
    "static/js/video-stream.js",
]

print("\n=== Webapp Files ===")
all_ok = True
for rel in checks:
    f = webapp / rel
    ok = f.exists()
    if not ok:
        all_ok = False
    status = "OK     " if ok else "MISSING"
    print(f"  [{status}] {rel}")

print()
if all_ok:
    print("All webapp files present.")
else:
    print("Some files MISSING!")

print()
if trained.exists():
    print("=> Will use: TRAINED model (best.pt) - fine-tuned for shrimp!")
elif fallback.exists():
    print("=> Will use: FALLBACK yolo26n.pt (pretrained, not fine-tuned for shrimp yet)")
    print("   Detection will still work but accuracy will be low until training is complete.")
else:
    print("=> ERROR: No model found!")
