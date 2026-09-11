"""Оркестратор долгого опыта закрытия лампового блока."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation/raw/tube_closure"
RAW.mkdir(parents=True, exist_ok=True)
steps = [
    ("isolated_validation", [sys.executable, "simulation/run_tube_validation.py"]),
    ("integration_smoke", [sys.executable, "simulation/check_detailed_mna.py", "--triode", "RSD-1"]),
    ("closure_matrix", [sys.executable, "simulation/run_tube_closure.py"]),
]
summary = {"status": "running", "steps": []}
for name, command in steps:
    log_path = RAW / f"{name}.log"
    started = time.monotonic()
    print(f"\n=== {name} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    summary["steps"].append(dict(name=name, returncode=code, elapsed_s=time.monotonic()-started,
                                 log=str(log_path.relative_to(ROOT))))
    summary["status"] = "failed" if code else "running"
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if code:
        raise SystemExit(code)
summary["status"] = "complete"
(RAW / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print("\nГотово:", RAW / "summary.json")
