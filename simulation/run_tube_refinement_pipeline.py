"""Один запускаемый файл для продолжения сгущения лампового опыта."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
log_path = root / "simulation/raw/tube_closure/refinement.log"
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen([sys.executable, "simulation/run_tube_closure_refinement.py"], cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
raise SystemExit(process.wait())
