"""Возобновляемый оркестратор этапов 1–3: лампы, MNA, новое установление."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "simulation/raw/jcm800_pipeline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triode", choices=("RSD-1", "RSD-2", "EHX-1"), default="RSD-1")
    parser.add_argument("--start", choices=("validate", "connect", "settle", "analyze"), default="validate")
    parser.add_argument("--stop-after", choices=("validate", "connect", "settle", "analyze"), default="analyze")
    parser.add_argument("--settle-only", action="store_true",
                        help="Остановиться после нового периодического режима и баланса без атак")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(RAW / "matplotlib"))
    stages = [
        ("validate", [sys.executable, "simulation/run_tube_validation.py"]),
        ("connect", [sys.executable, "simulation/check_detailed_mna.py", "--triode", args.triode]),
        ("settle", [sys.executable, "simulation/run_settling.py", "--tube-set", f"detailed:{args.triode}"]
                   + (["--settle-only"] if args.settle_only else [])),
        ("analyze", [sys.executable, "simulation/analyze_detailed_run.py", "--triode", args.triode]),
    ]
    begin = [name for name, _ in stages].index(args.start)
    end = [name for name, _ in stages].index(args.stop_after)
    if args.settle_only and args.stop_after == "analyze":
        end = [name for name, _ in stages].index("settle")
    if end < begin:
        parser.error("--stop-after must not precede --start")
    summary = dict(status="running", triode=args.triode, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stages=[])
    summary_path = RAW / "summary.json"
    for name, command in stages[begin:end+1]:
        print(f"\n=== {name}: {' '.join(command)} ===", flush=True)
        started = time.monotonic()
        log_path = RAW / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        record = dict(name=name, command=command, returncode=code,
                      elapsed_s=time.monotonic()-started, log=str(log_path.relative_to(ROOT)))
        summary["stages"].append(record)
        summary["status"] = "failed" if code else "running"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        if code:
            print(f"Этап {name} завершился ошибкой. Итог: {summary_path}", file=sys.stderr)
            raise SystemExit(code)
    summary["status"] = "complete"
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\nГотово. Для следующего разбора достаточно файла {summary_path}")


if __name__ == "__main__":
    main()
