"""Reproducible full-signal-path ngspice baseline; run from any directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NODES = "p1 k1 p2 k2 p3 k3 cf bpre1 bpre2 bpi pi_p1 pi_p2 pi_k pi_tail feedback bplus bscreen plate_a plate_b bias s4 out".split()
TRACE = "in p1 p2 p3 cf master_out pi_p1 pi_p2 out bplus bscreen drive1 drive2".split()


def executable() -> Path:
    candidates = [os.environ.get("NGSPICE_EXE"), shutil.which("ngspice"),
                  ROOT.parent / ".tools/ngspice-47/Spice64/bin/ngspice_con.exe"]
    for item in candidates:
        if item and Path(item).is_file():
            return Path(item).resolve()
    raise FileNotFoundError("Set NGSPICE_EXE to ngspice executable")


def table(path: Path) -> np.ndarray:
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"Nonfinite data: {path}")
    return data


def run_case(name: str, amplitude: float, step_us: float, nfb: int = 1,
             controls: float = .5, stop: float = .06) -> dict:
    folder = ROOT / "simulation/raw/reference_baseline" / name
    folder.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "simulation/ngspice").glob("*"):
        if source.is_file():
            shutil.copy2(source, folder / source.name)
    deck = f"""JCM800 2203 24-Apr-1981 initial full-chain reference
.include jcm800_2203_1981.inc
.param NFB={nfb} GAIN={controls} MASTER={controls}
Vin in 0 DC 0 AC 1 SIN(0 {amplitude} 1000)
.options reltol=1e-6 abstol=1e-11 vntol=1e-8 itl1=500 itl4=1000 method=gear maxord=2
.control
set noaskquit
set wr_vecnames
set wr_singlescale
op
wrdata op.txt {' '.join('v('+n+')' for n in NODES)}
ac dec 80 10 100000
let gain_db = db(v(out))
let phase_deg = 180/PI*cph(v(out))
wrdata ac.txt gain_db phase_deg
tran {step_us}u {stop} 0 {step_us}u
wrdata tran.txt {' '.join('v('+n+')' for n in TRACE)}
quit
.endc
.end
"""
    (folder / "run.cir").write_text(deck, encoding="ascii")
    result = subprocess.run([str(executable()), "-b", "run.cir"], cwd=folder,
                            capture_output=True, text=True, errors="replace", timeout=180)
    log = result.stdout + "\n" + result.stderr
    (folder / "ngspice.log").write_text(log, encoding="utf-8")
    if result.returncode or any(t in log.lower() for t in ("timestep too small", "fatal error", "error:")):
        raise RuntimeError(f"ngspice failed: {folder / 'ngspice.log'}\n{log[-2500:]}")
    op, ac, tr = (table(folder / (kind + ".txt")) for kind in ("op", "ac", "tran"))
    if tr[-1, 0] < stop * .999:
        raise RuntimeError("Incomplete transient")
    out = tr[:, TRACE.index("out") + 1]
    # Uniform 192 kHz measurement grid, last 20ms, no gain/delay fitting.
    grid = np.arange(stop - .02, stop, 1 / 192000)
    y = np.interp(grid, tr[:, 0], out)
    metrics = dict(name=name, amplitude_v=amplitude, max_step_us=step_us, nfb=nfb,
                   gain=controls, master=controls, rows=len(tr),
                   output_rms_v=float(np.sqrt(np.mean(y*y))),
                   output_power_w=float(np.mean(y*y) / 16),
                   output_peak_v=float(np.max(np.abs(out))),
                   dc_v=dict(zip(NODES, op[0, 1:].tolist())))
    print(f"{name}: peak {metrics['output_peak_v']:.4f} V, {metrics['output_power_w']:.4f} W", flush=True)
    return dict(metrics=metrics, tr=tr, ac=ac, grid=grid, y=y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only baseline smoke run")
    args = parser.parse_args()
    cases = [run_case("baseline_25mv_2us", .025, 2)]
    if not args.quick:
        cases += [run_case("baseline_25mv_1us", .025, 1),
                  run_case("drive_100mv_2us", .1, 2),
                  run_case("drive_500mv_2us", .5, 2),
                  run_case("nfb_open_25mv_2us", .025, 2, nfb=0),
                  run_case("full_gain_100mv_2us", .1, 2, controls=.999)]
    dest = ROOT / "simulation/experiments/reference_baseline"
    dest.mkdir(parents=True, exist_ok=True)
    base = cases[0]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "simulation/ngspice").glob("*") if p.is_file()}
    summary = {"netlist_sha256": hashes, "cases": [c["metrics"] for c in cases]}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for case in cases:
        axes[0].plot(case["grid"] * 1000, case["y"], label=case["metrics"]["name"])
    axes[0].set(xlim=(50, 55), xlabel="Time, ms", ylabel="16 ohm output, V")
    axes[0].legend(fontsize=7)
    for case in (cases[0], *[c for c in cases if c["metrics"]["nfb"] == 0]):
        axes[1].semilogx(case["ac"][:, 0], case["ac"][:, 1], label=case["metrics"]["name"])
    axes[1].set(xlabel="Frequency, Hz", ylabel="Small-signal gain, dB")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=.3)
    fig.savefig(dest / "reference.png", dpi=150)
    plt.close(fig)
    report = ["# Первый полный SPICE-прототип 2203 (1981)", "",
              "Автоматический отчёт `simulation/run_reference.py`. Это численный прототип,",
              "ещё не аттестованный эталон конкретного усилителя. См. ограничения в `docs/03 Модель.md`.", "",
              "ngspice, Gear порядка не выше 2, reltol=1e-6, abstol=1e-11 A, vntol=1e-8 V.",
              "Синус 1 кГц, длительность 60 мс, старт из рабочей точки. RMS последних 20 мс;",
              "электрическая нагрузка 16 Ом. Bass/Mid/Treble/Presence=0,5; Gain/Master в таблице.",
              "Позиции потенциометров — электрические доли сопротивления, не шкала ручек.", "",
              "| Опыт | Вход, В peak | Gain/Master | Шаг max, мкс | Выход peak, В | Мощность, Вт |",
              "|:---|---:|---:|---:|---:|---:|"]
    for case in cases:
        m = case["metrics"]
        report.append(f"| {m['name']} | {m['amplitude_v']} | {m['gain']} | {m['max_step_us']} | {m['output_peak_v']:.5f} | {m['output_power_w']:.5f} |")
    if not args.quick:
        err = base["y"] - cases[1]["y"]
        relative = float(np.linalg.norm(err) / np.linalg.norm(cases[1]["y"]))
        summary["step_refinement_relative_rms"] = relative
        report += ["", f"Уменьшение максимального шага 2 → 1 мкс: разность выхода {relative*100:.6f}% RMS.",
                   "Это локальная проверка одного сигнала; общая сходимость сетки ещё не установлена."]
    report += ["", "## Рабочая точка", "", "| Узел | В |", "|:---|---:|"]
    report += [f"| {n} | {v:.6f} |" for n, v in base["metrics"]["dc_v"].items()]
    report += ["", "![Переходные сигналы и АЧХ](reference.png)", "",
               "Все записанные значения конечны, переходные расчёты достигли 60 мс.",
               "Аппаратные такты STM32N6 не измерялись. Сетевые пульсации, магнитное насыщение",
               "и акустический кабинет отсутствуют. Параметры источника питания предварительные.", ""]
    (dest / "report.md").write_text("\n".join(report), encoding="utf-8")
    (dest / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
