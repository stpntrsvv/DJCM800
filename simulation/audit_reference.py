"""Audit saved reference runs without rerunning the circuit."""
import hashlib
import json
from run_reference import ROOT, executable, table
import numpy as np


def main():
    folder = ROOT / "simulation/experiments/reference_baseline"
    summary = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    for filename, digest in summary["netlist_sha256"].items():
        if hashlib.sha256((ROOT / "simulation/ngspice" / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Saved reference is stale: " + filename)
    raw = ROOT / "simulation/raw/reference_baseline"
    closed = table(raw / "baseline_25mv_2us/ac.txt")
    opened = table(raw / "nfb_open_25mv_2us/ac.txt")
    idx = int(np.argmin(np.abs(closed[:, 0] - 1000)))
    reduction = float(opened[idx, 1] - closed[idx, 1])
    if reduction <= 0:
        raise AssertionError("Feedback does not reduce 1kHz gain")
    if summary["step_refinement_relative_rms"] > .001:
        raise AssertionError("Baseline 2us/1us disagreement exceeds 0.1%")
    # Voltage test point numbers from 100W MV column of factory preamp drawing.
    # Meter impedance is unspecified, so grid-node readings are not used here.
    references = [(1, "k2", 2.6), (2, "p2", 250), (3, "k1", 1.75),
                  (4, "p1", 210), (5, "bpre1", 280), (6, "bpre2", 290),
                  (7, "k3", 1), (8, "p3", 165), (9, "cf", 167),
                  (10, "bpi", 330), (12, "pi_p1", 220), (13, "pi_p2", 210),
                  (15, "pi_k", 37.5), (16, "pi_tail", 36.5), (17, "feedback", 11)]
    dc = summary["cases"][0]["dc_v"]
    lines = ["# Аудит первого связанного расчёта", "",
             "Проверены SHA-256 текущих netlist относительно сохранённого metrics.json.",
             "Источник DC-ориентиров: колонка **100W / MV** заводского листа",
             "[24-4-81](https://www.drtube.com/schematics/marshall/jcm800pr.gif).",
             "Это ориентировочные напряжения, не допуски приёмки; тип измерителя не указан.", "",
             "| Точка | Узел | Схема, В | Расчёт, В | Разность, % |",
             "|---:|:---|---:|---:|---:|"]
    for number, node, ref in references:
        lines.append(f"| {number} | {node} | {ref:g} | {dc[node]:.4f} | {100*(dc[node]/ref-1):+.3f} |")
    lines += ["", f"При 1 кГц ООС снижает малосигнальное усиление на **{reduction:.4f} дБ**.",
              "Это контроль знака воздействия при одной частоте, не измерение запаса устойчивости.", "",
              "На входе 25 мВ уменьшение maxstep 2 → 1 мкс даёт разность менее 0,1% RMS",
              "(точное значение — в основном отчёте). Сильные 100/500 мВ пока не прошли",
              "такое сгущение сетки. Длинная DI ещё не запускалась.", "",
              "Мощности порядка 225–235 Вт в основном отчёте относятся к сильно ограниченному",
              "сигналу в коротком переходном опыте и предварительному питанию/OT.",
              "Они не подтверждают паспортные 100 Вт чистой мощности и требуют отдельного",
              "анализа энергетического баланса, установления питания и моделей выходных ламп.", "",
              "Параметры Koren имеют известные расхождения с табличными режимами Philips:",
              "[отчёт по лампам](../tube_validation/report.md). Поэтому низкая численная ошибка",
              "не даёт оснований объявлять физический эталон готовым.", ""]
    (folder / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    version = {"ngspice_path": str(executable()),
               "ngspice_sha256": hashlib.sha256(executable().read_bytes()).hexdigest(),
               "nfb_reduction_1khz_db": reduction}
    (folder / "audit.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
    print(f"Saved-reference audit passed; NFB reduction {reduction:.4f} dB")


if __name__ == "__main__":
    main()
