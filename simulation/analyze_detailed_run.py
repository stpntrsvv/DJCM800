"""Квалификация фактически посещённых областей подробных ламповых законов."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from full_mna import ROOT, build

def relative_voltage(c, states, node, cathode):
    v = np.zeros(len(states)) if node == "0" else states[:, c.index[node]]
    k = np.zeros(len(states)) if cathode == "0" else states[:, c.index[cathode]]
    return v-k


def inspect(c, states):
    result = {}
    for part in c.parts:
        if part[0] != "T":
            continue
        _, name, model, plate, grid, cathode, screen = part
        va = relative_voltage(c, states, plate, cathode)
        vg = relative_voltage(c, states, grid, cathode)
        row = dict(model=model, samples=len(states), positive_grid_fraction=float(np.mean(vg > 0)))
        if model.startswith("dempwolf:"):
            outside = (va < 20) | (va > 300) | (vg < -5) | (vg > 3)
            row.update(reference_box="Va=20..300 V, Vg1=-5..3 V",
                       outside_fraction=float(np.mean(outside)), outside_samples=int(np.count_nonzero(outside)))
        else:
            vs = relative_voltage(c, states, screen, cathode)
            # This is the grid exercised by our implementation/Jacobian tests,
            # not a claim that the physical EL34 is qualified throughout it.
            outside = (va < .1) | (va > 600) | (vg < -80) | (vg > 3) | (vs < 50) | (vs > 550)
            continuation = (va < 0) | (vs <= 1e-6)
            row.update(reference_box="numerically tested box: Va=.1..600 V, Vg1=-80..3 V, Vg2=50..550 V",
                       outside_fraction=float(np.mean(outside)), outside_samples=int(np.count_nonzero(outside)),
                       continuation_samples=int(np.count_nonzero(continuation)))
        result[name] = row
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--triode", choices=("RSD-1", "RSD-2", "EHX-1"), default="RSD-1")
    args = parser.parse_args()
    slug = args.triode.lower().replace("-", "")
    raw = ROOT / "simulation/raw" / f"settling_power_detailed_{slug}"
    dest = ROOT / "simulation/experiments" / f"detailed_{slug}_qualification"
    c = build(True, amplitude=0., controls=.5, tube_set=f"detailed:{args.triode}")
    scenarios = {"idle": np.load(raw / "idle.npz")["states"]}
    for mv in (25, 100, 500):
        saved = np.load(raw / f"burst_{mv}mv.npz")
        scenarios[f"burst_{mv}mv"] = saved["on"]
        scenarios[f"release_{mv}mv"] = saved["off"]
    result = dict(status="qualification_incomplete", tube_set=f"detailed:{args.triode}",
                  scenarios={name: inspect(c, states) for name, states in scenarios.items()})
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines = ["# Область применения подробных ламп в полной MNA", "",
             "Этот отчёт сопоставляет фактически посещённые точки с областью исходной проверки.",
             "Выход за неё не доказывает ошибку закона, но запрещает считать физику аттестованной.", "",
             "| Сценарий | Лампа | Модель | Вне проверенной области | Положительная Vg1 |",
             "|:---|:---|:---|---:|---:|"]
    for scenario, tubes in result["scenarios"].items():
        for name, row in tubes.items():
            lines.append(f"| {scenario} | {name} | {row['model']} | {row['outside_fraction']:.3%} | {row['positive_grid_fraction']:.3%} |")
    lines += ["", "## Вывод", "",
              "Рабочая точка без сигнала лежит внутри исследованных численных областей.",
              "Сильный сигнал посещает неподтверждённые области ECC83 и EL34; особенно важны",
              "глубокая отсечка управляющих сеток, Va EL34 выше 600 В и положительная Vg1.",
              "Численное продолжение Reefman на отрицательный Va/неположительный Vg2 в сохранённых",
              "траекториях не использовалось. Следующий физический опыт должен расширять данные именно здесь.", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str((dest/'report.md').relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
