"""Qualify compact nonlinear assembly on the previous identical test matrix."""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import re
import time

from full_mna import ROOT, Circuit, build
from compact_mna import CompactCircuit
from reduced_mna import ReducedCircuit
import run_linear_reduction as comparison
import numpy as np
import matplotlib.pyplot as plt

DEST = ROOT / "simulation/experiments/compact_assembly"
RAW = ROOT / "simulation/raw/compact_assembly"
OPTIONS = comparison.OPTIONS
LIMITS = comparison.LIMITS


def pair(full=True):
    return tuple(build(full, amplitude=0., circuit_type=cls, **OPTIONS)
                 for cls in (Circuit, CompactCircuit))


def assembly_check(c, states):
    maxima = dict(current_a=0., jacobian_a_per_v=0., charge_c=0., capacitance_f=0.)
    for x in states:
        for key, a, b in zip(maxima, Circuit.nonlinear(c, x), c.nonlinear(x)):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-15)
            maxima[key] = max(maxima[key], float(np.max(np.abs(a-b))))
    return dict(states=len(states), max_absolute_difference=maxima)


def compare_previous_schur(c, row, arrays, quick):
    folder = ROOT / "simulation/raw/linear_reduction"
    if quick:
        folder = folder / "quick"
    source = folder / f"{row['name']}.npz"
    if not source.exists():
        return dict(status="not_available")
    with np.load(source) as prior:
        previous = prior["reduced"]
        with np.load(RAW / f"{row['name']}.npz") as current:
            if previous.shape != arrays[1].shape or not np.array_equal(prior["time"], current["time"]):
                return dict(status="different_grid")
        # Also compare full reference arrays: this independently checks that the
        # reused run actually has the same initialization, forcing and grid.
        if not np.array_equal(prior["full"], arrays[0]):
            return dict(status="different_reference_trajectory")
    difference = comparison.errors(c, previous, arrays[1])
    comparison.assert_errors(difference)
    return dict(status="compared", bitwise_equal=bool(np.array_equal(previous, arrays[1])),
                **difference, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def profile_cost(initial, start):
    """Warm imports/factorizations; separate unprofiled timings from cProfile."""
    result = {}
    classes = dict(full=Circuit, schur=ReducedCircuit, compact=CompactCircuit)
    steps, h, repeats = 200, .625e-6, 5
    for name, cls in classes.items():
        c = build(True, amplitude=.5, circuit_type=cls, **OPTIONS)
        c.step(initial, start+h, h)
        samples = []
        for _ in range(repeats):
            x = initial.copy()
            t = time.perf_counter()
            for j in range(steps):
                x, _ = c.step(x, start+(j+1)*h, h)
            samples.append(time.perf_counter()-t)
        x = initial.copy()
        profiler = cProfile.Profile()
        profiler.enable()
        for j in range(steps):
            x, _ = c.step(x, start+(j+1)*h, h)
        profiler.disable()
        stats = pstats.Stats(profiler)
        cost = {}
        for function in ("step", "newton", "nonlinear", "compact_nonlinear", "electrode",
                         "linearized_delta", "compact_delta"):
            rows = [v for k, v in stats.stats.items() if k[2] == function]
            cost[function] = dict(calls=sum(v[1] for v in rows), self_s=sum(v[2] for v in rows),
                                  cumulative_s=sum(v[3] for v in rows))
        result[name] = dict(steps=steps, repeats=repeats, unprofiled_s=samples,
                            median_unprofiled_s=float(np.median(samples)), profile=cost)
        print("profile", name, result[name]["median_unprofiled_s"], flush=True)
    return result


def render(result):
    status = result["status"]
    lines = ["# Сокращённая сборка нелинейностей JCM800", "", f"Статус: **{status}**.", "",
             "Сравнение полной MNA и CompactCircuit при прежней физике Dempwolf RSD-1 + Reefman,",
             "RGI+Rs=2001 Ом и паспортных ёмкостях. Система Ньютона остаётся 39×39;",
             "все 98 неизвестных, состояния накопителей и критерии полной невязки сохранены.", "",
             "Каждый элемент штампует только свои 1–4 узла. Якобиан и производная заряда",
             "собираются в сокращённом пространстве; в поиске шага и построении истории",
             "собираются только токи и заряды. Сами законы по-прежнему вычисляют производные,",
             "но неиспользуемые матрицы из них больше не строятся.", "",
             "## Полные траектории", "",
             "| Режим | мс | Шаг, мкс | max ΔV, В | max ΔI ветвей, А | Отн. ошибка выхода | Деления | ПК full/compact |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        lines.append(f"| {row['name']} | {row['duration_s']*1e3:g} | {row['h_s']*1e6:g} | "
                     f"{row['max_node_difference_v']:.3g} | {row['max_branch_difference_a']:.3g} | "
                     f"{row['relative_output_error']:.3g} | {row['shared_subdivisions']} | {row['host_speed_ratio']:.3f} |")
    lines += ["", "## Проверки сборки и профиль", "",
              "До траекторий проверяются DC, токи, заряды и обе матрицы производных.",
              "На перегрузе сборка дополнительно сверяется на сохранённых состояниях.",
              "Допуски совпадения траекторий прежние: 100 мкВ, 1 мкА, 1e-6 относительной ошибки выхода.",
              "При отказе любого решателя оба повторяют шаг на общей сгущённой сетке.", ""]
    prior = [row.get("previous_schur", {}) for row in result["cases"]]
    if prior and all(row.get("status") == "compared" for row in prior):
        equal = all(row["bitwise_equal"] for row in prior)
        lines += [f"Сравнение с сохранёнными траекториями предыдущего Schur-решателя: "
                  f"{'побитовое совпадение всех состояний' if equal else 'совпадение в прежних допусках'}.",
                  "Перед сравнением проверено побитовое совпадение полной эталонной траектории и сетки.", ""]
    if "profile" in result:
        lines += ["Пять повторов по 200 шагов после прогрева; медиана времени без профилировщика:", "",
                  "| Реализация | Время, с | Ускорение к полной MNA |", "|:---|---:|---:|"]
        base = result["profile"]["full"]["median_unprofiled_s"]
        for name, row in result["profile"].items():
            elapsed = row["median_unprofiled_s"]
            lines.append(f"| {name} | {elapsed:.6f} | {base/elapsed:.3f} |")
        lines += ["", "В metrics.json отдельно сохранены времена функций cProfile. Вложенные",
                  "накопленные времена не складываются; они включают вызываемые функции."]
    if "error" in result:
        lines += ["", "## Незавершённый прогон", "", result["error"]]
    lines += ["", "## Границы результата", "",
              "Это проверка вычислительной реализации существующей инженерной макромодели.",
              "Положение Gain/Master=0,5 и прежние две атаки не заменяют длинную DI и сетку ручек.",
              "Все времена — Python/ПК, не такты STM32N6. Интегратор и частота не менялись.", "",
              "Запуск: `.venv/bin/python simulation/run_compact_assembly.py`.",
              "`--quick` — отдельный короткий прогон, `--report-only` — только построение отчёта.",
              "Полные массивы: `simulation/raw/compact_assembly/`, вне Git.", ""]
    (DEST / "report.md").write_text("\n".join(lines), encoding="utf-8")
    attacks = [r for r in result["cases"] if r["name"].startswith("attack")]
    if not attacks:
        return
    fig, axes = plt.subplots(len(attacks), 2, figsize=(11, 3*len(attacks)), squeeze=False)
    out = result["output_index"]
    for ax, row in zip(axes, attacks):
        data = np.load(RAW / f"{row['name']}.npz")
        t = (data["time"]-data["time"][0])*1e3
        ax[0].plot(t, data["full"][:, out], label="Full MNA")
        ax[0].plot(t, data["reduced"][:, out], "--", label="Compact")
        ax[0].set(title=row["name"], xlabel="Time from attack, ms", ylabel="Output, V")
        ax[0].legend()
        ax[1].plot(t, (data["reduced"][:, out]-data["full"][:, out])*1e6)
        ax[1].set(xlabel="Time from attack, ms", ylabel="Difference, µV")
        for a in ax:
            a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(DEST / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    global RAW, DEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--run-name", help="Separate named rerun; preserve previous large runs")
    args = parser.parse_args()
    if args.run_name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
            parser.error("run-name must contain only letters, numbers, underscores or hyphens")
        RAW, DEST = RAW / args.run_name, DEST / args.run_name
    if args.quick:
        RAW, DEST = RAW / "quick", DEST / "quick"
    RAW.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    # Reuse the exact paired stepping / bisection / error checks of the previous
    # experiment, directing only its raw output to this independent experiment.
    comparison.RAW = RAW
    output = DEST / "metrics.json"
    if args.report_only:
        render(json.loads(output.read_text(encoding="utf-8")))
        return
    if output.exists() and not args.quick:
        raise FileExistsError("Large-run results already exist; use --run-name to preserve them")
    began_ns = time.time_ns()
    result = dict(status="running", quick=args.quick, criteria=LIMITS, cases=[],
                  solver_names=dict(full="Circuit", reduced="CompactCircuit"),
                  source_sha256={name: hashlib.sha256((ROOT / "simulation" / name).read_bytes()).hexdigest()
                                 for name in ("full_mna.py", "reduced_mna.py", "compact_mna.py", "tube_models.py",
                                              "run_linear_reduction.py", "run_compact_assembly.py")})
    def save():
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        circuits = pair()
        full, compact = circuits
        fingerprint = hashlib.sha256(full.fingerprint().encode()).hexdigest()
        source = ROOT / "simulation/raw/linear_reduction/initial.npz"
        with np.load(source) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("Common initial state fingerprint does not match the current circuit")
            initial, start = saved["state"], float(saved["time"])
        if initial.shape != (full.size,) or not np.all(np.isfinite(initial)):
            raise ValueError("Invalid initial state")
        result.update(unknowns=full.size, retained=len(compact.retained), output_index=full.index["out"],
                      circuit_sha256=fingerprint, initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      initial_time_s=start)
        signal = pair(False)
        dc = [c.dc()[0] for c in signal]
        result["signal_dc"] = comparison.errors(signal[0], *dc)
        comparison.assert_errors(result["signal_dc"], relative=False)
        result["assembly"] = assembly_check(compact, [np.zeros(full.size), initial])
        result["profile"] = profile_cost(initial, start)
        save()
        arrays, row = comparison.paired_segment(circuits, [np.zeros(full.size)]*2, 0.,
                                            .002 if args.quick else .06, 100e-6, "electrical_start")
        row["previous_schur"] = compare_previous_schur(full, row, arrays, args.quick)
        result["cases"].append(row)
        save()
        for amplitude, h in ((.1, 1.25e-6), (.5, .625e-6)):
            for c in circuits:
                c.sources["Vin"] = (0., amplitude, 1000., 0.)
            arrays, row = comparison.paired_segment(circuits, [initial]*2, start,
                                                    .0002 if args.quick else .02, h,
                                                    f"attack_{round(amplitude*1000)}mv")
            result["cases"].append(row)
            row["previous_schur"] = compare_previous_schur(full, row, arrays, args.quick)
            result[f"assembly_{round(amplitude*1000)}mv"] = assembly_check(compact, arrays[0][::max(1, len(arrays[0])//20)])
            if amplitude == .1:
                recovery_initial = [a[-1].copy() for a in arrays]
            save()
        for c in circuits:
            c.sources["Vin"] = (0., 0., 1000., 0.)
        arrays, row = comparison.paired_segment(circuits, recovery_initial, start+(.0002 if args.quick else .02),
                                            .002 if args.quick else .5, 20e-6, "recovery_100mv")
        row["previous_schur"] = compare_previous_schur(full, row, arrays, args.quick)
        result["cases"].append(row)
        result["status"] = "pass"
        save()
        render(result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        known = {row["name"] for row in result["cases"]}
        for saved in sorted(RAW.glob("*.metrics.json")):
            if saved.stat().st_mtime_ns < began_ns:
                continue
            row = json.loads(saved.read_text(encoding="utf-8"))
            if row["name"] not in known:
                result["cases"].append(row)
        save()
        render(result)
        raise
    print("Report:", DEST / "report.md", flush=True)


if __name__ == "__main__":
    main()
