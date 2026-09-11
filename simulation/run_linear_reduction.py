"""Reproduce exact linear elimination against the unchanged full MNA solver.

Both trajectories retain every physical state and share accepted time steps.
Raw arrays stay outside Git; reports include complete-state error and provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from full_mna import ROOT, Circuit, build
from reduced_mna import ReducedCircuit
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "simulation/raw/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_tube_closure import refine

DEST = ROOT / "simulation/experiments/linear_reduction"
RAW = ROOT / "simulation/raw/linear_reduction"
OPTIONS = dict(controls=.5, tube_set="detailed:RSD-1", tube_caps="datasheet", el34_grid_r=2001.)
# Acceptance limits set before comparing independently accumulated trajectories.
LIMITS = dict(max_node_difference_v=1e-4, max_branch_difference_a=1e-6,
              relative_output_error=1e-6, delta_scaled_error=1e-8)


def pair(full=True, amplitude=0.):
    return (build(full, amplitude=amplitude, **OPTIONS),
            build(full, amplitude=amplitude, circuit_type=ReducedCircuit, **OPTIONS))


def errors(c, a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    diff = b-a
    out = c.index["out"]
    norm = float(np.linalg.norm(a[:, out]))
    return dict(max_node_difference_v=float(np.max(np.abs(diff[:, :len(c.nodes)]))),
                max_branch_difference_a=float(np.max(np.abs(diff[:, len(c.nodes):]))),
                relative_output_error=float(np.linalg.norm(diff[:, out])/norm) if norm > 1e-12 else 0.,
                output_rms_difference_v=float(np.sqrt(np.mean(diff[:, out]**2))))


def assert_errors(row, relative=True):
    for key in ("max_node_difference_v", "max_branch_difference_a", "relative_output_error"):
        if key == "relative_output_error" and not relative:
            continue
        if row[key] > LIMITS[key]:
            raise AssertionError(f"{key}: {row[key]:g} exceeds {LIMITS[key]:g}")


def paired_segment(circuits, initial, t0, duration, h, name):
    count = round(duration/h)
    if abs(count*h-duration) > 1e-12:
        raise ValueError("Duration must be a multiple of h")
    arrays = [np.empty((count+1, c.size)) for c in circuits]
    for data, x in zip(arrays, initial):
        data[0] = x
    stats = [dict(host_s=0., kcl_a=0., voltage_residual_v=0., max_iterations=0,
                  attempted_steps=0, failures=0) for _ in circuits]
    subdivisions = 0

    def advance(states, t, step, depth=0):
        nonlocal subdivisions
        outputs, failed = [], False
        for c, x, s in zip(circuits, states, stats):
            start = time.perf_counter()
            try:
                y, info = c.step(x, t+step, step)
                outputs.append(y)
                s["kcl_a"] = max(s["kcl_a"], info["kcl"])
                s["voltage_residual_v"] = max(s["voltage_residual_v"], info["voltage"])
                s["max_iterations"] = max(s["max_iterations"], info["iterations"])
            except RuntimeError:
                s["failures"] += 1
                failed = True
            finally:
                s["host_s"] += time.perf_counter()-start
                s["attempted_steps"] += 1
        if failed:
            if depth >= 12:
                raise RuntimeError(f"Both solvers cannot share an accepted step in {name}")
            subdivisions += 1
            middle = advance(states, t, step/2, depth+1)
            return advance(middle, t+step/2, step/2, depth+1)
        return outputs

    for j in range(count):
        outputs = advance([a[j] for a in arrays], t0+j*h, h)
        for a, x in zip(arrays, outputs):
            a[j+1] = x
        if (j+1) % 2000 == 0:
            print(f"{name}: {j+1}/{count}", flush=True)
    metrics = dict(name=name, duration_s=duration, h_s=h, shared_subdivisions=subdivisions,
                   full=stats[0], reduced=stats[1], **errors(circuits[0], *arrays))
    metrics["host_speed_ratio"] = stats[0]["host_s"]/stats[1]["host_s"]
    metrics["cached_factorizations"] = circuits[1].factorizations
    np.savez_compressed(RAW / f"{name}.npz", time=t0+np.arange(count+1)*h,
                        full=arrays[0], reduced=arrays[1])
    # Preserve a failed completed trajectory and its diagnostics before raising.
    (RAW / f"{name}.metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    assert_errors(metrics)
    print(name, json.dumps(metrics), flush=True)
    return arrays, metrics


def algebra_check(circuits, states):
    full, reduced = circuits
    rng = np.random.default_rng(800)
    worst, backward = 0., 0.
    count = 0
    for x in states:
        _, jac, _, cap = full.nonlinear(x)
        # A periodic AC state is not a DC operating point. Removing all storage
        # there produces an ill-conditioned artificial problem (see diagnostics).
        alphas = (0., 1e3, 1e4, 2e5, 8e5, 1.6e6) if not np.any(x) else (1e3, 1e4, 2e5, 8e5, 1.6e6)
        for alpha in alphas:
            A = full.G+alpha*full.C
            nl = jac+alpha*cap
            # Mixed KCL and branch equations, including arbitrary RHS, not only
            # the residual of an already converged point.
            residual = rng.normal(size=full.size)
            residual[:len(full.nodes)] *= .001
            a = full.linearized_delta(A, nl, residual, alpha)
            b = reduced.linearized_delta(A, nl, residual, alpha)
            worst = max(worst, float(np.max(np.abs(a-b)/np.maximum(1., np.abs(a)))))
            J = A+nl
            denominator = np.abs(J)@np.abs(b)+np.abs(residual)+1e-30
            backward = max(backward, float(np.max(np.abs(J@b+residual)/denominator)))
            count += 1
    if worst > LIMITS["delta_scaled_error"] or backward > 1e-12:
        raise AssertionError((worst, backward))
    return dict(cases=count, delta_scaled_error=worst, componentwise_backward_error=backward)


def render(result):
    lines = ["# Точное исключение линейных неизвестных JCM800", "",
             "Физика: Dempwolf RSD-1 + Reefman, RGI+Rs=2001 Ом, паспортные ёмкости.",
             f"Полная система: {result['unknowns']} неизвестных. Решение Ньютона: "
             f"{result['retained']} неизвестных; {result['eliminated']} восстанавливаются линейно.",
             "Все состояния, заряды, питание и обратная связь сохранены. Неявный Эйлер,",
             "Ньютон, поиск шага и допуски полной невязки одинаковы. При отказе любого",
             "решателя оба повторяют шаг на одной сгущённой сетке.", "",
             "## Сравнение полных траекторий", "",
             "| Режим | Длительность, мс | Шаг, мкс | max ΔV, В | max ΔI ветвей, А | Отн. ошибка выхода | Деления | ПК full/reduced |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        lines.append(f"| {row['name']} | {row['duration_s']*1e3:g} | {row['h_s']*1e6:g} | "
                     f"{row['max_node_difference_v']:.3g} | {row['max_branch_difference_a']:.3g} | "
                     f"{row['relative_output_error']:.3g} | {row['shared_subdivisions']} | {row['host_speed_ratio']:.3f} |")
    lines += ["", "## Дополнительные проверки", "",
              f"DC сигнальной схемы: max ΔV={result['signal_dc']['max_node_difference_v']:.3g} В.",
              f"Линейные поправки на {result['algebra']['cases']} сочетаниях состояния/шага: "
              f"масштабированная разность {result['algebra']['delta_scaled_error']:.3g}; "
              f"покомпонентная обратная ошибка {result['algebra']['componentwise_backward_error']:.3g}.",
              "Смена шага и возврат к сохранённому разложению проверяются в этой же матрице.", "",
              "## Границы вывода", "",
              "Это алгебраическое сокращение размера решения Ньютона, а не уменьшение числа",
              "физических состояний и не новая аттестация ламп. Нелинейные токи и якобиан",
              "пока собираются в полном пространстве; полная невязка также сохраняется.",
              "Времена относятся к Python/ПК и одному парному прогону, не к тактам STM32N6.",
              "Тесты сильного сигнала используют прежнюю пару уровней и положение Gain/Master=0,5;",
              "длинная DI, другие положения ручек и выбор интегратора остаются следующими опытами.", "",
              "Сырые массивы: `simulation/raw/linear_reduction/`. Метрики и хеши — `metrics.json`.",
              "Запуск: `.venv/bin/python simulation/run_linear_reduction.py`.", ""]
    (DEST / "report.md").write_text("\n".join(lines), encoding="utf-8")
    attacks = [r for r in result["cases"] if r["name"].startswith("attack")]
    fig, axes = plt.subplots(len(attacks), 2, figsize=(11, 3*len(attacks)), squeeze=False)
    out = result["output_index"]
    for ax, row in zip(axes, attacks):
        data = np.load(RAW / f"{row['name']}.npz")
        t = (data["time"]-data["time"][0])*1e3
        ax[0].plot(t, data["full"][:, out], label="Full MNA")
        ax[0].plot(t, data["reduced"][:, out], "--", label="Schur")
        ax[0].set(title=row["name"], xlabel="Time, ms", ylabel="Output, V")
        ax[0].legend()
        ax[1].plot(t, (data["reduced"][:, out]-data["full"][:, out])*1e6)
        ax[1].set(xlabel="Time, ms", ylabel="Difference, µV")
        for a in ax:
            a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(DEST / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Separate short smoke report; not final qualification")
    parser.add_argument("--report-only", action="store_true", help="Render saved successful metrics without recalculation")
    args = parser.parse_args()
    global DEST, RAW
    if args.quick:
        DEST, RAW = DEST / "quick", RAW / "quick"
    DEST.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    output = DEST / "metrics.json"
    if args.report_only:
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["status"] != "pass":
            raise ValueError("Only a completed successful run can be rendered")
        render(result)
        result["report_renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        return
    result = dict(status="running", quick=args.quick, criteria=LIMITS, cases=[])

    def save():
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")

    save()
    try:
        signal = pair(False)
        dc = [c.dc()[0] for c in signal]
        result["signal_dc"] = errors(signal[0], *dc)
        assert_errors(result["signal_dc"], relative=False)
        circuits = pair()
        c, reduced = circuits
        result.update(unknowns=c.size, retained=len(reduced.retained), eliminated=len(reduced.eliminated),
                      output_index=c.index["out"], retained_nodes=[c.nodes[i] for i in reduced.retained],
                      circuit_sha256=hashlib.sha256(c.fingerprint().encode()).hexdigest(),
                      source_sha256={name: hashlib.sha256((ROOT / "simulation" / name).read_bytes()).hexdigest()
                                     for name in ("full_mna.py", "reduced_mna.py", "tube_models.py", "run_linear_reduction.py")})
        # Existing accepted state is only a warm start: the corrected circuit
        # re-establishes periodicity before supplying the common comparison state.
        source = ROOT / "simulation/raw/settling_power_detailed_rsd1/settled.npz"
        with np.load(source) as seed:
            x, t = seed["state"], float(seed["time"])
        if x.shape != (c.size,) or not np.all(np.isfinite(x)):
            raise ValueError("Warm-start state is incompatible")
        result["warm_start_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        initial, start, history = refine(c, x, t)
        result["periodic_refinement"] = history
        np.savez_compressed(RAW / "initial.npz", state=initial, time=start,
                            fingerprint=result["circuit_sha256"])
        result["algebra"] = algebra_check(circuits, [np.zeros(c.size), initial])
        zero = [np.zeros(c.size), np.zeros(c.size)]
        _, row = paired_segment(circuits, zero, 0., .002 if args.quick else .06, 100e-6, "electrical_start")
        result["cases"].append(row)
        save()
        for amplitude, h in ((.1, 1.25e-6), (.5, .625e-6)):
            for a in circuits:
                a.sources["Vin"] = (0., amplitude, 1000., 0.)
            arrays, row = paired_segment(circuits, [initial, initial], start,
                                         .0002 if args.quick else .02, h,
                                         f"attack_{round(amplitude*1000)}mv")
            result["cases"].append(row)
            save()
            # Include an actual large-signal state in the algebraic audit.
            if amplitude == .5:
                result["overload_algebra"] = algebra_check(circuits, [arrays[0][len(arrays[0])//2]])
            if amplitude == .1:
                recovery_initial = [a[-1].copy() for a in arrays]
        for a in circuits:
            a.sources["Vin"] = (0., 0., 1000., 0.)
        _, row = paired_segment(circuits, recovery_initial, start+(.0002 if args.quick else .02),
                                .002 if args.quick else .5, 20e-6, "recovery_100mv")
        result["cases"].append(row)
        result["status"] = "pass"
        save()
        render(result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    print("Report:", DEST / "report.md", flush=True)


if __name__ == "__main__":
    main()
