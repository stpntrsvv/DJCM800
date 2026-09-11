"""Большой опыт закрытия блока ламп: Ig1/caps sensitivity, grid refinement, recovery."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from full_mna import ROOT, build
from run_settling import PERIOD, describe, segment

RAW = ROOT / "simulation/raw/tube_closure"
DEST = ROOT / "simulation/experiments/tube_closure"
VARIANTS = {
    # Author Reefman EL34 macro: RGI=2k and diode Rs=1 ohm; our diode has no
    # separate Rs, hence the exact series total 2001 ohm. Philips 1969/1970 caps.
    "author": dict(el34_grid_r=2001., tube_caps="datasheet"),
    # Sensitivity bounds, not claims about measured EL34 grid-current curves.
    "ig1_r1k": dict(el34_grid_r=1001., tube_caps="datasheet"),
    "ig1_r4k": dict(el34_grid_r=4001., tube_caps="datasheet"),
    "legacy_caps": dict(el34_grid_r=2001., tube_caps="legacy"),
}


def refine(c, state, start, quick=False):
    previous = None
    history = []
    limit = 3 if quick else 25
    for cycle in range(limit):
        data, stats = segment(c, state, start+cycle*PERIOD, PERIOD, 100e-6)
        state = data[-1]
        dv = di = None
        if previous is not None:
            diff = np.abs(data-previous)
            dv = float(diff[:, :len(c.nodes)].max())
            di = float(diff[:, len(c.nodes):].max())
        history.append(dict(cycle=cycle+1, max_dv_v=dv, max_di_a=di,
                            kcl_a=stats["kcl"], halvings=stats["halvings"]))
        if dv is not None and dv <= .01 and di <= 20e-6:
            return state, start+(cycle+1)*PERIOD, history
        previous = data
    raise RuntimeError("Variant did not re-establish periodic state")


def grid_current(c, states, resistance):
    result = {}
    for number in range(4, 8):
        grid, junction = f"g{number}", f"xv{number}_junction"
        current = (states[:, c.index[grid]]-states[:, c.index[junction]])/resistance
        result[f"Xv{number}"] = dict(mean_a=float(current.mean()), rms_a=float(np.sqrt(np.mean(current**2))),
                                      max_a=float(current.max()), conducting_fraction=float(np.mean(current > 1e-6)))
    return result


def run_case(name, options, initial, initial_time, quick=False):
    c = build(True, amplitude=0., controls=.5, tube_set="detailed:RSD-1", **options)
    state, phase_time, refinement = refine(c, initial, initial_time, quick)
    amplitudes = (.1,) if quick else (.1, .5)
    steps = (5e-6,) if quick else (5e-6, 2.5e-6)
    cases, waveforms = [], {}
    for amplitude in amplitudes:
        for h in steps:
            c.sources["Vin"] = (0., amplitude, 1000., 0.)
            duration = .002 if quick else .02
            states, stats = segment(c, state, 0., duration, h, audit=True)
            info = describe(c, states, stats, duration)
            info.update(amplitude_v=amplitude, h_s=h,
                        el34_grid_current=grid_current(c, states[1:], options["el34_grid_r"]))
            key = f"{round(amplitude*1000)}mv_{h*1e6:g}us"
            waveforms[key] = states[:, c.index["out"]]
            cases.append(info)
            print(name, key, "Pload", info["load_power_w"], "halvings", info["halvings"], flush=True)
    recovery = None
    if name == "author" and not quick:
        c.sources["Vin"] = (0., .1, 1000., 0.)
        attack, _ = segment(c, state, 0., .02, 2.5e-6)
        c.sources["Vin"] = (0., 0., 1000., 0.)
        release, stats = segment(c, attack[-1], .02, .5, 20e-6)
        output = release[:, c.index["out"]]
        recovery = dict(duration_s=.5, h_s=20e-6, last_output_v=float(output[-1]),
                        tail_rms_v=float(np.sqrt(np.mean(output[-1000:]**2))), halvings=stats["halvings"])
        waveforms["recovery_100mv"] = output
    folder = RAW / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "waveforms.npz", **waveforms)
    return dict(options=options, periodic_refinement=refinement, phase_time_s=phase_time,
                cases=cases, recovery=recovery, circuit_fingerprint=hashlib.sha256(c.fingerprint().encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    source = ROOT / "simulation/raw/settling_power_detailed_rsd1/settled.npz"
    if not source.exists():
        raise FileNotFoundError("Run run_jcm800_pipeline.py --triode RSD-1 first")
    saved = np.load(source)
    initial, initial_time = saved["state"], float(saved["time"])
    started = time.monotonic()
    results = dict(status="running", quick=args.quick, variants={})
    output = DEST / ("metrics_quick.json" if args.quick else "metrics.json")
    for name, options in VARIANTS.items():
        print(f"=== {name} ===", flush=True)
        results["variants"][name] = run_case(name, options, initial, initial_time, args.quick)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    # Same-method step refinement, no SPICE claim.
    for variant in results["variants"].values():
        lookup = {(x["amplitude_v"], x["h_s"]): x for x in variant["cases"]}
        if not args.quick:
            variant["step_refinement"] = {}
            for amplitude in (.1, .5):
                coarse, fine = lookup[(amplitude, 5e-6)], lookup[(amplitude, 2.5e-6)]
                variant["step_refinement"][str(amplitude)] = dict(
                    load_power_relative_change=(fine["load_power_w"]-coarse["load_power_w"])/fine["load_power_w"],
                    output_rms_relative_change=(fine["output_rms_v"]-coarse["output_rms_v"])/fine["output_rms_v"])
    results.update(status="complete", elapsed_s=time.monotonic()-started,
                   interpretation="sensitivity study; Ig1 resistance variants are not measured tube fits")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if not args.quick:
        render(results)
    print("Result:", output, flush=True)


def render(results):
    lines = ["# Закрытие блока ламп: Ig1, ёмкости, шаг и восстановление", "",
             "Базовый вариант воспроизводит авторскую ветвь Reefman: RGI=2000 Ом и Rs=1 Ом.",
             "Варианты 1/4 кОм являются анализом чувствительности, а не измеренными моделями EL34.",
             "Ёмкости datasheet: ECC83 1,6/1,6/0,33 пФ; EL34 15,2/1,1/8,4 пФ.", "",
             "| Вариант | Вход, мВ | Шаг, мкс | P нагрузки, Вт | RMS выхода, В | Деления |",
             "|:---|---:|---:|---:|---:|---:|"]
    for name, variant in results["variants"].items():
        for case in variant["cases"]:
            lines.append(f"| {name} | {case['amplitude_v']*1000:g} | {case['h_s']*1e6:g} | {case['load_power_w']:.6f} | {case['output_rms_v']:.6f} | {case['halvings']} |")
    recovery = results["variants"]["author"]["recovery"]
    lines += ["", "## Длинное восстановление", "",
              f"После атаки 100 мВ рассчитано {recovery['duration_s']:.3f} с: последний выход "
              f"{recovery['last_output_v']:.6g} В, RMS последних 20 мс {recovery['tail_rms_v']:.6g} В.", "",
              "## Ограничение вывода", "",
              "Этот опыт измеряет чувствительность результата к выбранной ветви Ig1 и ёмкостям.",
              "Он закрывает ошибки переноса авторского макромоделя и численную проверку шага,",
              "но без измеренного семейства Ig1 не превращает Shockley-ветвь в доказанную физику.", ""]
    (DEST / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
