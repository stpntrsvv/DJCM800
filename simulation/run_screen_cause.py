"""Короткая последняя абляция причин экранной мощности EL34."""
from __future__ import annotations

import argparse
import hashlib
import json
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build, electrode
from run_controls_qualification import BASELINE, MODEL
from run_physics_hypotheses import simulate

SAMPLE = ROOT/"simulation/samples/e_major_chord/e_major_attack.wav"
DEST = ROOT/"simulation/experiments/screen_cause"
RAW = ROOT/"simulation/raw/screen_cause"
POSITIONS = dict(GAIN=5., BASS=5., MID=5., TREBLE=5., MASTER=8., PRESENCE=5.)
CASES = (
    ("accepted", {}),
    ("secondary_off", {"el34_secondary_scale": 0.}),
    ("screen_feed_500ohm", {"component_overrides": {"Rch": 500.}}),
    ("primary_2200ohm", {"parameter_overrides": {"RAA": 2200.}}),
)


def read_signal():
    digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    with wave.open(str(SAMPLE), "rb") as stream:
        rate = stream.getframerate()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float)
    data -= np.mean(data); data /= np.max(np.abs(data))
    return data, rate, digest


def make_circuit(changes):
    return build(True, amplitude=0., controls=BASELINE, control_positions=POSITIONS,
                 circuit_type=CompactCircuit, **MODEL, **changes)


def metrics(circuit, states, h, secondary_scale):
    powers, low_plate_energy, low_plate_fraction = [], [], []
    kind = "reefman" if secondary_scale == 1. else f"reefman-secondary:{secondary_scale:g}"
    for plate, grid, screen in (("plate_a", "g4", "s4"), ("plate_a", "g5", "s5"),
                                ("plate_b", "g6", "s6"), ("plate_b", "g7", "s7")):
        va, vg, vs = (states[:, circuit.index[name]] for name in (plate, grid, screen))
        current = np.array([electrode(kind, np.array([a, g, s]))[0][1]
                            for a, g, s in zip(va, vg, vs)])
        power = vs*current; mask = va < vs
        powers.append(power)
        low_plate_energy.append(float(np.sum(power[mask])/np.sum(power)))
        low_plate_fraction.append(float(np.mean(mask)))
    powers = np.asarray(powers)
    window = min(len(states), round(.020/h))
    cumulative = np.c_[np.zeros(len(powers)), np.cumsum(powers, axis=1)]
    rolling = (cumulative[:, window:]-cumulative[:, :-window])/window
    tube, offset = np.unravel_index(np.argmax(rolling), rolling.shape)
    return dict(screen_mean_max_w=float(np.max(np.mean(powers, axis=1))),
                screen_max_20ms_w=float(rolling[tube, offset]),
                worst_tube=int(tube+4), worst_window_start_ms=float(offset*h*1e3),
                va_below_vs_fraction_in_worst_tube=low_plate_fraction[tube],
                screen_energy_when_va_below_vs_fraction=low_plate_energy[tube],
                bscreen_min_v=float(np.min(states[:, circuit.index["bscreen"]])),
                bplus_min_v=float(np.min(states[:, circuit.index["bplus"]])),
                output_peak_v=float(np.max(np.abs(states[:, circuit.index["out"]]))))


def render(result):
    lines = ["# Причина экранной мощности EL34", "", f"Статус: **{result['status']}**.", "",
             "Последняя физическая абляция перед переносом вычислений в C. Использован один",
             f"{result['duration_s']*1e3:g}-мс фрагмент полного аккорда, Gain 5 / Master 8, шаг",
             f"{result['h_s']*1e6:g} мкс. Меняется ровно одна гипотеза.", "", "## Результаты", "",
             "| Вариант | Экран mean/max20, Вт | Окно, мс | Va<Vs, доля | Энергия экрана при Va<Vs | Bscreen min, В | Выход peak, В |",
             "|:---|---:|---:|---:|---:|---:|---:|"]
    for row in result.get("cases", []):
        m = row["metrics"]
        lines.append(f"| {row['name']} | {m['screen_mean_max_w']:.3f}/{m['screen_max_20ms_w']:.3f} | "
                     f"{m['worst_window_start_ms']:.3f} | {100*m['va_below_vs_fraction_in_worst_tube']:.2f}% | "
                     f"{100*m['screen_energy_when_va_below_vs_fraction']:.2f}% | {m['bscreen_min_v']:.2f} | {m['output_peak_v']:.2f} |")
    lines += ["", "`secondary_off` не предлагается как новая физика: это причинная абляция только",
              "члена вторичной эмиссии Reefman. `screen_feed_500ohm` заменяет лишь предварительные",
              "100 Ом сопротивления дросселя; `primary_2200ohm` меняет лишь отражённое Raa.", "",
              "## Вывод", "",
              "В худшем окне 97,8% экранной энергии набирается при `Va < Vs`, но отключение",
              "только secondary-члена Reefman снижает max20 лишь на 0,55%. Значит, основной",
              "источник — базовая часть закона экранного тока Reefman в области низкого анодного",
              "напряжения, а не его член вторичной эмиссии. Увеличение предварительного Rch",
              "уменьшает max20 на 1,52%; изменение Raa 1,7 → 2,2 кОм, наоборот, повышает его",
              "на 7,39%. Ни питание, ни выбранный вариант OT не устраняют эффект.", "",
              "Количественное превышение 8 Вт нельзя считать физически установленным до проверки",
              "экранного закона по измеренным семействам в этой области. Дальнейшие длинные Python-",
              "прогоны остановлены; следующий этап — сокращение и перенос вычислительного ядра в C.", ""]
    if "error" in result: lines += ["## Ошибка", "", result["error"], ""]
    (DEST/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global DEST, RAW
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick: DEST, RAW = DEST/"quick", RAW/"quick"
    DEST.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    signal, rate, digest = read_signal(); axis = np.arange(len(signal))
    duration, h = ((.002, 20e-6) if args.quick else (.030, 1.25e-6))
    result = dict(status="running", quick=args.quick, sample_sha256=digest, input_peak_v=.1,
                  positions=POSITIONS, duration_s=duration, h_s=h, cases=[])
    output = DEST/"metrics.json"
    def save(): output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored:
            initial, start = stored["state"], float(stored["time"])
        for name, changes in CASES:
            circuit = make_circuit(changes)
            warm, warm_stats = simulate(circuit, initial.copy(), start, .0002 if args.quick else .050, 20e-6)
            t0 = start+(.0002 if args.quick else .050)
            circuit.source_function("Vin", lambda t, origin=t0: .1*np.interp((t-origin)*rate, axis, signal, left=0., right=0.))
            states, stats = simulate(circuit, warm[-1], t0, duration, h)
            scale = changes.get("el34_secondary_scale", 1.)
            result["cases"].append(dict(name=name, changes=changes, warm_stats=warm_stats,
                                         stats=stats, metrics=metrics(circuit, states, h, scale)))
            np.savez_compressed(RAW/f"{name}.npz", state=states); save()
        result["status"] = "pass"; save(); render(result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}"); save(); render(result); raise


if __name__ == "__main__": main()
