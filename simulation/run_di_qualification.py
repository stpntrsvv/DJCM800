"""Compare Full MNA and CompactCircuit on a provenance-tracked guitar DI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, Circuit, build
import run_linear_reduction as comparison
from run_controls_qualification import BASELINE, MODEL, ranges

SAMPLE = ROOT / "simulation/samples/e_major_chord/e_major_attack.wav"
EXPECTED_SHA256 = "10877cbd67e41d3e187d46b4463b82e30e1b40c6012e16bead2b0c5bf07ae1f1"
DEST = ROOT / "simulation/experiments/di_qualification"
RAW = ROOT / "simulation/raw/di_qualification"
CASES = (
    ("baseline_25mv", {}, .025),
    ("baseline_100mv", {}, .100),
    ("gain80_master20", {"GAIN": .8, "MASTER": .2}, .100),
    ("bright", {"BASS": .2, "MID": .3, "TREBLE": .9}, .100),
    ("nfb_half", {"NFB": .5}, .100),
)


def read_sample():
    digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"DI SHA-256 mismatch: {digest}")
    with wave.open(str(SAMPLE), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 DI")
        rate, frames = stream.getframerate(), stream.getnframes()
        signal = np.frombuffer(stream.readframes(frames), dtype="<i2").astype(np.float64)
    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    return signal, rate, digest


def pair(controls):
    options = dict(full_supply=True, amplitude=0., controls=controls, **MODEL)
    return build(circuit_type=Circuit, **options), build(circuit_type=CompactCircuit, **options)


def render(result):
    lines = ["# Квалификация JCM800 на сухом гитарном аккорде", "",
             f"Статус: **{result['status']}**.", "",
             "Неизменённая PCM16-атака аккорда из BigMuffDPi подаётся как безразмерная форма,",
             "нормированная к единичному peak. Физическая амплитуда назначается отдельно для",
             "каждого режима. Между исходными отсчётами 48 кГц используется линейная интерполяция;",
             "полная MNA и CompactCircuit вычисляют её в одних и тех же моментах общей сетки.", "",
             f"SHA-256 WAV: `{result['sample']['sha256']}`.", "", "## Результаты", "",
             "| Режим | Вход, мВ peak | Ручки | max ΔV, В | max ΔI, А | Ошибка выхода | Деления | full/compact |",
             "|:---|---:|:---|---:|---:|---:|---:|---:|"]
    for row in result.get("cases", []):
        controls = ", ".join(f"{k}={v:g}" for k, v in row["changed_controls"].items()) or "базовые"
        lines.append(f"| {row['name']} | {row['amplitude_v']*1000:g} | {controls} | "
                     f"{row['max_node_difference_v']:.3g} | {row['max_branch_difference_a']:.3g} | "
                     f"{row['relative_output_error']:.3g} | {row['shared_subdivisions']} | {row['host_speed_ratio']:.3f} |")
    lines += ["", "Диапазоны входа, фазоинвертора, выходных сеток, bias, B+ и выхода сохранены",
              "в `metrics.json`; полные траектории — вне Git в `simulation/raw/di_qualification/`.", "",
              "## Границы", "",
              "В полном режиме рассчитываются первые 20 мс атаки после одного периода без входа.",
              "Это квалификация вычислительной эквивалентности и начального фронта сложного сигнала,",
              "не весь 250-мс файл, не проверка сходимости интегратора и не длинное восстановление.",
              "Параметры ламп и трансформаторов сохраняют ранее описанные физические ограничения.", "",
              "Запуск: `.venv/bin/python simulation/run_di_qualification.py --quick` или без флага",
              "для полной матрицы. Повтор полного результата требует `--run-name ИМЯ`.", ""]
    if "error" in result:
        lines += ["## Ошибка", "", result["error"], ""]
    (DEST / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global DEST, RAW
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if args.run_name:
        if not all(ch.isalnum() or ch in "_-" for ch in args.run_name):
            parser.error("run-name must contain only letters, numbers, underscores or hyphens")
        DEST, RAW = DEST / args.run_name, RAW / args.run_name
    if args.quick:
        DEST, RAW = DEST / "quick", RAW / "quick"
    DEST.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    comparison.RAW = RAW
    output = DEST / "metrics.json"
    if output.exists() and not args.quick:
        raise FileExistsError("Large-run results already exist; use --run-name")
    signal, rate, digest = read_sample()
    result = {"status": "running", "quick": args.quick, "criteria": comparison.LIMITS,
              "sample": {"path": str(SAMPLE.relative_to(ROOT)), "sha256": digest,
                         "sample_rate_hz": rate, "frames": len(signal),
                         "interpolation": "linear", "normalization": "remove mean; unit peak"},
              "cases": []}
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        source = ROOT / "simulation/raw/linear_reduction/initial.npz"
        with np.load(source) as data:
            initial, start = data["state"], float(data["time"])
        result.update(initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), initial_time_s=start)
        pre_roll, duration = ((.0002, .0002) if args.quick else (.020, .020))
        for name, changes, amplitude in CASES:
            circuits = pair(BASELINE | changes)
            warm, _ = comparison.paired_segment(circuits, [initial.copy(), initial.copy()], start,
                                                  pre_roll, 20e-6, name+"_preroll", relative=False)
            signal_start = start+pre_roll
            positions = np.arange(len(signal), dtype=np.float64)
            def waveform(t, peak=amplitude):
                return peak*np.interp((t-signal_start)*rate, positions, signal, left=0., right=0.)
            for circuit in circuits:
                circuit.source_function("Vin", waveform)
            arrays, row = comparison.paired_segment(circuits, [warm[0][-1], warm[1][-1]], signal_start,
                                                     duration, 20e-6 if args.quick else 1.25e-6, name)
            row.update(amplitude_v=amplitude, changed_controls=changes, controls=BASELINE | changes,
                       ranges=ranges(circuits[0], arrays[0]))
            result["cases"].append(row)
            save()
        result["status"] = "pass"
        save()
        render(result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        render(result)
        raise
    print("Report:", DEST / "report.md")


if __name__ == "__main__":
    main()
