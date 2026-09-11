"""Qualify Full MNA and CompactCircuit across independent front-panel controls.

This experiment deliberately keeps the accepted tube laws and backward Euler.
It compares complete 98-variable trajectories on a shared time grid; audio DI is
reserved for a later experiment with explicit source provenance and scaling.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, Circuit, build
import run_linear_reduction as comparison

DEST = ROOT / "simulation/experiments/controls_qualification"
RAW = ROOT / "simulation/raw/controls_qualification"
MODEL = dict(tube_set="detailed:RSD-1", tube_caps="datasheet", el34_grid_r=2001.)
BASELINE = dict(GAIN=.5, BASS=.5, MID=.5, TREBLE=.5, MASTER=.5, PRESENCE=.5, NFB=1.)
CASES = (
    ("baseline_100mv", {}, .100),
    ("baseline_500mv", {}, .500),
    ("gain20_master80", {"GAIN": .2, "MASTER": .8}, .100),
    ("gain80_master20", {"GAIN": .8, "MASTER": .2}, .100),
    ("dark", {"BASS": .8, "MID": .7, "TREBLE": .1}, .100),
    ("bright", {"BASS": .2, "MID": .3, "TREBLE": .9}, .100),
    ("presence90", {"PRESENCE": .9}, .100),
    ("nfb_half", {"NFB": .5}, .100),
)


def circuits(controls, amplitude):
    options = dict(full_supply=True, amplitude=amplitude, controls=controls, **MODEL)
    return build(circuit_type=Circuit, **options), build(circuit_type=CompactCircuit, **options)


def ranges(circuit, states):
    names = ("in", "master_out", "pi_p1", "pi_p2", "g4", "g6", "bias", "bplus", "out")
    return {name: {"min_v": float(np.min(states[:, circuit.index[name]])),
                   "max_v": float(np.max(states[:, circuit.index[name]]))} for name in names}


def render(result):
    lines = ["# Квалификация сигналов и ручек JCM800", "",
             f"Статус: **{result['status']}**.", "",
             "Сравниваются полная MNA и CompactCircuit при неизменной физике Dempwolf RSD-1 + Reefman,",
             "RGI+Rs=2001 Ом, паспортных ёмкостях и неявном Эйлере. Для каждой конфигурации",
             "оба решателя получают одинаковое исходное состояние, вход и общую сетку; при отказе",
             "любого решателя шаг синхронно делится.", "", "## Матрица", "",
             "| Режим | Вход, мВ peak | Ручки | max ΔV, В | max ΔI, А | Ошибка выхода | Деления | full/compact |",
             "|:---|---:|:---|---:|---:|---:|---:|---:|"]
    for row in result.get("cases", []):
        changed = ", ".join(f"{k}={v:g}" for k, v in row["changed_controls"].items()) or "все 0,5; NFB=1"
        lines.append(f"| {row['name']} | {row['amplitude_v']*1000:g} | {changed} | "
                     f"{row['max_node_difference_v']:.3g} | {row['max_branch_difference_a']:.3g} | "
                     f"{row['relative_output_error']:.3g} | {row['shared_subdivisions']} | {row['host_speed_ratio']:.3f} |")
    lines += ["", "Диапазоны контрольных узлов сохранены в `metrics.json`. Полные траектории находятся",
              "в игнорируемом `simulation/raw/controls_qualification/`.", "",
              "## Границы", "",
              "Это первая вычислительная квалификация независимых ручек на синусе 1 кГц, а не",
              "физическая аттестация всего диапазона усилителя. Изменённые положения получают один",
              "предварительный период без входа; длительное переустановление питания проверяется отдельно.",
              "DI-гитара и аккорд Big Muff в этот опыт намеренно не копируются: их происхождение, хеш,",
              "масштаб в вольтах и пересчёт временной сетки будут зафиксированы в следующем опыте.", "",
              "Запуск: `.venv/bin/python simulation/run_controls_qualification.py --quick` для проверки",
              "и без `--quick` для полной матрицы. Существующий полный результат не перезаписывается;",
              "для повтора используется `--run-name ИМЯ`.", ""]
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
    result = {"status": "running", "quick": args.quick, "criteria": comparison.LIMITS,
              "baseline_controls": BASELINE, "cases": []}
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        source = ROOT / "simulation/raw/linear_reduction/initial.npz"
        with np.load(source) as data:
            initial, start = data["state"], float(data["time"])
        expected = build(True, amplitude=0., controls=BASELINE, circuit_type=Circuit, **MODEL)
        fingerprint = hashlib.sha256(expected.fingerprint().encode()).hexdigest()
        with np.load(source) as data:
            if str(data["fingerprint"]) != fingerprint:
                raise ValueError("Baseline initial state does not match the accepted model")
        result.update(unknowns=expected.size, retained=39, initial_time_s=start,
                      initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        duration = .0002 if args.quick else .020
        pre_roll = .0002 if args.quick else .020
        for name, changes, amplitude in CASES:
            controls = BASELINE | changes
            pair = circuits(controls, 0.)
            warm, _ = comparison.paired_segment(pair, [initial.copy(), initial.copy()], start,
                                                  pre_roll, 20e-6, name+"_preroll", relative=False)
            for circuit in pair:
                circuit.sources["Vin"] = (0., amplitude, 1000., 0.)
            arrays, row = comparison.paired_segment(pair, [warm[0][-1], warm[1][-1]], start+pre_roll,
                                                     duration, 20e-6 if args.quick else
                                                     (1.25e-6 if amplitude <= .1 else .625e-6), name)
            row.update(amplitude_v=amplitude, controls=controls, changed_controls=changes,
                       ranges=ranges(pair[0], arrays[0]))
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
