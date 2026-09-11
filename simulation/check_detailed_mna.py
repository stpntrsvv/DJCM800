"""Короткая проверка подключения подробных ламп к общей MNA до долгого установления."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from full_mna import ROOT, build, electrode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triode", choices=("RSD-1", "RSD-2", "EHX-1"), default="RSD-1")
    args = parser.parse_args()
    tube_set = f"detailed:{args.triode}"
    circuit = build(False, amplitude=0., controls=.5, tube_set=tube_set)
    state, reports = circuit.dc()
    current, _, charge, _ = circuit.nonlinear(state)
    residual = circuit.G@state+current-circuit.rhs()
    rows = {}
    for part in circuit.parts:
        if part[0] != "T":
            continue
        kind, name, model, p, g, k, screen = part
        vk = 0. if k == "0" else state[circuit.index[k]]
        voltages = [state[circuit.index[p]]-vk, state[circuit.index[g]]-vk]
        if screen is not None:
            voltages.append(state[circuit.index[screen]]-vk)
        currents, _ = electrode(model, voltages)
        rows[name] = dict(model=model, voltages_v=voltages, currents_a=currents.tolist())
    result = dict(status="pass", tube_set=tube_set, nodes=len(circuit.nodes), unknowns=circuit.size,
                  dc_continuation_steps=len(reports), max_kcl_a=float(np.max(np.abs(residual[:len(circuit.nodes)]))),
                  max_voltage_residual_v=float(np.max(np.abs(residual[len(circuit.nodes):]))), tubes=rows)
    destination = ROOT / "simulation/raw/jcm800_pipeline"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "detailed_mna_smoke.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "tube_set", "max_kcl_a", "max_voltage_residual_v")}, indent=2))


if __name__ == "__main__":
    main()
