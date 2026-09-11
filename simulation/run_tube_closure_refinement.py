"""Адаптивное продолжение сгущения 2.5 -> 1.25 -> .625 мкс для формы сигнала."""
from __future__ import annotations
import json
import numpy as np
from full_mna import ROOT, build
from run_settling import segment
from run_tube_closure import refine

RAW = ROOT / "simulation/raw/tube_closure"
DEST = ROOT / "simulation/experiments/tube_closure"


def difference(fine, coarse):
    tf, tc = np.linspace(0., .02, len(fine)), np.linspace(0., .02, len(coarse))
    delta = fine-np.interp(tf, tc, coarse)
    return dict(relative_waveform=float(np.linalg.norm(delta)/np.linalg.norm(fine)),
                max_abs_v=float(np.max(np.abs(delta))))


def main():
    saved = np.load(ROOT / "simulation/raw/settling_power_detailed_rsd1/settled.npz")
    initial, initial_time = saved["state"], float(saved["time"])
    results = {"status": "running", "criterion": "relative waveform change <= 1%", "cases": {}}
    output = DEST / "refinement_metrics.json"
    for variant, caps in (("author", "datasheet"), ("legacy_caps", "legacy")):
        c = build(True, amplitude=0., controls=.5, tube_set="detailed:RSD-1",
                  el34_grid_r=2001., tube_caps=caps)
        state, _, history = refine(c, initial, initial_time)
        old = np.load(RAW / variant / "waveforms.npz")
        results["cases"][variant] = {"periodic_refinement": history, "amplitudes": {}}
        for mv in (100, 500):
            previous = old[f"{mv}mv_2.5us"]
            rows = []
            for h in (1.25e-6, .625e-6):
                c.sources["Vin"] = (0., mv/1000, 1000., 0.)
                states, stats = segment(c, state, 0., .02, h)
                waveform = states[:, c.index["out"]]
                row = dict(h_s=h, halvings=stats["halvings"], max_iterations=stats["iterations"],
                           **difference(waveform, previous))
                rows.append(row)
                folder = RAW / "refinement"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"{variant}_{mv}mv_{h*1e6:g}us.npz", output=waveform)
                print(variant, mv, h, row, flush=True)
                previous = waveform
                if row["relative_waveform"] <= .01:
                    break
            results["cases"][variant]["amplitudes"][str(mv)] = rows
            output.write_text(json.dumps(results, indent=2)+"\n")
    results["status"] = "complete"
    output.write_text(json.dumps(results, indent=2)+"\n")
    print("Result:", output)


if __name__ == "__main__":
    main()
