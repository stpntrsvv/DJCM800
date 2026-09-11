"""Qualify mechanical 0..10 controls with explicit LOG Gain/Master mapping."""
from __future__ import annotations
import argparse
import hashlib
import json
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, Circuit, AUDIO_TAPER_MIDPOINT, build, control_position
import run_linear_reduction as comparison
from run_controls_qualification import BASELINE, MODEL, ranges

SAMPLE=ROOT/"simulation/samples/e_major_chord/e_major_attack.wav"
DEST=ROOT/"simulation/experiments/log_controls_qualification"
RAW=ROOT/"simulation/raw/log_controls_qualification"
MID=dict(GAIN=5.,BASS=5.,MID=5.,TREBLE=5.,MASTER=5.,PRESENCE=5.)
CASES=(
    ("all5_25mv",MID,.025),
    ("all5_100mv",MID,.100),
    ("gain2_master8",MID|{"GAIN":2.,"MASTER":8.},.100),
    ("gain8_master2",MID|{"GAIN":8.,"MASTER":2.},.100),
    ("gain8_master5",MID|{"GAIN":8.,"MASTER":5.},.100),
)


def read_signal():
    with wave.open(str(SAMPLE),"rb") as stream:
        rate=stream.getframerate(); data=np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2").astype(float)
    data-=np.mean(data); data/=np.max(np.abs(data)); return data,rate


def pair(positions):
    options=dict(full_supply=True,amplitude=0.,controls=BASELINE,control_positions=positions,**MODEL)
    return build(circuit_type=Circuit,**options),build(circuit_type=CompactCircuit,**options)


def render(result):
    lines=["# Квалификация LOG-ручек JCM800","",f"Статус: **{result['status']}**.","",
           "Gain и Master задаются механическими положениями 0–10. Их электрическая доля",
           f"вычисляется гладкой степенной LOG-кривой с долей {result['audio_taper_midpoint']:g} в положении 5.",
           "Остальные ручки пока линейны. Точная характеристика реальных деталей не измерена.","",
           "| Режим | Вход, мВ | Gain | Master | Gain электр. | Master электр. | Выход peak, В | Pavg 16Ω, Вт | max ΔV, В | full/compact |",
           "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result.get("cases",[]):
        q=r["ranges"]["out"]; peak=max(abs(q["min_v"]),abs(q["max_v"])); p=r["output_mean_power_w"]
        lines.append(f"| {r['name']} | {r['amplitude_v']*1000:g} | {r['positions']['GAIN']:g} | {r['positions']['MASTER']:g} | "
                     f"{r['electrical']['GAIN']:.4f} | {r['electrical']['MASTER']:.4f} | {peak:.3f} | {p:.3f} | "
                     f"{r['max_node_difference_v']:.3g} | {r['host_speed_ratio']:.3f} |")
    lines += ["", f"Каждое положение получает синхронный {result['duration_s']*1e3:g}-мс предварительный период без входа,",
              f"затем первые {result['duration_s']*1e3:g} мс DI. Полная и компактная MNA используют общую сетку; времена",
              "относятся к Python/ПК. Результат устанавливает рабочую координату ручек, но не",
              "превращает номинальную 10%-кривую в измерение заводского потенциометра.",""]
    if "error" in result: lines += ["## Ошибка","",result["error"],""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    global DEST,RAW
    p=argparse.ArgumentParser();p.add_argument("--quick",action="store_true");p.add_argument("--run-name");a=p.parse_args()
    if a.run_name:
        if not all(ch.isalnum() or ch in "_-" for ch in a.run_name): p.error("invalid run-name")
        DEST,RAW=DEST/a.run_name,RAW/a.run_name
    if a.quick: DEST,RAW=DEST/"quick",RAW/"quick"
    DEST.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True);comparison.RAW=RAW
    output=DEST/"metrics.json"
    if output.exists() and not a.quick: raise FileExistsError("use --run-name")
    signal,rate=read_signal();positions_axis=np.arange(len(signal));duration=.0002 if a.quick else .020;pre=duration
    result=dict(status="running",quick=a.quick,audio_taper_midpoint=AUDIO_TAPER_MIDPOINT,duration_s=duration,cases=[])
    def save(): output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as z: initial,start=z["state"],float(z["time"])
        for name,positions,amplitude in CASES:
            circuits=pair(positions)
            warm,_=comparison.paired_segment(circuits,[initial.copy(),initial.copy()],start,pre,20e-6,name+"_preroll",relative=False)
            t0=start+pre
            waveform=lambda t,origin=t0,peak=amplitude: peak*np.interp((t-origin)*rate,positions_axis,signal,left=0.,right=0.)
            for c in circuits:c.source_function("Vin",waveform)
            arrays,row=comparison.paired_segment(circuits,[warm[0][-1],warm[1][-1]],t0,duration,
                                                  20e-6 if a.quick else 1.25e-6,name)
            out=circuits[0].index["out"]
            row.update(positions=positions,amplitude_v=amplitude,
                       electrical={k:control_position(k,v) for k,v in positions.items()},
                       output_mean_power_w=float(np.mean(arrays[0][:,out]**2/16.)),ranges=ranges(circuits[0],arrays[0]))
            result["cases"].append(row);save()
        result["status"]="pass";save();render(result)
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}");save();render(result);raise
    print("Report:",DEST/"report.md")


if __name__=="__main__":main()
