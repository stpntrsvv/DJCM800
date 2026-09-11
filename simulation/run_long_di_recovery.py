"""Run the 250 ms guitar DI plus 500 ms recovery on mechanical LOG controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build, control_position
from run_controls_qualification import BASELINE, MODEL
from run_physics_hypotheses import diagnostics, simulate

SAMPLE=ROOT/"simulation/samples/e_major_chord/e_major_attack.wav"
DEST=ROOT/"simulation/experiments/long_di_recovery"
RAW=ROOT/"simulation/raw/long_di_recovery"
MID=dict(GAIN=5.,BASS=5.,MID=5.,TREBLE=5.,MASTER=5.,PRESENCE=5.)
CASES=(("gain5_master5",MID),("gain8_master2",MID|{"GAIN":8.,"MASTER":2.}),
       ("gain5_master8",MID|{"GAIN":5.,"MASTER":8.}),("gain8_master5",MID|{"GAIN":8.,"MASTER":5.}))


def read_signal():
    digest=hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    with wave.open(str(SAMPLE),"rb") as w:
        rate=w.getframerate(); data=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(float)
    data-=np.mean(data); data/=np.max(np.abs(data)); return data,rate,digest


def circuit(positions):
    return build(True,amplitude=0.,controls=BASELINE,control_positions=positions,circuit_type=CompactCircuit,**MODEL)


def write_wav(path,signal,rate,scale):
    pcm=np.int16(np.clip(signal/scale,-1.,1.)*32767.)
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm.astype("<i2",copy=False).tobytes())


def render(result):
    lines=["# Полный DI и восстановление JCM800","",f"Статус: **{result['status']}**.","",
           f"Сухой аккорд длительностью {result['signal_duration_s']*1e3:g} мс подан с peak 100 мВ. После него рассчитано",
           f"{result['recovery_duration_s']*1e3:g} мс нулевого входа. Все режимы используют механическую шкалу LOG Gain/Master,",
           "CompactCircuit, неявный Эйлер и одинаковые шаги. Холостой эталон рассчитан до",
           "того же абсолютного времени и фазы сети.","",
           "| Режим | Pavg сигнала, Вт | Ppeak, Вт | B+ min, В | Ig1 peak, мА | Ig1 >1мкА | Выход последних 20 мс, мВ RMS | Остаток V RMS, мВ | WAV |",
           "|:---|---:|---:|---:|---:|---:|---:|---:|:---|"]
    for r in result.get("cases",[]):
        d=r["signal_diagnostics"]; e=r["recovery_error"]
        lines.append(f"| {r['name']} | {d['load_mean_w']:.3f} | {d['load_peak_w']:.3f} | {d['bplus_min_v']:.3f} | "
                     f"{d['el34_grid_peak_a']*1e3:.3f} | {100*d['el34_grid_above_1ua_fraction']:.2f}% | "
                     f"{r['recovery_output_last20_rms_v']*1e3:.3f} | {e['node_rms_v']*1e3:.3f} | [{r['name']}.wav]({r['name']}.wav) |")
    lines += ["", "WAV-файлы имеют общую амплитудную шкалу между режимами; это электрический",
              "выход на 16 Ом, нормированный только для прослушивания. Абсолютные вольты и мощности",
              "берутся из metrics.json, не из PCM. Остаток восстановления сравнивает все 98",
              "переменных с синхронным холостым состоянием, а не с несогласованной фазой сети.",""]
    if "error" in result: lines += ["## Ошибка","",result["error"],""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    global DEST,RAW
    p=argparse.ArgumentParser();p.add_argument("--quick",action="store_true");p.add_argument("--run-name");p.add_argument("--report-only",action="store_true");a=p.parse_args()
    if a.run_name:
        if not all(ch.isalnum() or ch in "_-" for ch in a.run_name):p.error("invalid run-name")
        DEST,RAW=DEST/a.run_name,RAW/a.run_name
    if a.quick:DEST,RAW=DEST/"quick",RAW/"quick"
    DEST.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    output=DEST/"metrics.json"
    if a.report_only:
        result=json.loads(output.read_text(encoding="utf-8"))
        for row in result["cases"]:
            c=circuit(row["positions"]); recovery=np.load(RAW/f"{row['name']}.npz")["recovery"]
            count=min(len(recovery),round(.020/result["recovery_h_s"])+1)
            y=recovery[-count:,c.index["out"]]
            row["recovery_output_last20_rms_v"]=float(np.sqrt(np.mean(y*y)))
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        render(result);return
    if output.exists() and not a.quick:raise FileExistsError("use --run-name")
    source,rate,digest=read_signal(); axis=np.arange(len(source)); input_peak=.1
    signal_duration=.002 if a.quick else len(source)/rate
    recovery_duration=.002 if a.quick else .500
    signal_h=20e-6 if a.quick else 1.25e-6;recovery_h=20e-6
    result=dict(status="running",quick=a.quick,sample_sha256=digest,sample_rate_hz=rate,
                input_peak_v=input_peak,signal_duration_s=signal_duration,recovery_duration_s=recovery_duration,
                signal_h_s=signal_h,recovery_h_s=recovery_h,cases=[])
    def save():output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as z:initial,start=z["state"],float(z["time"])
        t0=start+(.020 if not a.quick else .0002)
        outputs={}
        for name,positions in CASES:
            c=circuit(positions);pre,_=simulate(c,initial.copy(),start,.020 if not a.quick else .0002,20e-6)
            idle=circuit(positions)
            idle_all,idle_stats=simulate(idle,pre[-1],t0,signal_duration+recovery_duration,recovery_h)
            idle_end=idle_all[-1]
            c.source_function("Vin",lambda t,origin=t0:input_peak*np.interp((t-origin)*rate,axis,source,left=0.,right=0.))
            driven,signal_stats=simulate(c,pre[-1],t0,signal_duration,signal_h)
            recovered,recovery_stats=simulate(c,driven[-1],t0+signal_duration,recovery_duration,recovery_h)
            diff=recovered[-1]-idle_end
            node=diff[:len(c.nodes)];branch=diff[len(c.nodes):]
            row=dict(name=name,positions=positions,electrical={k:control_position(k,v) for k,v in positions.items()},
                     signal_stats=signal_stats,recovery_stats=recovery_stats,idle_stats=idle_stats,
                     signal_diagnostics=diagnostics(c,driven,signal_h,MODEL["el34_grid_r"]),
                     recovery_error=dict(node_rms_v=float(np.sqrt(np.mean(node*node))),node_max_v=float(np.max(np.abs(node))),
                                         branch_rms_a=float(np.sqrt(np.mean(branch*branch))),branch_max_a=float(np.max(np.abs(branch)))))
            tail=round(.020/recovery_h)+1; y=recovered[-min(tail,len(recovered)):,c.index["out"]]
            row["recovery_output_last20_rms_v"]=float(np.sqrt(np.mean(y*y)))
            result["cases"].append(row);save()
            np.savez_compressed(RAW/f"{name}.npz",signal=driven,recovery=recovered)
            sample_t=np.arange(round(signal_duration*rate)+round(recovery_duration*rate))/rate
            internal_t=np.r_[np.arange(len(driven))*signal_h,signal_duration+np.arange(1,len(recovered))*recovery_h]
            internal_y=np.r_[driven[:,c.index["out"]],recovered[1:,c.index["out"]]]
            outputs[name]=np.interp(sample_t,internal_t,internal_y)
        scale=1.05*max(np.max(np.abs(x)) for x in outputs.values());result["wav_common_scale_v"]=float(scale)
        for name,x in outputs.items():write_wav(DEST/f"{name}.wav",x,rate,scale)
        result["status"]="pass";save();render(result)
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}");save();render(result);raise
    print("Report:",DEST/"report.md")


if __name__=="__main__":main()
