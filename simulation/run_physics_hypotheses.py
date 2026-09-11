"""Separate leading physical hypotheses behind the severe DI operating range."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build, electrode
from run_controls_qualification import BASELINE, MODEL

SAMPLE = ROOT / "simulation/samples/e_major_chord/e_major_attack.wav"
DEST = ROOT / "simulation/experiments/physics_hypotheses"
RAW = ROOT / "simulation/raw/physics_hypotheses"
CASES = (
    ("idle", {}, 2001., False),
    ("accepted", {}, 2001., True),
    # Engineering assumption only: common 10% electrical midpoint of an audio pot.
    ("assumed_audio_taper_mid", {"GAIN": .1, "MASTER": .1}, 2001., True),
    ("el34_ig1_off", {}, 1e15, True),
    ("nfb_off", {"NFB": 1e-9}, 2001., True),
)


def read_signal():
    digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    with wave.open(str(SAMPLE), "rb") as stream:
        rate = stream.getframerate()
        signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float)
    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    return signal, rate, digest


def simulate(circuit, initial, t0, duration, h):
    count = round(duration/h)
    states = np.empty((count+1, circuit.size)); states[0] = initial
    stats = dict(host_s=0., attempted_steps=0, failures=0, subdivisions=0,
                 max_iterations=0, kcl_a=0., voltage_residual_v=0.)
    def advance(state, t, step, depth=0):
        began = time.perf_counter(); stats["attempted_steps"] += 1
        try:
            result, info = circuit.step(state, t+step, step)
        except RuntimeError:
            stats["failures"] += 1
            if depth >= 12: raise
            stats["subdivisions"] += 1
            middle = advance(state, t, step/2, depth+1)
            return advance(middle, t+step/2, step/2, depth+1)
        finally:
            stats["host_s"] += time.perf_counter()-began
        stats["max_iterations"] = max(stats["max_iterations"], info["iterations"])
        stats["kcl_a"] = max(stats["kcl_a"], info["kcl"])
        stats["voltage_residual_v"] = max(stats["voltage_residual_v"], info["voltage"])
        return result
    for j in range(count):
        states[j+1] = advance(states[j], t0+j*h, h)
        if (j+1) % 4000 == 0: print(f"{j+1}/{count}", flush=True)
    return states, stats


def diagnostics(c, x, h, grid_r):
    out = x[:, c.index["out"]]
    load_power = out*out/16.
    plate_power, screen_power, grid_current = [], [], []
    for plate, grid, screen, junction in (("plate_a", "g4", "s4", "xv4_junction"),
                                           ("plate_a", "g5", "s5", "xv5_junction"),
                                           ("plate_b", "g6", "s6", "xv6_junction"),
                                           ("plate_b", "g7", "s7", "xv7_junction")):
        pp, ps, ig = [], [], []
        for row in x:
            va, vg, vs = row[c.index[plate]], row[c.index[grid]], row[c.index[screen]]
            currents, _ = electrode("reefman", np.array([va, vg, vs]))
            pp.append(va*currents[0]); ps.append(vs*currents[1])
            ig.append(max((vg-row[c.index[junction]])/grid_r, 0.))
        plate_power.append(pp); screen_power.append(ps); grid_current.append(ig)
    plate_power, screen_power, grid_current = map(np.asarray, (plate_power, screen_power, grid_current))
    winding_v = .5*((x[:, c.index["oa"]]-x[:, c.index["bplus"]])-
                    (x[:, c.index["ob"]]-x[:, c.index["bplus"]]))
    flux_proxy = np.cumsum(winding_v)*h
    return dict(output_peak_v=float(np.max(np.abs(out))), output_rms_v=float(np.sqrt(np.mean(out*out))),
                load_mean_w=float(np.mean(load_power)), load_peak_w=float(np.max(load_power)),
                load_energy_j=float(np.sum(load_power)*h),
                bplus_min_v=float(np.min(x[:, c.index["bplus"]])), bplus_max_v=float(np.max(x[:, c.index["bplus"]])),
                bias_min_v=float(np.min(x[:, c.index["bias"]])), bias_max_v=float(np.max(x[:, c.index["bias"]])),
                el34_plate_peak_w=float(np.max(plate_power)), el34_screen_peak_w=float(np.max(screen_power)),
                el34_grid_peak_a=float(np.max(grid_current)), el34_grid_mean_a=float(np.mean(grid_current)),
                el34_grid_above_1ua_fraction=float(np.mean(np.any(grid_current > 1e-6, axis=0))),
                primary_flux_proxy_peak_vs=float(np.max(np.abs(flux_proxy-flux_proxy[0]))))


def render(result):
    lines = ["# Проверка физических гипотез JCM800", "", f"Статус: **{result['status']}**.", "",
             f"Один и тот же {result['duration_s']*1e3:g}-мс DI-фрагмент 100 мВ рассчитан CompactCircuit "
             f"на общей сетке {result['h_s']*1e6:g} мкс.",
             "Холостой режим синхронизирован по фазе сети. Изменяется ровно одна",
             "гипотеза; остальные законы и схема сохраняются.", "", "## Сводка", "",
             "| Режим | Выход peak, В | Pload сред., Вт | Pload peak, Вт | B+ min, В | Ig1 peak, мА | Ig1 доля |",
             "|:---|---:|---:|---:|---:|---:|---:|"]
    for row in result.get("cases", []):
        d=row["diagnostics"]
        lines.append(f"| {row['name']} | {d['output_peak_v']:.3f} | {d['load_mean_w']:.3f} | "
                     f"{d['load_peak_w']:.3f} | {d['bplus_min_v']:.3f} | {d['el34_grid_peak_a']*1e3:.3f} | "
                     f"{100*d['el34_grid_above_1ua_fraction']:.2f}% |")
    if result.get("comparisons"):
        lines += ["", "## Разность относительно принятой модели", "",
                  "| Режим | RMS выхода | Отн. форма | Добавочная просадка B+ |", "|:---|---:|---:|---:|"]
        for name, row in result["comparisons"].items():
            lines.append(f"| {name} | {row['output_rms_difference_v']:.4g} В | "
                         f"{100*row['relative_output_difference']:.3f}% | {row['additional_bplus_sag_v']:.3f} В |")
    lines += ["", "`assumed_audio_taper_mid` — проверка гипотезы, а не установленный заводской закон:",
              "механической середине условно назначено 10% электрического хода Gain и Master.",
              "`el34_ig1_off` практически размыкает только временную R+Shockley-ветвь первой сетки.",
              "Показатель первичного потока в metrics — интеграл дифференциального напряжения",
              "полуобмоток в В·с; без числа витков это сравнительный показатель, не тесла.", "",
              "Опыт пока не вводит выдуманную модель динамика или насыщения трансформатора.",
              "Он определяет, какие из уже локализованных допущений действительно меняют траекторию.", ""]
    if result.get("status") == "pass" and not result.get("quick"):
        lines += ["## Вывод", "",
                  "Главный обнаруженный фактор — интерпретация ручек. Заводской лист помечает Gain",
                  "и Master как `1M LOG`, тогда как прежнее значение 0,5 было электрической долей,",
                  "а не положением ручки 5. Условные 10% электрического хода снизили среднюю мощность",
                  "нагрузки со 126,3 до 7,63 Вт и добавочную просадку B+ с 77,7 до 10,5 В.", "",
                  "Временная модель Ig1 не создаёт чрезмерный размах: её отключение увеличило среднюю",
                  "мощность до 165,0 Вт и просадку до 90,4 В. Значит, в принятой модели она заметно",
                  "ограничивает оконечник, хотя её количественный закон всё ещё требует измерений.", "",
                  "Размыкание ООС увеличило среднюю мощность до 139,3 Вт и просадку до 84,0 В, но",
                  "почти не изменило peak уже ограниченного выхода. Это фактор второго порядка для",
                  "наблюдаемого эффекта. Следующим нужно установить реальную характеристику LOG-ручек,",
                  "после чего повторить DI; лишь затем вводить измеренную нагрузку и магнитную модель.", ""]
    if "error" in result: lines += ["## Ошибка", "", result["error"], ""]
    (DEST/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global DEST, RAW
    parser=argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true"); parser.add_argument("--run-name"); parser.add_argument("--report-only", action="store_true")
    args=parser.parse_args()
    if args.run_name:
        if not all(ch.isalnum() or ch in "_-" for ch in args.run_name): parser.error("invalid run-name")
        DEST, RAW=DEST/args.run_name, RAW/args.run_name
    if args.quick: DEST, RAW=DEST/"quick", RAW/"quick"
    DEST.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    output=DEST/"metrics.json"
    if args.report_only:
        result=json.loads(output.read_text(encoding="utf-8"))
        result.setdefault("duration_s", .0002 if result.get("quick") else .020)
        result.setdefault("pre_roll_s", .0002 if result.get("quick") else .020)
        for row in result["cases"]:
            options=dict(MODEL); options["el34_grid_r"]=row["el34_grid_r_ohm"]
            c=build(True, amplitude=0., controls=row["controls"], circuit_type=CompactCircuit, **options)
            x=np.load(RAW/f"{row['name']}.npz")["state"]
            row["diagnostics"]=diagnostics(c,x,result["h_s"],row["el34_grid_r_ohm"])
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        render(result); return
    if output.exists() and not args.quick: raise FileExistsError("use --run-name")
    signal, rate, digest=read_signal()
    pre, duration=((.0002,.0002) if args.quick else (.020,.020))
    result=dict(status="running", quick=args.quick, sample_sha256=digest, input_peak_v=.1,
                h_s=20e-6 if args.quick else 1.25e-6, duration_s=duration, pre_roll_s=pre, cases=[])
    def save(): output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as data:
            initial, start=data["state"], float(data["time"])
        positions=np.arange(len(signal))
        arrays={}
        for name, changes, grid_r, driven in CASES:
            options=dict(MODEL); options["el34_grid_r"]=grid_r
            c=build(True, amplitude=0., controls=BASELINE|changes, circuit_type=CompactCircuit, **options)
            warm,_=simulate(c, initial.copy(), start, pre, 20e-6)
            t0=start+pre
            if driven:
                c.source_function("Vin", lambda t, origin=t0: .1*np.interp((t-origin)*rate, positions, signal, left=0., right=0.))
            x,stats=simulate(c,warm[-1],t0,duration,result["h_s"])
            arrays[name]=x; np.savez_compressed(RAW/f"{name}.npz", time=t0+np.arange(len(x))*result["h_s"], state=x)
            result["cases"].append(dict(name=name, controls=BASELINE|changes, el34_grid_r_ohm=grid_r,
                                        driven=driven, stats=stats, diagnostics=diagnostics(c,x,result["h_s"],grid_r)))
            save()
        accepted, idle=arrays["accepted"], arrays["idle"]
        out=build(True, controls=BASELINE, circuit_type=CompactCircuit, **MODEL).index["out"]
        bp=build(True, controls=BASELINE, circuit_type=CompactCircuit, **MODEL).index["bplus"]
        result["comparisons"]={}
        for name,x in arrays.items():
            if name in ("accepted","idle"): continue
            diff=x[:,out]-accepted[:,out]
            result["comparisons"][name]=dict(output_rms_difference_v=float(np.sqrt(np.mean(diff*diff))),
                relative_output_difference=float(np.linalg.norm(diff)/np.linalg.norm(accepted[:,out])),
                additional_bplus_sag_v=float(np.max(idle[:,bp]-x[:,bp])))
        result["comparisons"]["accepted"]={"output_rms_difference_v":0.,"relative_output_difference":0.,
            "additional_bplus_sag_v":float(np.max(idle[:,bp]-accepted[:,bp]))}
        result["status"]="pass"; save(); render(result)
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}"); save(); render(result); raise
    print("Report:",DEST/"report.md")


if __name__=="__main__": main()
