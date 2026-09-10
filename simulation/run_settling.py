"""Periodic supply settling, discrete energy audit, and strong-signal bursts.

All 104 unknowns remain in the solve. Coarse steps are only initialization;
the accepted periodic state is recomputed with the reported final step.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from full_mna import ROOT, VT, build, diode, electrode
from run_reference import executable, table
import numpy as np
import matplotlib.pyplot as plt

RAW = ROOT / "simulation/raw/settling_power"
DEST = ROOT / "simulation/experiments/settling_power"
PERIOD = .02


def diode_energy(v, iss, c, tt):
    if v < .5:
        u = math.sqrt(1-v)
        w = 2*c/3*(u-1)**2*(u+2)
    else:
        u = math.sqrt(.5)
        w = 2*c/3*(u-1)**2*(u+2)+c*math.sqrt(2)*((v**3-.125)/3+.25*(v*v-.25))
    z = v/VT
    if z < 40:
        w += tt*iss*VT*(z*math.exp(z)-math.expm1(z))
    else:
        v0 = 40*VT
        w += tt*iss*VT*(40*math.exp(40)-math.expm1(40))
        w += .5*tt*iss*math.exp(40)/VT*(v*v-v0*v0)
    return w


class Energy:
    def __init__(self, c):
        self.c = c
        self.D = c.C.copy()
        self.D[len(c.nodes):] *= -1
        self.R = c.G[:len(c.nodes), :len(c.nodes)]
        self.n = len(c.nodes)

    def quantities(self, x):
        cur, charge = np.zeros(self.c.size), np.zeros(self.c.size)
        energy = .5*float(x@self.D@x)
        for kind, control, output, params in self.c.nl:
            if kind == "T":
                i, _ = electrode(params, control@x)
                cur += output.T@i
            else:
                v = float(control@x)
                i, _, q, _ = diode(v, *params)
                cur += control*i
                charge += control*q
                energy += diode_energy(v, *params)
        resistive = float(x[:self.n]@self.R@x[:self.n])
        return energy, resistive, float(x@cur), charge

    def step(self, old, new, t, h):
        w0, _, _, q0 = self.quantities(old)
        w, pr, pnl, q = self.quantities(new)
        supplied = -float(new@self.c.rhs(t))
        storage_work = float(new@(self.D@(new-old)+q-q0))
        numerical_loss = storage_work-(w-w0)
        residual = supplied-pr-pnl-storage_work/h
        return supplied*h, pr*h, pnl*h, w-w0, numerical_loss, residual


def advance(c, x, t, h, audit=None, depth=0):
    try:
        y, stats = c.step(x, t+h, h)
    except RuntimeError:
        if depth >= 12:
            raise
        mid, a = advance(c, x, t, h/2, audit, depth+1)
        y, b = advance(c, mid, t+h/2, h/2, audit, depth+1)
        a["halvings"] += b["halvings"]+1
        for key in ("kcl", "voltage", "iterations", "power_residual"):
            a[key] = max(a[key], b[key])
        a["energy"] += b["energy"]
        return y, a
    out = dict(stats, halvings=0, energy=np.zeros(5), power_residual=0.)
    if audit is not None:
        *e, residual = audit.step(x, y, t+h, h)
        out["energy"] = np.array(e)
        out["power_residual"] = abs(residual)
    return y, out


def segment(c, x, t0, duration, h, audit=False):
    count = round(duration/h)
    if abs(count*h-duration) > 1e-12:
        raise ValueError("Duration must be a multiple of h")
    energy = Energy(c) if audit else None
    data = np.empty((count+1, c.size))
    data[0] = x
    total = dict(kcl=0., voltage=0., iterations=0, halvings=0, energy=np.zeros(5), power_residual=0.)
    for j in range(count):
        data[j+1], s = advance(c, data[j], t0+j*h, h, energy)
        total["halvings"] += s["halvings"]
        total["energy"] += s["energy"]
        for key in ("kcl", "voltage", "iterations", "power_residual"):
            total[key] = max(total[key], s[key])
    return data, total


def settle(c, reuse=False):
    cache = RAW / "settled.npz"
    fingerprint = hashlib.sha256(c.export().encode("ascii")).hexdigest()
    if reuse and cache.exists():
        saved = np.load(cache)
        if str(saved["fingerprint"]) != fingerprint:
            raise ValueError("Cached circuit differs")
        meta = json.loads((RAW / "settling.json").read_text(encoding="utf-8"))
        if not meta["accepted"]:
            raise ValueError("Cached state is not accepted")
        return saved["state"], float(saved["time"]), meta
    x, _ = c.dc()
    t = 0.
    history = []
    accepted = False
    # Absolute full-waveform period-to-period tolerances, fixed before the run.
    voltage_tol, current_tol = .01, 20e-6
    for label, h, limit in (("coarse", .001, 30.), ("fine", .0001, 10.)):
        previous = None
        consecutive = 0
        start = t
        for cycle in range(round(limit/PERIOD)):
            data, s = segment(c, x, t, PERIOD, h)
            x = data[-1]
            t += PERIOD
            dv = di = float("inf")
            if previous is not None:
                diff = np.abs(data-previous)
                dv = float(diff[:, :len(c.nodes)].max())
                di = float(diff[:, len(c.nodes):].max())
            previous = data
            consecutive = consecutive+1 if dv <= voltage_tol and di <= current_tol else 0
            row = dict(stage=label, time=t, max_dv=dv if np.isfinite(dv) else None,
                       max_di=di if np.isfinite(di) else None, bplus=float(x[c.index["bplus"]]),
                       bias=float(x[c.index["bias"]]), bpi=float(x[c.index["bpi"]]),
                       kcl=s["kcl"], halvings=s["halvings"])
            history.append(row)
            if cycle % 25 == 0 or consecutive >= 3:
                print(f"{label}: t={t:.3f}s, B+={row['bplus']:.2f}, bias={row['bias']:.3f}, Bpi={row['bpi']:.2f}, dV={dv:.4g}, dI={di:.4g}", flush=True)
                np.savez_compressed(RAW / "checkpoint.npz", state=x, time=t)
                (RAW / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if consecutive >= 3 and t-start >= .2:
                accepted = label == "fine"
                break
        else:
            raise RuntimeError(f"Periodic settling tolerance not reached in {label} stage")
    meta = dict(accepted=accepted, voltage_tolerance_v=voltage_tol, current_tolerance_a=current_tol,
                required_consecutive_cycles=3, final_h_s=h, history=history)
    np.savez_compressed(cache, state=x, time=t, fingerprint=fingerprint)
    (RAW / "settling.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return x, t, meta


def describe(c, data, stats, duration):
    info = dict(max_kcl_a=stats["kcl"], max_voltage_residual_v=stats["voltage"],
                max_iterations=stats["iterations"], halvings=stats["halvings"],
                max_power_identity_residual_w=stats["power_residual"])
    names = "bplus bscreen bpi bpre2 bpre1 bias out".split()
    info["nodes"] = {n: dict(mean=float(data[1:, c.index[n]].mean()),
                             pp=float(np.ptp(data[1:, c.index[n]]))) for n in names}
    y = data[1:, c.index["out"]]
    info["output_rms_v"] = float(np.sqrt(np.mean(y*y)))
    info["load_power_w"] = float(np.mean(y*y)/16)
    heater = data[1:, c.index["heater1"]]-data[1:, c.index["heater2"]]
    info["heater_power_w"] = float(np.mean(heater*heater)/(6.3/6.9))
    info["mean_power_w"] = dict(zip(("sources", "resistors", "nonlinear_devices", "storage_change", "backward_euler_loss"),
                                   (stats["energy"]/duration).tolist()))
    tubes = {}
    for kind, name, *a in c.parts:
        if kind != "T" or a[0] != "pentode": continue
        _, p, g, k, screen = a
        vals = data[1:, [c.index[p], c.index[g], c.index[screen]]]
        currents = np.array([electrode("pentode", v)[0] for v in vals])
        tubes[name] = dict(plate_mean_a=float(currents[:, 0].mean()),
                           screen_mean_a=float(currents[:, 1].mean()),
                           plate_dissipation_w=float(np.mean(vals[:, 0]*currents[:, 0])),
                           screen_dissipation_w=float(np.mean(vals[:, 2]*currents[:, 1])))
    info["el34"] = tubes
    if stats["power_residual"] > 1e-4:
        raise AssertionError("Electrical power identity failed")
    return info


def spice_burst(c, initial, duration=.02, h=1e-6):
    folder = RAW / "spice_100mv"
    folder.mkdir(exist_ok=True)
    lines = []
    for line in c.export().splitlines():
        name = line.split()[0] if line.split() else ""
        if name in c.branches and name.startswith("L"):
            line += f" IC={initial[c.branches[name]]:.17g}"
        lines.append(line)
    for node in c.nodes:
        lines.append(f".ic v({node})={initial[c.index[node]]:.17g}")
    (folder / "circuit.inc").write_text("\n".join(lines), encoding="ascii")
    deck = f"""Strong signal from common accepted electrical initial state
.include circuit.inc
.options method=gear maxord=2 reltol=1e-7 abstol=1e-12 vntol=1e-9 itl4=1000 gmin=1e-15 temp=27 tnom=27
.control
set wr_vecnames
set wr_singlescale
set numdgt=15
tran {h} {duration} 0 {h} uic
wrdata trace.txt v(out) v(bplus) v(bias)
quit
.endc
.end
"""
    (folder / "run.cir").write_text(deck, encoding="ascii")
    result = subprocess.run([str(executable()), "-b", "run.cir"], cwd=folder,
                            capture_output=True, text=True, errors="replace", timeout=300)
    log = result.stdout+result.stderr
    (folder / "ngspice.log").write_text(log, encoding="utf-8")
    if result.returncode or "timestep too small" in log.lower() or "error:" in log.lower():
        raise RuntimeError(log[-2500:])
    data = table(folder / "trace.txt")
    if data[-1, 0] < duration*.999:
        raise RuntimeError("Incomplete ngspice burst")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-settled", action="store_true")
    parser.add_argument("--settle-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild figures/report from saved successful runs")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    c = build(True, amplitude=0., controls=.5)
    if args.report_only:
        results = json.loads((DEST / "metrics.json").read_text(encoding="utf-8"))
        if results["circuit_sha256"] != hashlib.sha256(c.export().encode("ascii")).hexdigest():
            raise ValueError("Saved results refer to another circuit")
        idle = np.load(RAW / "idle.npz")["states"]
        heater = idle[1:, c.index["heater1"]]-idle[1:, c.index["heater2"]]
        results["idle"]["heater_power_w"] = float(np.mean(heater*heater)/(6.3/6.9))
        bursts = []
        for info in results["bursts"]:
            a = info["amplitude_v"]
            saved = np.load(RAW / f"burst_{round(a*1000)}mv.npz")
            bursts.append((a, saved["on"], saved["off"], info))
        (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        plot_and_report(c, results, idle, bursts)
        return
    initial, t, settling = settle(c, args.reuse_settled)
    results = dict(settling=settling, circuit_sha256=hashlib.sha256(c.export().encode("ascii")).hexdigest())
    data, stats = segment(c, initial, t, PERIOD, .0001, audit=True)
    results["idle"] = describe(c, data, stats, PERIOD)
    np.savez_compressed(RAW / "idle.npz", states=data, time=np.linspace(0, PERIOD, len(data)))
    print("Idle audit:", json.dumps(results["idle"]), flush=True)
    bursts = []
    if not args.settle_only:
        for amplitude in (.025, .1, .5):
            c.sources["Vin"] = (0., amplitude, 1000., 0.)
            print(f"Burst {amplitude} V", flush=True)
            on, st = segment(c, initial, 0., .02, 5e-6, audit=True)
            info = describe(c, on, st, .02)
            info["amplitude_v"] = amplitude
            c.sources["Vin"] = (0., 0., 1000., 0.)
            off, _ = segment(c, on[-1], .02, .02, 10e-6)
            info["release_last_output_v"] = float(off[-1, c.index["out"]])
            bursts.append((amplitude, on, off, info))
            np.savez_compressed(RAW / f"burst_{round(amplitude*1000)}mv.npz", on=on, off=off)
            if amplitude == .1:
                c.sources["Vin"] = (0., amplitude, 1000., 0.)
                ref = spice_burst(c, initial)
                grid = np.arange(1, len(on))*5e-6
                delta = on[1:, c.index["out"]]-np.interp(grid, ref[:, 0], ref[:, 1])
                info["spice_rms_difference_v"] = float(np.sqrt(np.mean(delta*delta)))
                info["spice_relative_rms"] = float(np.linalg.norm(delta)/np.linalg.norm(np.interp(grid, ref[:, 0], ref[:, 1])))
            print("Burst done:", json.dumps(info), flush=True)
            results["bursts"] = [b[3] for b in bursts]
            (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["bursts"] = [b[3] for b in bursts]
    (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_and_report(c, results, data, bursts)


def plot_and_report(c, results, idle, bursts):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    history = results["settling"]["history"]
    for n in ("bplus", "bias", "bpi"):
        axes[0].plot([r["time"] for r in history], [r[n] for r in history], label=n)
    axes[0].set(xlabel="Elapsed electrical settling, s", ylabel="Phase-zero voltage, V")
    for n in ("bplus", "bscreen", "bias"):
        axes[1].plot(np.linspace(0, 20, len(idle)), idle[:, c.index[n]], label=n)
    axes[1].set(xlabel="One mains period, ms", ylabel="Voltage, V")
    for a, on, off, info in bursts:
        axes[2].plot(np.linspace(0, 20, len(on)), on[:, c.index["out"]], label=f"{a*1000:g} mV")
    axes[2].set(xlabel="Burst time, ms", ylabel="Output, V", xlim=(0, 5))
    for ax in axes:
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(DEST / "results.png", dpi=150)
    plt.close(fig)
    idle_info = results["idle"]
    lines = ["# Установление питания, баланс мощности и сильный сигнал", "",
             "Полная MNA: 90 узлов / 104 неизвестных, без исключения каскадов. Лампы считаются горячими.",
             "Сеть 230 В RMS / 50 Гц, нагрузка 16 Ом; все ручки электрически 0,5.",
             "Начальные магнитные и ламповые параметры прежние; совпадение с реальным усилителем не заявляется.", "",
             "## Критерий установления", "",
             "Сравниваются ВСЕ напряжения и токи ветвей на всей сетке двух соседних периодов сети.",
             "Допуски: 10 мВ и 20 мкА, три периода подряд. Инициализация шагом 1 мс,",
             "затем повторное установление шагом 100 мкс. Критерий проверяет периодичность при этом шаге,",
             "а не отсутствие ошибки дискретизации. Все нелинейности сохранены на обеих сетках.",
             f"Принятое время электрического установления: {history[-1]['time']:.3f} с.", "",
             "| Узел | Среднее, В | Размах пульсаций, В |", "|:---|---:|---:|"]
    for node, val in idle_info["nodes"].items():
        lines.append(f"| {node} | {val['mean']:.6f} | {val['pp']:.6f} |")
    lines += ["", "## Баланс за период без сигнала", "",
              "Источник отдаёт энергию резисторам, токовым ветвям ламп/диодов, накопителям",
              "и численной диссипации неявного Эйлера. Последняя учитывается отдельно:",
              "она не является физическим нагревом. Включены взаимные индуктивности и",
              "энергия нелинейных зарядов диодов. Накал представлен резистивной нагрузкой.", "",
              "| Составляющая | Средняя мощность, Вт |", "|:---|---:|"]
    for label, value in idle_info["mean_power_w"].items():
        lines.append(f"| {label} | {value:.9f} |")
    lines += ["", f"Максимальная невязка мгновенного дискретного баланса: {idle_info['max_power_identity_residual_w']:.3e} Вт.",
              "Замыкание баланса проверяет учёт токов и энергии, но само по себе не аттестует физические параметры.", "",
              "## EL34 без сигнала", "", "| Лампа | Ia, мА | Ig2, мА | Pa, Вт | Pg2, Вт |",
              "|:---|---:|---:|---:|---:|"]
    for name, values in idle_info["el34"].items():
        lines.append(f"| {name} | {values['plate_mean_a']*1000:.6f} | {values['screen_mean_a']*1000:.6f} | {values['plate_dissipation_w']:.6f} | {values['screen_dissipation_w']:.6f} |")
    lines += ["", "## Сильный сигнал", "",
              "Каждый опыт начинается из одного принятого периодического состояния: 20 мс синуса",
              "1 кГц (шаг 5 мкс), затем 20 мс без входа (10 мкс). Это атака и короткое восстановление,",
              "а не установившаяся перегрузка. Шаг нового опыта отличается от сетки инициализации.", "",
              "| Вход peak, мВ | Выход RMS, В | Нагрузка, Вт | KCL max, А | Баланс max, Вт |",
              "|---:|---:|---:|---:|---:|"]
    for info in results["bursts"]:
        lines.append(f"| {info['amplitude_v']*1000:g} | {info['output_rms_v']:.6f} | {info['load_power_w']:.6f} | {info['max_kcl_a']:.3e} | {info['max_power_identity_residual_w']:.3e} |")
    for info in results["bursts"]:
        if "spice_relative_rms" in info:
            lines += ["", f"100 мВ: разность с ngspice Gear до 2-го порядка / maxstep 1 мкс — {100*info['spice_relative_rms']:.6f}% RMS.",
                      "Оба расчёта начинают с одних узловых напряжений и токов индуктивностей;",
                      "SPICE не устанавливает питание заново. Разность включает интеграторы и сетки.", ""]
    lines += ["", "## Интерпретация и ограничения", "",
              "Нулевой Ig2 при анодном токе около 21 мА — ограничение текущего закона EL34:",
              "экранная ветвь обнуляется при Vs/11 + Vg <= 0. Сходимость Ньютона и совпадение",
              "со SPICE не подтверждают этот закон физически; нужна отдельная проверка экранных характеристик.", "",
              "264–275 Вт в нагрузке относятся только к первым 20 мс атаки. Накопители отдают",
              "в среднем 141–147 Вт; это не проверка длительной паспортной мощности 100 Вт.",
              "После 20 мс отключённого входа выход ещё не вернулся к покою; полное восстановление не измерено.", "",
              "Для атак 25/100/500 мВ потребовалось соответственно 15/42/55 делений шага после",
              "неудачи Ньютона. Это проверяет защитный механизм на данных опытах, но не даёт гарантии",
              "для любого входа. Разность со SPICE 2,56% требует отдельного сгущения сетки в перегрузке;",
              "сходимость по шагу предыдущего малосигнального опыта на этот режим не переносится.", "",
              "![Результаты](results.png)", ""]
    (DEST / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
