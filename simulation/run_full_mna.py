"""Validate the unreduced solver against ngspice; preserve failed cases explicitly."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
import hashlib
import numpy as np
from full_mna import ROOT, build, electrode
from run_reference import executable, table
import matplotlib.pyplot as plt

NODES = "in p1 p2 p3 cf pi_p1 pi_p2 plate_a plate_b out bplus bscreen bpi bias main_mid s4".split()


def jacobian_check():
    errors = []
    for kind, values in [("triode", [100., -1.]), ("triode", [250., -2.]),
                         ("triode", [100., 2.]), ("pentode", [250., -13.5, 265.]),
                         ("pentode", [450., -42., 465.])]:
        v = np.array(values)
        _, analytic = electrode(kind, v)
        numeric = np.empty_like(analytic)
        for j in range(len(v)):
            d = np.zeros_like(v)
            d[j] = 1e-4
            numeric[:, j] = (electrode(kind, v+d)[0]-electrode(kind, v-d)[0])/(2e-4)
        errors.append(float(np.max(np.abs(numeric-analytic)/np.maximum(np.abs(analytic), 1e-10))))
    if max(errors) > 1e-5:
        raise AssertionError(f"Electrode Jacobian mismatch: {max(errors)}")
    return max(errors)


def run(full, h, stop, name, spice_order=2):
    c = build(full, amplitude=.001, controls=.1)
    folder = ROOT / "simulation/raw/full_mna" / name
    folder.mkdir(parents=True, exist_ok=True)
    exported = c.export()
    (folder / "circuit.inc").write_text(exported, encoding="ascii")
    dc, stats = c.dc()
    current = dc.copy()
    times = np.arange(round(stop/h)+1)*h
    selected = [c.index[n] for n in NODES]
    data = np.empty((len(times), len(NODES)))
    data[0] = current[selected]
    max_kcl, max_voltage, max_iter, retries = 0., 0., 0, 0
    started = time.perf_counter()
    def advance(x, start, duration, depth=0):
        nonlocal max_kcl, max_voltage, max_iter, retries
        try:
            result, s = c.step(x, start+duration, duration)
        except RuntimeError:
            if depth >= 12:
                raise
            retries += 1
            middle = advance(x, start, duration/2, depth+1)
            return advance(middle, start+duration/2, duration/2, depth+1)
        max_kcl = max(max_kcl, s["kcl"])
        max_voltage = max(max_voltage, s["voltage"])
        max_iter = max(max_iter, s["iterations"])
        return result
    for n in range(1, len(times)):
        current = advance(current, times[n-1], h)
        data[n] = current[selected]
        if n % 1000 == 0:
            print(f"{name}: {n}/{len(times)-1} steps, B+={current[c.index['bplus']]:.2f} V", flush=True)
    elapsed = time.perf_counter()-started
    np.savez_compressed(folder / "python.npz", time=times, voltages=data, dc=dc)
    hspice = h/4
    deck = f"""Full MNA independent ngspice verification
.include circuit.inc
.options method=gear maxord={spice_order} reltol=1e-7 abstol=1e-12 vntol=1e-9 itl1=500 itl4=1000 gmin=1e-15 temp=27 tnom=27
.control
set noaskquit
set wr_vecnames
set wr_singlescale
set numdgt=15
op
wrdata dc.txt {' '.join('v('+n+')' for n in c.nodes)}
tran {hspice:.17g} {stop:.17g} 0 {hspice:.17g}
wrdata transient.txt {' '.join('v('+n+')' for n in NODES)}
quit
.endc
.end
"""
    (folder / "run.cir").write_text(deck, encoding="ascii")
    result = subprocess.run([str(executable()), "-b", "run.cir"], cwd=folder,
                            capture_output=True, text=True, errors="replace", timeout=180)
    log = result.stdout+result.stderr
    (folder / "ngspice.log").write_text(log, encoding="utf-8")
    if result.returncode or any(s in log.lower() for s in ("error:", "timestep too small", "fatal")):
        raise RuntimeError(log[-4000:])
    dc_spice = table(folder / "dc.txt")[0, 1:]
    tr_spice = table(folder / "transient.txt")
    if tr_spice[-1, 0] < stop*.999:
        raise RuntimeError("Incomplete SPICE transient")
    interpolated = np.column_stack([np.interp(times, tr_spice[:, 0], tr_spice[:, j+1]) for j in range(len(NODES))])
    # Ignore SPICE's first initialization microstep for transient metrics only.
    delta = data[1:]-interpolated[1:]
    rms = np.sqrt(np.mean(delta**2, axis=0))
    metrics = dict(name=name, full_supply=full, nodes=len(c.nodes), unknowns=c.size, elements=len(c.parts),
                   h=h, spice_max_h=hspice, spice_max_order=spice_order, stop=stop, max_kcl_a=max_kcl, max_voltage_residual_v=max_voltage,
                   max_newton_iterations=max_iter, bisected_steps=retries,
                   dc_max_difference_v=float(np.max(np.abs(dc[:len(c.nodes)]-dc_spice))),
                   rms_difference_v=dict(zip(NODES, rms.tolist())), final_v=dict(zip(NODES, data[-1].tolist())),
                   host_elapsed_s=elapsed, circuit_sha256=hashlib.sha256(exported.encode("ascii")).hexdigest())
    if metrics["dc_max_difference_v"] > 1e-4:
        raise AssertionError(f"DC mismatch: {metrics['dc_max_difference_v']}")
    print(name, "PASS", "DC difference", metrics["dc_max_difference_v"], "output RMS error", metrics["rms_difference_v"]["out"], flush=True)
    return metrics, times, data, interpolated, exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--spice-order", type=int, choices=(1, 2), default=2,
                        help="Order 1 reproduces the saved ngspice convergence failure")
    args = parser.parse_args()
    derivative_error = jacobian_check()
    cases = [run(False, 2e-6, .004, "signal_dc_2us", args.spice_order),
             run(True, 2e-6, .004, "full_start_2us", args.spice_order)]
    if not args.quick:
        cases += [run(False, .5e-6, .004, "signal_dc_05us", args.spice_order),
                  run(True, 10e-6, .060, "full_start_60ms", args.spice_order)]
    dest = ROOT / "simulation/experiments/full_mna"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "full_1981.inc").write_text(cases[-1][4], encoding="ascii")
    (dest / "metrics.json").write_text(json.dumps(dict(jacobian_relative_error=derivative_error,
                                                       cases=[c[0] for c in cases]), indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for node in ("bplus", "bias", "main_mid"):
        index = NODES.index(node)
        axes[0].plot(cases[-1][1]*1000, cases[-1][2][:, index], label=node+" MNA")
        axes[0].plot(cases[-1][1]*1000, cases[-1][3][:, index], "--", label=node+" SPICE", alpha=.6)
    for case in cases[:2]:
        index = NODES.index("out")
        axes[1].plot(case[1]*1000, case[2][:, index], label=case[0]["name"]+" MNA")
        axes[1].plot(case[1]*1000, case[3][:, index], "--", label=case[0]["name"]+" SPICE", alpha=.6)
        axes[2].plot(case[1]*1000, case[2][:, index]-case[3][:, index], label=case[0]["name"])
    for ax, ylabel in zip(axes, ("Supply voltages, V", "Output voltage, V", "MNA minus SPICE, V")):
        ax.set(xlabel="Time, ms", ylabel=ylabel)
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(dest / "comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# Полная схема и собственный общий решатель", "",
             "Реализован общий MNA float64 с аналитическим якобианом, демпфированным Ньютоном",
             "и неявным Эйлером. Узлы не исключаются, каскады не разрываются. Нелинейные",
             "заряды переходов входят в дискретное уравнение заряда, а не заменены постоянной ёмкостью.", "",
             "SPICE получает тот же список элементов и ламповые законы, но решает их своим алгоритмом.",
             "Диоды экспортированы нативными D-моделями с теми же Is/Cjo/Vj/M/Fc/Tt;",
             "численные продолжения диодного тока в SPICE могут отличаться от Python.",
             f"SPICE: Gear maxord={args.spice_order}, reltol=1e-7, abstol=1e-12 A, vntol=1e-9 V.",
             "Максимальный шаг SPICE в четыре раза меньше шага Python. Поэтому временная",
             "разность включает ошибку сетки и разницу интеграторов; это не изолированная ошибка реализации решателя.", "",
             "Вход 1 мВ peak, 1 кГц; Gain/Master — электрические доли 0,1, остальные ручки 0,5.",
             "В полном варианте сеть 230 В RMS, 50 Гц, старт из нулевого электрического состояния.",
             "Эмиссия ламп считается установившейся: тепловой прогрев не моделируется.", "",
             "| Опыт | Узлов / неизвестных | Время, мс | DC max, В | KCL max, А | RMS выхода к SPICE, В | Делений шага |",
             "|:---|:---|---:|---:|---:|---:|---:|"]
    for metrics, *_ in cases:
        lines.append(f"| {metrics['name']} | {metrics['nodes']} / {metrics['unknowns']} | {metrics['stop']*1000:g} | {metrics['dc_max_difference_v']:.3e} | {metrics['max_kcl_a']:.3e} | {metrics['rms_difference_v']['out']:.3e} | {metrics['bisected_steps']} |")
    lines += ["", f"Максимальная относительная ошибка проверки аналитического якобиана: {derivative_error:.3e}.",
              "При несходимости состояние не записывается; интервал делится пополам с ограничением",
              "глубины. DC использует продолжение по источникам и заканчивается исходной системой.", "",
              "Добавлены мост, средняя точка HV, отдельная обмотка и выпрямитель bias, фильтры",
              "10 мкФ / 15 кОм / 10 мкФ, 56 кОм + регулятор 22 кОм (электрическая половина),",
              "нагрузка накала и реальные ветви отводов OT 4/8/16 Ом с током обратной связи.", "",
              "Параметры магнитных элементов и выпрямителей, отсутствующие на схеме, всё ещё",
              "предварительные. Начальные законы Koren сохранены как контроль реализации;",
              "их физические ограничения не устранены написанием решателя. Этот опыт не доказывает",
              "точность конкретного серийного Marshall, магнитное насыщение или установившийся прогрев.", "",
              "![Сверка](comparison.png)", "",
              "Важно: 60 мс сетевого старта не подтверждают установившийся режим.",
              "В сохранённом полном прогоне bias в конце около −1,14 В, Bpi около 7,61 В;",
              "нагрузка питания и смещение ещё изменяются. Нулевой DC этого варианта —",
              "начальное выключенное состояние. Проверка звукового рабочего режима требует",
              "отдельного установления питания и дальнейшего сигнала.", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
