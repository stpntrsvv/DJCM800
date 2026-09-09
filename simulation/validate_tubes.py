"""Compare explicit tube laws with ngspice DC and published Philips anchor points."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
from run_reference import ROOT, executable, table
import numpy as np
import matplotlib.pyplot as plt


def triode(vp, vg):
    e = vp / 600 * np.logaddexp(0, 600 * (1 / 100 + vg / np.sqrt(300 + vp * vp)))
    return 2 * np.maximum(e, 0) ** 1.4 / 1060


def pentode(vp, vg, vs):
    e = np.maximum(vs, 0) / 60 * np.logaddexp(0, 60 * (1 / 11 + vg / np.maximum(vs, .001)))
    plate = 2 * np.maximum(e, 0) ** 1.35 / 650 * np.arctan(np.maximum(vp, 0) / 24)
    screen = np.maximum(vs / 11 + vg, 0) ** 1.35 / 4200
    return plate, screen


def spice_point(name, vp, vg, vs=None):
    folder = ROOT / "simulation/raw/tube_validation" / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "simulation/ngspice/tubes.lib", folder / "tubes.lib")
    device = "Xt p g 0 ECC83_K" if vs is None else "Xt p g 0 s EL34_K"
    screen_source = "" if vs is None else f"Vs s 0 {vs}"
    screen_vector = "" if vs is None else " iscreen"
    screen_let = "" if vs is None else "let iscreen = -i(Vs)"
    text = f"""Tube DC anchor {name}
.include tubes.lib
Vp p 0 {vp}
Vg g 0 {vg}
{screen_source}
{device}
.options reltol=1e-9 abstol=1e-14 vntol=1e-10
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
op
let ip = -i(Vp)
{screen_let}
wrdata point.txt ip{screen_vector}
quit
.endc
.end
"""
    (folder / "point.cir").write_text(text, encoding="ascii")
    run = subprocess.run([str(executable()), "-b", "point.cir"], cwd=folder,
                         capture_output=True, text=True, errors="replace", timeout=30)
    log = run.stdout + run.stderr
    (folder / "ngspice.log").write_text(log, encoding="utf-8")
    if run.returncode or "error:" in log.lower():
        raise RuntimeError(log)
    return table(folder / "point.txt")[0, 1:]


def main():
    dest = ROOT / "simulation/experiments/tube_validation"
    dest.mkdir(parents=True, exist_ok=True)
    # Philips ECC83 1970 p.2; EL34 1969 p.2, Rg2=0 => screen at 265V supply.
    anchors = [("ECC83_100V", 100., -1., None, .0005, .00125, None),
               ("ECC83_250V", 250., -2., None, .0012, .0016, None),
               ("EL34_classA", 250., -13.5, 265., .100, .0125, .0149)]
    results = []
    for name, vp, vg, vs, ref_ip, ref_gm, ref_is in anchors:
        fn = (lambda g: triode(vp, g)) if vs is None else (lambda g: pentode(vp, g, vs)[0])
        ip = float(fn(vg))
        gm = float((fn(vg + 1e-4) - fn(vg - 1e-4)) / 2e-4)
        measured = spice_point(name, vp, vg, vs)
        # SPICE includes the explicitly retained 1 GOhm plate resistor.
        agreement = abs(float(measured[0]) - (ip + vp / 1e9))
        if agreement > 1e-11:
            raise AssertionError(f"SPICE/Python current mismatch: {name}: {agreement}")
        item = dict(name=name, vp=vp, vg=vg, vs=vs, ip_a=ip, gm_s=gm,
                    reference_ip_a=ref_ip, reference_gm_s=ref_gm,
                    spice_python_difference_a=agreement)
        if vs is not None:
            item.update(screen_a=float(pentode(vp, vg, vs)[1]), reference_screen_a=ref_is)
            if abs(measured[1] - item["screen_a"]) > 1e-11:
                raise AssertionError("Screen current mismatch")
        results.append(item)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    v = np.linspace(0, 600, 601)
    for g in range(-5, 1):
        ax[0].plot(v, 1e3 * triode(v, g), label=f"Vg={g} V")
    for g in range(-50, 1, 10):
        ax[1].plot(v, 1e3 * pentode(v, g, 450)[0], label=f"Vg={g} V")
    for a, title in zip(ax, ("ECC83 Koren", "EL34 Koren, screen 450 V")):
        a.set(title=title, xlabel="Plate-cathode, V", ylabel="Plate current, mA")
        a.grid(True, alpha=.3)
        a.legend(fontsize=7)
    fig.savefig(dest / "plate_curves.png", dpi=150)
    plt.close(fig)
    lines = ["# Начальная проверка моделей ламп", "",
             "Сравниваются две разные вещи: реализация закона (Python против ngspice)",
             "и физическая точность начальных параметров (против табличных точек Philips).",
             "Это не оцифровка всех заводских кривых и не подбор параметров.", "",
             "Источники: [ECC83, 1970, стр. 2](https://www.drtube.com/datasheets/ecc83-philips1970.pdf)",
             "и [EL34, 1969, стр. 2](https://www.drtube.com/datasheets/el34-philips1969.pdf).",
             "Для EL34 взята колонка Rg2=0: Va=250 В, Vg1=−13,5 В, Vg2=265 В.", "",
             "| Точка | Ia модель, мА | Ia Philips, мА | gm модель, мА/В | gm Philips, мА/В |",
             "|:---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['ip_a']*1000:.6f} | {r['reference_ip_a']*1000:g} | {r['gm_s']*1000:.6f} | {r['reference_gm_s']*1000:g} |")
    el = results[-1]
    lines += ["", f"Экранный ток EL34: модель {el['screen_a']*1000:.6f} мА, Philips 14,9 мА.",
              "Это существенное расхождение; перед исследованием питания и оконечного перегруза",
              "экранный закон требует улучшения либо выбора другой проверенной модели.", "",
              f"Максимальное расхождение Python/ngspice по анодному току: {max(r['spice_python_difference_a'] for r in results):.3e} А.",
              "В сравнении учтён явно заданный резистор анод–катод 1 ГОм. Крутизна вычислена",
              "центральной разностью с приращением сетки 0,1 мВ. Параметры не подгонялись.", "",
              "![Семейства модельных характеристик](plate_curves.png)", "",
              "Нарисованные кривые — модель Koren, а не экспериментальные точки Philips.",
              "Отдельно ещё нужны проверка сеточного тока, ёмкостей и характеристик в области",
              "катодного повторителя. Сходимость SPICE сама по себе не аттестует физику.", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (dest / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Tube DC checks passed; report: simulation/experiments/tube_validation/report.md")


if __name__ == "__main__":
    main()
