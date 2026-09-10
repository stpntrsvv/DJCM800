"""Воспроизводимый отдельный стенд ламп: законы, якобианы, SPICE, Philips.

Не запускает усилитель и не перезаписывает tube_validation (старый опыт).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import numpy as np
import scipy
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tube_models import dempwolf, reefman, grid_diode, TRIODES, EL34
from full_mna import electrode

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "simulation/experiments/detailed_tubes"
RAW = ROOT / "simulation/raw/detailed_tubes"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def author_reference():
    """Use the author's equations as independent SPICE reference, not our export.

    Only syntax adaptations: explicit parameter declaration for ngspice,
    PWRS(x,e) -> sgn(x)*abs(x)^e, LOG -> LN. Source stays byte-identical.
    """
    path = ROOT / "reference/originals/reefman/TubeLib.inc"
    manifest = json.loads((ROOT / "reference/tube_validation_sources.json").read_text())
    if digest(path) != manifest["library_sha256"]:
        raise ValueError("Author TubeLib hash mismatch")
    text = path.read_text(encoding="latin1")
    block = re.search(r"(?im)^\.SUBCKT EL34 .*?^\.ENDS[^\n]*", text, re.S).group()
    parameter_text = block[block.index("MU="):].split(".ENDS")[0]
    pairs = re.findall(r"(\w+)\s*=\s*([+\-\d.]+(?:[eE][+\-]?\d+)?[pP]?)", parameter_text)
    params = " ".join(f"{key}={value}" for key, value in pairs)
    generic = re.search(r"(?im)^\.SUBCKT BTetrodeD .*?^\.ENDS[^\n]*", text, re.S).group()
    lines = generic.splitlines()
    lines[0] = ".subckt EL34_REFERENCE 1 2 3 4 params: "+params
    lines[-1] = ".ends EL34_REFERENCE"
    reference = "\n".join(lines).replace("LOG(", "LN(")
    reference = reference.replace("PWRS(V(7),EX)", "(sgn(V(7))*abs(V(7))^EX)")
    (RAW / "author_reference.inc").write_text(reference+"\n", encoding="ascii")
    return reference


def triode_reference(p):
    # Separate SPICE expression from the published equations, no Python current export.
    return f""".subckt TRI_REFERENCE a g k
.func sp(x) {{max(x,0)+ln(1+exp(-abs(x)))}}
Bk a k I={{{p.G}*(sp({p.C}*(v(a,k)/{p.mu}+v(g,k)))/{p.C})^{p.gamma}}}
Bg g a I={{{p.Gg}*(sp({p.Cg}*v(g,k))/{p.Cg})^{p.xi}+{p.Ig0}}}
.ends TRI_REFERENCE
"""


def spice_sweep(name, reference, vg, vs=None):
    folder = RAW / name
    folder.mkdir(exist_ok=True)
    device = "Xt a g 0 TRI_REFERENCE" if vs is None else "Xt a s g 0 EL34_REFERENCE"
    screen = "" if vs is None else f"Vs s 0 {vs}"
    vector = "i(Vg)" if vs is None else "i(Vs)"
    start, stop, step = (0, 320, 5) if vs is None else (0, 600, 5)
    deck = f"""Independent isolated tube sweep
{reference}
Va a 0 0
Vg g 0 {vg}
{screen}
{device}
.options reltol=1e-10 abstol=1e-14 vntol=1e-11 gmin=1e-15 temp=27 tnom=27
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
dc Va {start} {stop} {step}
wrdata trace.txt i(Va) {vector}
quit
.endc
.end
"""
    (folder / "run.cir").write_text(deck)
    run = subprocess.run([shutil.which("ngspice"), "-b", "run.cir"], cwd=folder, capture_output=True, text=True, timeout=30)
    log = run.stdout+run.stderr
    (folder / "ngspice.log").write_text(log)
    if run.returncode or any(w in log.lower() for w in ("error:", "fatal", "timestep too small")):
        raise RuntimeError(log[-3000:])
    trace = np.loadtxt(folder / "trace.txt", skiprows=1, ndmin=2)
    if len(trace) != round((stop-start)/step)+1 or not np.isfinite(trace).all():
        raise AssertionError("Incomplete/nonfinite independent sweep")
    return trace


def check_jacobian(fn, points):
    worst_abs, worst_scaled = 0., 0.
    for v in points:
        v = np.array(v, dtype=float)
        _, analytic = fn(*v)
        numeric = np.empty_like(analytic)
        for j in range(len(v)):
            h = 1e-4
            delta = np.zeros_like(v)
            delta[j] = h
            numeric[:, j] = (fn(*(v+delta))[0]-fn(*(v-delta))[0])/(2*h)
        error = np.abs(analytic-numeric)
        worst_abs = max(worst_abs, float(error.max()))
        worst_scaled = max(worst_scaled, float((error/(1e-9+1e-5*np.abs(analytic))).max()))
    if worst_scaled > 1:
        raise AssertionError(f"Analytic Jacobian failed: {worst_scaled}")
    return dict(points=len(points), max_abs_a_per_v=worst_abs, max_scaled_error=worst_scaled,
                tolerance="1e-9 A/V + 1e-5 * abs(analytic)")


def anchors():
    rows = []
    for va, vg, target, gm in [(100., -1., .0005, .00125), (250., -2., .0012, .0016)]:
        old, jac = electrode("triode", [va, vg])
        models = [("Koren", old[0], jac[0, 1])]
        for name, p in TRIODES.items():
            cur, jac = dempwolf(va, vg, p)
            models.append(("DZ "+name, cur[0], jac[0, 1]))
        for name, current, slope in models:
            rows.append(dict(tube="ECC83", model=name, va=va, vg=vg, vs=None,
                             ia_a=float(current), reference_ia_a=target, gm_s=float(slope), reference_gm_s=gm))
    # Both Philips class-A columns. 245 V is inferred from 265V-2k*10mA,
    # not an independently specified screen voltage.
    for vg, vs, target, target_screen in [(-13.5, 265., .100, .0149), (-14.5, 245., .070, .010)]:
        for name, fn in [("Koren", lambda a,g,s: electrode("pentode", [a,g,s])), ("Reefman EL34", reefman)]:
            cur, jac = fn(250., vg, vs)
            rows.append(dict(tube="EL34", model=name, va=250., vg=vg, vs=vs,
                             ia_a=float(cur[0]), reference_ia_a=target,
                             ig2_a=float(cur[1]), reference_ig2_a=target_screen))
    return rows


def main():
    if not shutil.which("ngspice"):
        raise RuntimeError("ngspice is required; there is no silent skip")
    DEST.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    reference = author_reference()
    results = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                   ngspice=subprocess.check_output(["ngspice", "--version"], text=True).strip(),
                   source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/"simulation/tube_models.py", ROOT/"simulation/full_mna.py", ROOT/"reference/tube_validation_sources.json"]})
    print("Checking analytic derivatives and broad voltage domains", flush=True)
    tri_points = [(a,g) for a in (1., 20., 100., 250., 300.) for g in (-10., -5., -2., -1., 0., 1., 3.)]
    pen_points = [(a,g,s) for a in (.1, 10., 50., 100., 250., 465., 600.) for g in (-80., -47.68, -30., -13.5, 0., 3.) for s in (50., 200., 265., 360., 465., 550.)]
    results["jacobians"] = {name: check_jacobian(lambda a,g: dempwolf(a,g,p), tri_points) for name,p in TRIODES.items()}
    results["jacobians"]["Reefman"] = check_jacobian(reefman, pen_points)
    results["spice"] = []
    for name,p in TRIODES.items():
        for g in (-5., -2., -1., 0., 1., 3.):
            trace = spice_sweep(f"dz_{name}_{g:g}", triode_reference(p), g)
            expected = np.array([dempwolf(a,g,p)[0][:2] for a in trace[:,0]])
            # Source currents are opposite to currents entering the tube.
            error = float(np.max(np.abs(expected+trace[:,1:])))
            if error > 1e-9:
                raise AssertionError(f"DZ SPICE mismatch: {error}")
            results["spice"].append(dict(model=name, vg=g, points=len(trace), max_difference_a=error))
    for s in (200., 265., 360., 465.):
        for g in (-80., -47.68, -30., -13.5, 0.):
            trace = spice_sweep(f"el34_{s:g}_{g:g}", reference, g, s)
            expected = np.array([reefman(a,g,s)[0][:2] for a in trace[:,0]])
            expected[:,0] += trace[:,0]/1e9 # author's explicit convergence resistor
            error = float(np.max(np.abs(expected+trace[:,1:])))
            if error > 1e-9:
                raise AssertionError(f"EL34 author SPICE mismatch: {error}")
            results["spice"].append(dict(model="Reefman EL34", vs=s, vg=g, points=len(trace), max_difference_a=error))
    print("Independent SPICE sweeps passed", flush=True)
    results["anchors"] = anchors()
    # Evaluate the same terminal voltages, not a claimed new amplifier DC solution.
    old = electrode("pentode", [463.1, -47.678437, 462.753633])[0]
    new = reefman(463.1, -47.678437, 462.753633)[0]
    results["old_idle_voltages_probe"] = dict(va=463.1, vg=-47.678437, vs=462.753633,
                                             koren_ia_a=float(old[0]), koren_ig2_a=float(old[1]),
                                             reefman_ia_a=float(new[0]), reefman_ig2_a=float(new[1]))
    # Screen resistor fixture: solve its current/voltage coupling, rather than
    # force the screen to the measured voltage for both models.
    results["screen_resistor_fixture"] = []
    for name, fn in [("Koren", lambda a,g,s: electrode("pentode", [a,g,s])), ("Reefman", reefman)]:
        s = brentq(lambda s: s+2000*fn(250.,-14.5,s)[0][1]-265., 1., 265., xtol=1e-11)
        cur = fn(250.,-14.5,s)[0]
        results["screen_resistor_fixture"].append(dict(model=name, vs=s, ia_a=float(cur[0]), ig2_a=float(cur[1]),
                                                     kcl_a=float((265.-s)/2000-cur[1])))
    # Domain probes deliberately expose published limitations without repairing them.
    results["triode_low_anode"] = {name: {str(g): dempwolf(0.,g,p)[0].tolist() for g in (-5.,0.,1.,3.)} for name,p in TRIODES.items()}
    currents = np.array([reefman(*v)[0] for v in pen_points])
    results["pentode_domain"] = dict(points=len(pen_points), finite=bool(np.isfinite(currents).all()),
                                      min_ia_a=float(currents[:,0].min()), min_ig2_a=float(currents[:,1].min()),
                                      current_sum_max_error_a=float(np.max(np.abs(currents[:,0]+currents[:,1]-currents[:,2]))))
    results["pentode_domain"]["reverse_anode_probes"] = [dict(va=v[0],vg=v[1],vs=v[2],ia_a=float(c[0]),ig2_a=float(c[1])) for v,c in zip(pen_points,currents) if c[0] < -1e-12]
    # Secondary emission can yield a reverse net anode current. This is a
    # physical validation question, not a reason to clip or assert positivity.
    # Preserve the extrapolation flags even when the implementation agrees.
    if not results["pentode_domain"]["finite"]:
        raise AssertionError("Pentode domain has nonfinite currents")
    results["status"] = "implementation_verified_physical_qualification_incomplete"
    # Reject unimplemented quadrants explicitly.
    for v in [(-1.,-10.,250.), (100.,-10.,0.), (100.,-10.,-1.)]:
        try: reefman(*v)
        except ValueError: pass
        else: raise AssertionError("Unsupported domain was silently accepted")
    render(results)
    (DEST/"metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(dict(spice_max_a=max(r['max_difference_a'] for r in results['spice']), probe=results['old_idle_voltages_probe']),indent=2))
    print("Report:", DEST/"report.md")


def render(r):
    fig, axes = plt.subplots(2, 2, figsize=(12,9), constrained_layout=True)
    volts = np.linspace(0., 600., 301)
    for g in (-40., -30., -20., -10., 0.):
        vals = np.array([reefman(a,g,360.)[0] for a in volts])
        old = np.array([electrode("pentode", [a,g,360.])[0] for a in volts])
        line, = axes[0,0].plot(volts, vals[:,0]*1000, label=f"Vg1={g:g} V")
        axes[0,0].plot(volts, old[:,0]*1000, "--", color=line.get_color(), alpha=.55)
        axes[0,1].plot(volts, vals[:,1]*1000, color=line.get_color(), label=f"Vg1={g:g} V")
        axes[0,1].plot(volts, old[:,1]*1000, "--", color=line.get_color(), alpha=.55)
    axes[0,0].set(title="EL34, Vg2=360 В: Reefman — сплошные, Koren — пунктир", ylabel="Ia, mA", xlabel="Va, V")
    axes[0,1].set(title="Экранный ток EL34 при тех же напряжениях", ylabel="Ig2, mA", xlabel="Va, V")
    grids = np.linspace(-5.,3.,241)
    for name,p in TRIODES.items():
        vals=np.array([dempwolf(100.,g,p)[0] for g in grids])
        axes[1,0].plot(grids,vals[:,0]*1000,label=name)
        axes[1,1].plot(grids,vals[:,1]*1000,label=name)
    axes[1,0].plot(grids,[electrode("triode",[100.,g])[0][0]*1000 for g in grids],"--",label="прежний Koren")
    axes[1,1].plot(grids,[grid_diode(g)[0]*1000 for g in grids],"--",label="прежние R+диод")
    axes[1,0].set(title="Анодный ток ECC83, Va=100 В",xlabel="Vg1, V",ylabel="Ia, mA")
    axes[1,1].set(title="Сеточный ток ECC83, Va=100 В",xlabel="Vg1, V",ylabel="Ig1, mA")
    for ax in axes.flat:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(DEST/"characteristics.png",dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,p in TRIODES.items():
        a=np.linspace(0,100,201)
        axes[0].plot(a,[dempwolf(v,1.,p)[0][0]*1000 for v in a],label=name)
    axes[0].axvspan(0,20,color="red",alpha=.12,label="вне подтверждённой области малого Va")
    axes[0].set(xlabel="Va, V",ylabel="Ia, mA",title="Проверка границы: Vg1=+1 В")
    selected=[x for x in r['anchors'] if x['tube']=='EL34' and x['vs']==265.]
    x=np.arange(2)
    axes[1].bar(x-.25,[100.,14.9],.25,label="Philips")
    for offset,row in zip((0.,.25),selected):
        axes[1].bar(x+offset,[row['ia_a']*1000,row['ig2_a']*1000],.25,label=row['model'])
    axes[1].set(xticks=x,xticklabels=["Ia","Ig2"],ylabel="mA",title="EL34: Va=250 V, Vg1=-13.5 V, Vg2=265 V")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(DEST/"limits_and_anchors.png",dpi=160); plt.close(fig)
    write_report(r)


def write_report(r):
    lines=["# Отдельная проверка подробных ламповых законов", "",
           "Ветка `research/detailed-tube-validation`. Усилитель и старый опыт `tube_validation` не изменены.",
           "Новые токи пока не подключены в общий MNA. Подбора параметров по Philips в этом опыте нет.", "",
           "EL34: рациональная модель Reefman со вторичной эмиссией, исходные коэффициенты авторской TubeLib.",
           "ECC83: три набора измеренных 12AX7 из Dempwolf–Zölzer. Название лампы не делает эти экземпляры Philips 1970.", "",
           "## Численная проверка", "",
           f"Независимый ngspice: {len(r['spice'])} развёрток, {sum(x['points'] for x in r['spice'])} DC-точек; максимальная разность токов {max(x['max_difference_a'] for x in r['spice']):.3e} А.",
           "EL34 сверена с исходными выражениями авторской библиотеки после описанной адаптации синтаксиса.",
           "Учтён отдельный анодный резистор 1 ГОм. Для ECC83 SPICE-выражения записаны отдельно по статье.",
           "Производные проверены центральными разностями; все заданные критерии пройдены.", "",
           "| Модель | Точек якобиана | Максимальная ошибка, А/В |", "|:---|---:|---:|"]
    for name,m in r['jacobians'].items(): lines.append(f"| {name} | {m['points']} | {m['max_abs_a_per_v']:.3e} |")
    lines += ["", "## Табличные точки Philips", "",
              "Проверены по изображениям страницы 2 обоих даташитов. Вторая экранная точка 245 В выведена из 265 В − 2 кОм × 10 мА.",
              "Крутизна для колонки с экранным резистором здесь не сравнивается: условия малого сигнала требуют отдельного определения.", "",
              "| Лампа/модель | Va / Vg1 / Vg2, В | Ia, мА | Philips Ia, мА | Ig2, мА | Philips Ig2, мА |", "|:---|:---|---:|---:|---:|---:|"]
    for x in r['anchors']:
        s="—" if x['vs'] is None else f"{x['vs']:g}"
        ig="—" if 'ig2_a' not in x else f"{x['ig2_a']*1000:.4f}"
        target="—" if 'reference_ig2_a' not in x else f"{x['reference_ig2_a']*1000:g}"
        lines.append(f"| {x['tube']} {x['model']} | {x['va']:g} / {x['vg']:g} / {s} | {x['ia_a']*1000:.4f} | {x['reference_ia_a']*1000:g} | {ig} | {target} |")
    lines += ["", "| ECC83 модель | Va, В | gm, мА/В | Philips gm, мА/В |", "|:---|---:|---:|---:|"]
    for x in r['anchors']:
        if 'gm_s' in x: lines.append(f"| {x['model']} | {x['va']:g} | {x['gm_s']*1000:.4f} | {x['reference_gm_s']*1000:g} |")
    p=r['old_idle_voltages_probe']
    lines += ["", "## Проверка прежнего дефекта экрана", "",
              f"При одинаковых Va={p['va']:g} В, Vg1={p['vg']:.4f} В, Vg2={p['vs']:.4f} В:",
              f"Koren Ia/Ig2 = {p['koren_ia_a']*1000:.4f}/{p['koren_ig2_a']*1000:.4f} мА; Reefman = {p['reefman_ia_a']*1000:.4f}/{p['reefman_ig2_a']*1000:.4f} мА.",
              "Это точечный тест при напряжениях прежнего опыта, а не пересчитанная рабочая точка усилителя.", "",
              "## Экранный резистор как отдельная связанная цепь", "",
              "Va=250 В, Vg1=−14,5 В, экран питается от 265 В через 2 кОм. Решено Vs+R·Ig2=265 В.",
              "Philips: Ia=70 мА, Ig2=10 мА (Vs=245 В).", "",
              "| Модель | Vs, В | Ia, мА | Ig2, мА | Невязка, А |", "|:---|---:|---:|---:|---:|"]
    for x in r['screen_resistor_fixture']: lines.append(f"| {x['model']} | {x['vs']:.4f} | {x['ia_a']*1000:.4f} | {x['ig2_a']*1000:.4f} | {x['kcl_a']:.3e} |")
    lines += ["", "## Ограничения и решение", "",
              "Ненулевой экранный ток и согласие со SPICE устраняют конкретный дефект прежнего закона, но не аттестуют лампу по всему диапазону.",
              "При Va=250 В, Vg1=−80 В, Vg2=465/550 В модель даёт обратный суммарный анодный ток. Вторичная эмиссия в принципе допускает такой знак; без независимых данных эти точки остаются неподтверждённой экстраполяцией, а не доказанной ошибкой формулы.",
              "Положительный сеточный ток EL34 в авторском наборе всё ещё представлен R+полупроводниковым диодом; проверенной модели AB2 пока нет.",
              "Dempwolf–Zölzer при Va<20 В и положительной сетке не воспроизводит спад Ia к нулю. Стенд сохраняет этот отрицательный результат без скрытого ограничения тока.",
              "Reefman не продолжен на отрицательный анод и неположительный экран. Эти области явно отвергаются до выбора обоснованного продолжения для пробных точек Ньютона.",
              "Проверка статическая: ёмкости, прогрев, динамический сеточный заряд и реальные музыкальные атаки этим опытом не аттестованы.",
              "Совпадение с полными измеренными семействами не проверено: исходные сырые uTracer-данные EL34 в скачанном пакете не найдены. Табличные точки не заменяют такие измерения.",
              "Следующий опыт: расширить независимые данные характеристик, проверить область траекторий ламп усилителя; затем выбирать модель и подключать её в MNA.", "",
              "![Характеристики](characteristics.png)", "", "![Границы и точки](limits_and_anchors.png)", "",
              "## Воспроизведение и источники", "",
              "```sh", ".venv/bin/python reference/fetch_tube_validation.py", ".venv/bin/python simulation/run_tube_validation.py", "```", "",
              "Требуются зависимости `requirements.txt`, ngspice; для первой распаковки авторского архива — 7zz/7z.",
              "[Dempwolf–Zölzer, 2011](https://dafx.de/paper-archive/2011/Papers/76_e.pdf),",
              "[Reefman, PDF с датой 14.03.2019](https://www.dos4ever.com/uTracer3/EM4_Theory.pdf),",
              "[авторская библиотека в ExtractModel](https://www.dos4ever.com/uTracer3/uTracer3_pag14.html),",
              "[Philips ECC83](https://www.drtube.com/datasheets/ecc83-philips1970.pdf),",
              "[Philips EL34](https://www.drtube.com/datasheets/el34-philips1969.pdf).", "",
              "URL и SHA-256 сохранены в `reference/tube_validation_sources.json`. Сторонние оригиналы и сырые SPICE-трассы вне Git.",
              "Версии среды, хеши программ, результаты всех сравнений и отрицательные диагностические точки — в `metrics.json`.", ""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
