"""Финальная сводка уже рассчитанных результатов закрытия лампового блока."""
import json
from pathlib import Path
import numpy as np
from full_mna import ROOT

DEST = ROOT / "simulation/experiments/tube_closure"
RAW = ROOT / "simulation/raw/tube_closure/refinement"
refinement = json.loads((DEST / "refinement_metrics.json").read_text())
closure = json.loads((DEST / "metrics.json").read_text())

caps = {}
for mv, h_us in ((100, 1.25), (500, .625)):
    author = np.load(RAW / f"author_{mv}mv_{h_us:g}us.npz")["output"]
    legacy = np.load(RAW / f"legacy_caps_{mv}mv_{h_us:g}us.npz")["output"]
    direct = float(np.linalg.norm(legacy-author)/np.linalg.norm(author))
    best = None
    for lag in range(round(-100/h_us), round(100/h_us)+1):
        a = author[max(lag, 0):len(author)+min(lag, 0)]
        b = legacy[max(-lag, 0):len(legacy)-max(lag, 0)]
        error = float(np.linalg.norm(b-a)/np.linalg.norm(a))
        if best is None or error < best[0]:
            best = error, lag
    caps[str(mv)] = dict(h_s=h_us*1e-6, direct_relative=direct,
                         best_delay_s=best[1]*h_us*1e-6, delay_aligned_relative=best[0])

grid_power = {}
for mv in (.1, .5):
    powers = {}
    for name in ("ig1_r1k", "author", "ig1_r4k"):
        case = next(x for x in closure["variants"][name]["cases"]
                    if x["amplitude_v"] == mv and x["h_s"] == 2.5e-6)
        powers[name] = case["load_power_w"]
    grid_power[str(mv)] = dict(values_w=powers,
                               span_relative=(max(powers.values())-min(powers.values()))/powers["author"])

result = dict(status="accepted_as_engineering_macromodel_with_documented_physical_limitations",
              convergence=refinement["cases"], capacitance_sensitivity=caps,
              ig1_power_sensitivity=grid_power,
              recovery=closure["variants"]["author"]["recovery"])
(DEST / "final_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")

lines = ["# Итог закрытия блока ламп", "",
         "Численная реализация Dempwolf RSD-1 + Reefman принята как текущая инженерная",
         "макромодель. Это не аттестация конкретных ламп Marshall и не измеренная модель Ig1 EL34.", "",
         "## Сходимость формы", "",
         "| Вход | Принятый шаг | Изменение к предыдущей сетке | Деления шага |",
         "|---:|---:|---:|---:|"]
for mv in ("100", "500"):
    row = refinement["cases"]["author"]["amplitudes"][mv][-1]
    lines.append(f"| {mv} мВ | {row['h_s']*1e6:g} мкс | {row['relative_waveform']*100:.3f}% | {row['halvings']} |")
lines += ["", "## Чувствительность", "",
          f"Изменение RGI 1001…4001 Ом меняет мощность на {grid_power['0.1']['span_relative']*100:.3f}% при 100 мВ и {grid_power['0.5']['span_relative']*100:.3f}% при 500 мВ.",
          f"Паспортные и старые ёмкости дают прямую разность формы {caps['100']['direct_relative']*100:.3f}% / {caps['500']['direct_relative']*100:.3f}%.",
          f"Она преимущественно является задержкой {abs(caps['100']['best_delay_s'])*1e6:.2f} / {abs(caps['500']['best_delay_s'])*1e6:.2f} мкс; после выравнивания остаётся {caps['100']['delay_aligned_relative']*100:.3f}% / {caps['500']['delay_aligned_relative']*100:.3f}%.", "",
          "## Восстановление", "",
          f"После 500 мс последний выход {result['recovery']['last_output_v']*1000:.3f} мВ, RMS последних 20 мс {result['recovery']['tail_rms_v']*1000:.3f} мВ.", "",
          "## Вердикт", "",
          "Ошибки переноса RGI и ёмкостей исправлены; временная сетка сильного сигнала",
          "сошлась до заранее заданного порога 1%. Макромодель пригодна как текущий эталон",
          "для следующего этапа сокращения. Остаются документированные физические ограничения:",
          "Ig1 EL34 — авторская Shockley-ветвь без независимого измеренного семейства, а",
          "сильный сигнал выходит за диапазоны исходной проверки статических законов.", ""]
(DEST / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
print(DEST / "final_report.md")
