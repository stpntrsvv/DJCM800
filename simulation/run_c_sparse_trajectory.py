"""Сравнить generated C sparse solve с полной NumPy MNA на короткой траектории."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from c_sparse_mna import CSparseCircuit
from full_mna import Circuit, ROOT, build
from run_controls_qualification import MODEL
import run_linear_reduction as comparison

DEST=ROOT/"simulation/experiments/c_sparse_trajectory"
RAW=ROOT/"simulation/raw/c_sparse_trajectory"


def render(result):
    lines=["# C sparse solve в полной траектории MNA","",f"Статус: **{result['status']}**.","",
           "Сравниваются исходный NumPy dense solve полной 98×98 MNA и тот же Newton,",
           "в котором только линейная поправка заменена сгенерированным C sparse98.",
           "Физика, float64, начальное состояние, временная сетка, line search и критерии",
           "полной невязки одинаковы.","", "| Режим | Шаг, мкс | max ΔV, В | max ΔI, А | Ошибка выхода | Деления | NumPy/C |",
           "|:---|---:|---:|---:|---:|---:|---:|"]
    for row in result.get("cases",[]):
        lines.append(f"| {row['name']} | {row['h_s']*1e6:g} | {row['max_node_difference_v']:.3e} | "
                     f"{row['max_branch_difference_a']:.3e} | {row['relative_output_error']:.3e} | "
                     f"{row['shared_subdivisions']} | {row['host_speed_ratio']:.3f} |")
    lines += ["","Это интеграционная квалификация solve, а не итоговый C timestep: нелинейности,",
              "stamping, residual и line search здесь ещё исполняются Python-классом Circuit.",""]
    if "error" in result: lines += ["## Ошибка","",result["error"],""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    DEST.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True); comparison.RAW=RAW
    options=dict(full_supply=True,amplitude=.1,**MODEL)
    circuits=(build(circuit_type=Circuit,**options),build(circuit_type=CSparseCircuit,**options))
    with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored:
        initial,start=stored["state"],float(stored["time"])
    result=dict(status="running",unknowns=98,cases=[],solver_source_sha256=hashlib.sha256((ROOT/"csrc/jcm800_sparse98.c").read_bytes()).hexdigest())
    output=DEST/"metrics.json"
    def save(): output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        arrays,row=comparison.paired_segment(circuits,[initial.copy(),initial.copy()],start,.002,1.25e-6,"attack_100mv")
        comparison.assert_errors(row); result["cases"].append(row)
        result["status"]="pass"; save(); render(result)
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}"); save(); render(result); raise


if __name__=="__main__": main()
