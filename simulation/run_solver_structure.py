"""Инвентаризация разреженности перед генерацией фиксированного C-решателя."""
from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import MODEL

DEST=ROOT/"simulation/experiments/solver_structure"


def matrix_at(circuit,state,h):
    alpha=1/h; _,jac,_,cap=circuit.nonlinear(state)
    return circuit.G+alpha*circuit.C+jac+alpha*cap


def factor_metrics(matrix,ordering):
    factor=splu(csc_matrix(matrix),permc_spec=ordering)
    rhs=np.sin(np.arange(len(matrix))+.25); solution=factor.solve(rhs)
    return dict(l_nnz=int(factor.L.nnz),u_nnz=int(factor.U.nnz),total_nnz=int(factor.L.nnz+factor.U.nnz),
                relative_residual=float(np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs)),
                dynamic_row_pivots=int(np.count_nonzero(factor.perm_r != np.arange(len(matrix)))))


def render(result):
    lines=["# Структура линейного solve JCM800","",f"Статус: **{result['status']}**.","",
           "Физика и 98 состояний не сокращаются. Сравнивается цена текущего dense LU",
           "для Schur-системы 39×39 с разреженной факторизацией полного Jacobian 98×98.","",
           f"Полный Jacobian содержит **{result['full']['nnz']}** ненулевых коэффициентов из",
           f"{result['full']['size']**2}; Schur 39×39 — **{result['schur']['nnz']}**.","",
           "| Порядок полного LU | nnz L+U | Переставленных строк | Относительная невязка |", "|:---|---:|---:|---:|"]
    for name,row in result["orderings"].items():
        lines.append(f"| {name} | {row['total_nnz']} | {row['dynamic_row_pivots']} | {row['relative_residual']:.3e} |")
    lines += ["",f"Dense Schur выполняет {result['dense_schur']['multiply_subtract_pairs']} пар",
              "multiply-subtract только в исключении. Лучший найденный разреженный порядок имеет",
              f"{result['best']['factor_nnz']} коэффициентов L+U с pivoting; это структурный кандидат, а не",
              "готовая оценка тактов.","",
              "Следующий шаг: зафиксировать перестановку, сгенерировать список ненулевых операций",
              "C LU и проверить отсутствие pivot breakdown на коротких эталонных траекториях.",
              "Лишь после этого сравнивать реальные DWT/PMU-такты на STM32N6.",""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    circuit=build(True,amplitude=.1,circuit_type=CompactCircuit,**MODEL)
    with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored: state=stored["state"]
    h=.625e-6; full_unscaled=matrix_at(circuit,state,h)
    row_scale=np.maximum(np.max(np.abs(full_unscaled),axis=1),1e-15)
    full=full_unscaled/row_scale[:,None]
    alpha=1/h
    _,jac,_,cap=circuit.compact_nonlinear(state[circuit.retained])
    schur=circuit._blocks(circuit.G+alpha*circuit.C,alpha)[0]+jac+alpha*cap
    orderings={name:factor_metrics(full,name) for name in ("NATURAL","MMD_ATA","MMD_AT_PLUS_A","COLAMD")}
    best_name=min(orderings,key=lambda name:orderings[name]["total_nnz"])
    n=len(schur)
    result=dict(status="pass",h_s=h,full=dict(size=len(full),nnz=int(np.count_nonzero(full))),
                schur=dict(size=n,nnz=int(np.count_nonzero(schur))),orderings=orderings,
                dense_schur=dict(multiply_subtract_pairs=int(n*(n-1)*(2*n-1)/6)),
                best=dict(ordering=best_name,factor_nnz=orderings[best_name]["total_nnz"]))
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    render(result)


if __name__=="__main__": main()
