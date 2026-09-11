"""Проверить и измерить сгенерированный float64 sparse LU 98×98."""
from __future__ import annotations

import ctypes
import json
import platform
import subprocess
import time

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import BASELINE, MODEL

DEST=ROOT/"simulation/experiments/generated_sparse_solver"
SOURCE=ROOT/"csrc/jcm800_sparse98.c"
LIBRARY=ROOT/"build/c_nonlinear_kernel"/("libjcm800_sparse98.dylib" if platform.system()=="Darwin" else "libjcm800_sparse98.so")


def compile_and_bind():
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",str(SOURCE),"-o",str(LIBRARY),"-lm"],check=True)
    library=ctypes.CDLL(str(LIBRARY)); pointer=ctypes.POINTER(ctypes.c_double)
    function=library.jcm800_sparse98_solve; function.argtypes=[pointer,pointer,pointer]; function.restype=ctypes.c_int
    return function


def solve(function,matrix,rhs):
    pointer=ctypes.POINTER(ctypes.c_double); matrix=np.ascontiguousarray(matrix); rhs=np.ascontiguousarray(rhs); out=np.empty(len(rhs))
    status=function(matrix.ctypes.data_as(pointer),rhs.ctypes.data_as(pointer),out.ctypes.data_as(pointer))
    if status: raise np.linalg.LinAlgError(f"static pivot failed at stage {status-1}")
    return out


def jacobian(circuit,state,h):
    alpha=1/h; _,j,_,cap=circuit.nonlinear(state); matrix=circuit.G+alpha*circuit.C+j+alpha*cap
    scale=np.maximum(np.max(np.abs(matrix),axis=1),1e-15)
    return matrix/scale[:,None],scale


def render(result):
    lines=["# Сгенерированный sparse LU полной MNA","",f"Статус: **{result['status']}**.","",
           "Сохраняются все 98 переменных и исходный float64 Jacobian. Фиксированные строковая",
           "и столбцовая перестановки получены на принятой рабочей точке; C-код хранит 611",
           "коэффициентов и выполняет только символически ненулевые операции.","",
           f"Проверено Jacobian: **{result['validation']['matrices']}**. Отказов pivot: **{result['validation']['pivot_failures']}**.",
           f"Худшая относительная невязка: **{result['validation']['max_relative_residual']:.3e}**;",
           f"max разность решения с NumPy: **{result['validation']['max_solution_difference']:.3e}**.","",
           "## Производительность текущего ПК","",
           f"Медиана sparse solve 98×98: **{result['benchmark']['median_ns']} нс**.",
           f"Для сравнения сохранённый dense LU 39×39 занимал около 10000 нс. Это benchmark",
           "текущего CPU, не такты STM32N6; копирование/stamping и Newton сюда не входят.","",
           "Статический solve использует 837 пар multiply-subtract, 246 делений исключения",
           "и 267 членов обратного хода вместо 19019 пар dense Schur. Перед принятием нужны",
           "полная C-траектория и проверка перестановки на всех шагах, а не только выборке.",""]
    if "error" in result: lines += ["## Ошибка","",result["error"],""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    DEST.mkdir(parents=True,exist_ok=True); function=compile_and_bind()
    circuit=build(True,amplitude=0.,controls=BASELINE,circuit_type=CompactCircuit,**MODEL)
    samples=[]
    with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored: samples.append((stored["state"],.625e-6))
    candidates=(("gain5_master5","signal",1.25e-6),("gain5_master8","signal",1.25e-6),
                ("gain8_master5","signal",1.25e-6),("gain5_master8","recovery",20e-6))
    for name,key,h in candidates:
        with np.load(ROOT/f"simulation/raw/long_di_recovery/{name}.npz") as stored:
            values=stored[key]
            samples.extend((values[i],h) for i in np.linspace(0,len(values)-1,12,dtype=int))
    maximum_residual=maximum_difference=0.; failures=0
    for number,(state,h) in enumerate(samples):
        matrix,scale=jacobian(circuit,state,h); rhs=np.sin(np.arange(98)+.25+number)/scale
        try: actual=solve(function,matrix,rhs)
        except np.linalg.LinAlgError: failures+=1; continue
        expected=np.linalg.solve(matrix,rhs)
        maximum_residual=max(maximum_residual,float(np.linalg.norm(matrix@actual-rhs)/np.linalg.norm(rhs)))
        maximum_difference=max(maximum_difference,float(np.max(np.abs(actual-expected))))
    matrix,scale=jacobian(circuit,samples[0][0],samples[0][1]); rhs=np.sin(np.arange(98)+.25)/scale
    timings=[]
    for _ in range(5000):
        began=time.perf_counter_ns(); solve(function,matrix,rhs); timings.append(time.perf_counter_ns()-began)
    result=dict(status="pass" if failures==0 else "failed",
                validation=dict(matrices=len(samples),pivot_failures=failures,max_relative_residual=maximum_residual,
                                max_solution_difference=maximum_difference),
                benchmark=dict(median_ns=int(np.median(timings)),samples=5000))
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    render(result)


if __name__=="__main__": main()
