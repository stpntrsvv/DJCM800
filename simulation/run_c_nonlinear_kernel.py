"""Сверить и измерить float64 C-ядро неизменённых ламповых законов."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from full_mna import ROOT
from tube_models import EL34, TRIODES, dempwolf, reefman

DEST=ROOT/"simulation/experiments/c_nonlinear_kernel"
BUILD=ROOT/"build/c_nonlinear_kernel"
SOURCE=ROOT/"csrc/jcm800_nonlinear.c"
HEADER=ROOT/"csrc/jcm800_nonlinear.h"


def compile_library():
    BUILD.mkdir(parents=True,exist_ok=True)
    library=BUILD/("libjcm800_nonlinear.dylib" if platform.system()=="Darwin" else "libjcm800_nonlinear.so")
    command=["cc","-O3","-std=c11","-fPIC","-shared",str(SOURCE),"-o",str(library),"-lm"]
    subprocess.run(command,check=True,capture_output=True,text=True)
    return library,command


def bind(path):
    library=ctypes.CDLL(str(path)); pointer=ctypes.POINTER(ctypes.c_double)
    for name in ("jcm800_dempwolf_rsd1_batch","jcm800_reefman_el34_batch"):
        function=getattr(library,name)
        function.argtypes=[pointer,ctypes.c_size_t,pointer,pointer,ctypes.c_uint]
        function.restype=None
    return library


def call(function, inputs, outputs_per_row, jacobian_per_row, derivatives):
    inputs=np.ascontiguousarray(inputs,dtype=np.float64); count=len(inputs)
    outputs=np.empty((count,outputs_per_row)); jac=np.empty((count,jacobian_per_row)) if derivatives else None
    pointer=ctypes.POINTER(ctypes.c_double)
    function(inputs.ctypes.data_as(pointer),count,outputs.ctypes.data_as(pointer),
             jac.ctypes.data_as(pointer) if jac is not None else None,1 if derivatives else 0)
    return outputs,jac


def reference(inputs,kind):
    function,parameters=(dempwolf,TRIODES["RSD-1"]) if kind=="ecc83" else (reefman,EL34)
    values=[function(*row,parameters) for row in inputs]
    return np.asarray([x[0] for x in values]),np.asarray([x[1] for x in values])


def benchmark(function,inputs,out_count,jac_count,derivatives,repeats):
    samples=[]
    for _ in range(repeats):
        began=time.perf_counter_ns(); call(function,inputs,out_count,jac_count,derivatives)
        samples.append(time.perf_counter_ns()-began)
    return dict(median_ns=int(np.median(samples)),ns_per_device=float(np.median(samples)/len(inputs)),samples_ns=samples)


def benchmark_python(inputs,kind,repeats):
    samples=[]
    for _ in range(repeats):
        began=time.perf_counter_ns(); reference(inputs,kind); samples.append(time.perf_counter_ns()-began)
    return dict(median_ns=int(np.median(samples)),ns_per_device=float(np.median(samples)/len(inputs)),samples_ns=samples)


def render(result):
    lines=["# Float64 C-ядро нелинейностей JCM800","",f"Статус: **{result['status']}**.","",
           "В C перенесены без смены параметров законы Dempwolf RSD-1 и Reefman EL34.",
           "Общий примитив softplus+sigmoid использует один exp вместо двух; режим токов",
           "не вычисляет производные, которые не нужны при проверке line search и истории.","",
           "## Численная сверка","", "| Закон | max abs ток, А | max abs Jacobian |", "|:---|---:|---:|"]
    for name,row in result.get("validation",{}).items():
        lines.append(f"| {name} | {row['current_max_abs']:.3e} | {row['jacobian_max_abs']:.3e} |")
    lines += ["","## Пакетный benchmark на текущем ПК","",
              "| Закон | Режим | нс/лампу |", "|:---|:---|---:|"]
    for name,row in result.get("benchmark",{}).items():
        for mode,value in row.items():
            if isinstance(value,dict) and "ns_per_device" in value:
                lines.append(f"| {name} | {mode} | {value['ns_per_device']:.2f} |")
    if result.get("model_budget"):
        b=result["model_budget"]
        lines += ["", "## Бюджет одного вызова нелинейностей", "",
                  f"В принятой схеме 6 секций ECC83 и 4 EL34. По медианам выше их C-вызов",
                  f"занимает на этом CPU около **{b['c_currents_only_ns']:.0f} нс** только для токов и",
                  f"**{b['c_currents_and_jacobian_ns']:.0f} нс** с Jacobian. Это не полный шаг Ньютона:",
                  "сюда не входят диоды, stamping, невязка, Schur/LU и line search.", ""]
    lines += ["","Это время скомпилированного C на текущем CPU, не такты STM32N6. Для оценки",
              "микроконтроллера API отделяет число вызовов ламп от линейной алгебры; реальные",
              "такты получаются аппаратным DWT/PMU замером того же ядра после кросс-компиляции.","",
              "Статически на одну лампу: ECC83 — 2 exp, 2 log1p и 2 pow; EL34 — exp,",
              "log1p, pow, sqrt и tanh. В currents-only исключаются степени и операции, нужные",
              "только производным, но физические токи вычисляются теми же формулами.",""]
    if "error" in result: lines += ["## Ошибка","",result["error"],""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--quick",action="store_true"); args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True); output=DEST/("metrics_quick.json" if args.quick else "metrics.json")
    result=dict(status="running",compiler="cc -O3 -std=c11 -fPIC -shared",validation={},benchmark={})
    try:
        path,command=compile_library(); library=bind(path); result["compile_command"]=command
        rng=np.random.default_rng(800); count=100 if args.quick else 10_000
        inputs={"ecc83":np.c_[rng.uniform(20,300,count),rng.uniform(-5,3,count)],
                "el34":np.c_[rng.uniform(0,800,count),rng.uniform(-150,2,count),rng.uniform(250,500,count)]}
        specs={"ecc83":(library.jcm800_dempwolf_rsd1_batch,3,6),
               "el34":(library.jcm800_reefman_el34_batch,3,9)}
        for name,x in inputs.items():
            function,out_count,jac_count=specs[name]; actual,jac=call(function,x,out_count,jac_count,True)
            expected,expected_jac=reference(x,name)
            result["validation"][name]=dict(
                current_max_abs=float(np.max(np.abs(actual-expected))),
                current_max_relative=float(np.max(np.abs(actual-expected)/np.maximum(np.abs(expected),1e-15))),
                jacobian_max_abs=float(np.max(np.abs(jac.reshape(expected_jac.shape)-expected_jac))),
                jacobian_max_relative=float(np.max(np.abs(jac.reshape(expected_jac.shape)-expected_jac)/np.maximum(np.abs(expected_jac),1e-15))))
            bench_count=1_000 if args.quick else 200_000; tiled=np.resize(x,(bench_count,x.shape[1]))
            result["benchmark"][name]={
                "currents_only":benchmark(function,tiled,out_count,jac_count,False,3 if args.quick else 9),
                "currents_and_jacobian":benchmark(function,tiled,out_count,jac_count,True,3 if args.quick else 9),
                "python_reference":benchmark_python(x[:min(len(x),100 if args.quick else 2000)],name,1 if args.quick else 3)}
            cjac=result["benchmark"][name]["currents_and_jacobian"]["ns_per_device"]
            py=result["benchmark"][name]["python_reference"]["ns_per_device"]
            result["benchmark"][name]["python_to_c_batch_speedup"]=py/cjac
        eb=result["benchmark"]["ecc83"]; pb=result["benchmark"]["el34"]
        result["model_budget"]={
            "ecc83_sections":6,"el34_tubes":4,
            "c_currents_only_ns":6*eb["currents_only"]["ns_per_device"]+4*pb["currents_only"]["ns_per_device"],
            "c_currents_and_jacobian_ns":6*eb["currents_and_jacobian"]["ns_per_device"]+4*pb["currents_and_jacobian"]["ns_per_device"]}
        result.update(status="pass",source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,HEADER)})
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8"); render(result)


if __name__=="__main__": main()
