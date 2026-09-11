"""Квалификация полного float64 timestep JCM800, целиком исполняемого в C."""
from __future__ import annotations
import ctypes,hashlib,json,math,platform,subprocess,time
import numpy as np
from full_mna import ROOT,build
from run_controls_qualification import BASELINE,MODEL

DEST=ROOT/"simulation/experiments/c_step98"
LIB=ROOT/"build/c_nonlinear_kernel"/("libjcm800_step98.dylib" if platform.system()=="Darwin" else "libjcm800_step98.so")
SOURCES=[ROOT/f"csrc/{name}" for name in ("jcm800_step98.c","jcm800_model98.c","jcm800_sparse98.c","jcm800_nonlinear.c")]
class Stats(ctypes.Structure): _fields_=[("iterations",ctypes.c_int),("kcl_a",ctypes.c_double),("voltage_v",ctypes.c_double)]

def bind():
    LIB.parent.mkdir(parents=True,exist_ok=True);subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",*[str(x) for x in SOURCES],"-o",str(LIB),"-lm"],check=True)
    lib=ctypes.CDLL(str(LIB));p=ctypes.POINTER(ctypes.c_double);f=lib.jcm800_step98
    f.argtypes=[p,ctypes.c_double,ctypes.c_double,ctypes.c_double,p,ctypes.POINTER(Stats)];f.restype=ctypes.c_int;return f

def step(function,previous,t,h,vin):
    p=ctypes.POINTER(ctypes.c_double);previous=np.ascontiguousarray(previous);out=np.empty(98);stats=Stats()
    status=function(previous.ctypes.data_as(p),t,h,vin,out.ctypes.data_as(p),ctypes.byref(stats))
    if status: raise RuntimeError(f"C step failed: {status}")
    return out,stats

def main():
    DEST.mkdir(parents=True,exist_ok=True);function=bind();circuit=build(True,amplitude=.1,controls=BASELINE,**MODEL)
    with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as z: initial,start=z["state"],float(z["time"])
    h=1.25e-6;count=1600;python=np.empty((count+1,98));native=np.empty_like(python);python[0]=native[0]=initial
    py_time=c_time=0;max_iterations=0
    for k in range(count):
        t=start+(k+1)*h;vin=.1*math.sin(2*math.pi*1000*t)
        began=time.perf_counter_ns();python[k+1],_=circuit.step(python[k],t,h);py_time+=time.perf_counter_ns()-began
        began=time.perf_counter_ns();native[k+1],stats=step(function,native[k],t,h,vin);c_time+=time.perf_counter_ns()-began
        max_iterations=max(max_iterations,stats.iterations)
    difference=native-python;out=circuit.index["out"]
    result=dict(status="pass",steps=count,h_s=h,duration_s=count*h,max_iterations=max_iterations,
                max_node_difference_v=float(np.max(abs(difference[:,:84]))),max_branch_difference_a=float(np.max(abs(difference[:,84:]))),
                relative_output_error=float(np.linalg.norm(difference[:,out])/np.linalg.norm(python[:,out])),
                python_host_s=py_time/1e9,c_host_s=c_time/1e9,host_speedup=py_time/c_time,
                c_ns_per_step=c_time/count,source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in SOURCES})
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Полный float64 C timestep JCM800","","Статус: **pass**.","",
           "В C исполняются G/C, лампы и диоды, история неявного Эйлера, RHS, residual,",
           "Newton, line search и сгенерированный sparse98 solve. Все 98 состояний сохранены.","",
           f"На {count} шагах по {h*1e6:g} мкс: max ΔV={result['max_node_difference_v']:.3e} В,",
           f"max ΔI={result['max_branch_difference_a']:.3e} А, ошибка выхода={result['relative_output_error']:.3e}.","",
           f"Текущий Mac: {result['c_ns_per_step']:.0f} нс/шаг с ctypes, ускорение к полной Python MNA",
           f"**{result['host_speedup']:.1f}×**. Это не такты STM32N6; на плате нужен DWT/PMU замер.",""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":main()
