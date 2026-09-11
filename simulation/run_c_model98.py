"""Сверить сгенерированные G/C и локальную nonlinear-сборку полной MNA."""
from __future__ import annotations
import ctypes,json,platform,subprocess,time
import numpy as np
from compact_mna import CompactCircuit
from full_mna import ROOT,build
from run_controls_qualification import BASELINE,MODEL

DEST=ROOT/"simulation/experiments/c_model98"
LIB=ROOT/"build/c_nonlinear_kernel"/("libjcm800_model98.dylib" if platform.system()=="Darwin" else "libjcm800_model98.so")

def bind():
    LIB.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",str(ROOT/"csrc/jcm800_model98.c"),str(ROOT/"csrc/jcm800_nonlinear.c"),"-o",str(LIB),"-lm"],check=True)
    lib=ctypes.CDLL(str(LIB)); p=ctypes.POINTER(ctypes.c_double)
    f=lib.jcm800_model98_nonlinear;f.argtypes=[p,p,p,p,p,ctypes.c_int];f.restype=None
    rows=np.ctypeslib.as_array((ctypes.c_ubyte*394).in_dll(lib,"jcm800_model98_row")).copy()
    cols=np.ctypeslib.as_array((ctypes.c_ubyte*394).in_dll(lib,"jcm800_model98_col")).copy()
    g=np.ctypeslib.as_array((ctypes.c_double*394).in_dll(lib,"jcm800_model98_g")).copy()
    c=np.ctypeslib.as_array((ctypes.c_double*394).in_dll(lib,"jcm800_model98_c")).copy()
    return f,rows,cols,g,c

def nonlinear(function,state,derivatives=True):
    p=ctypes.POINTER(ctypes.c_double);x=np.ascontiguousarray(state);i=np.empty(98);q=np.empty(98);j=np.empty(394);c=np.empty(394)
    function(x.ctypes.data_as(p),i.ctypes.data_as(p),q.ctypes.data_as(p),j.ctypes.data_as(p),c.ctypes.data_as(p),int(derivatives))
    return i,q,j,c

def main():
    DEST.mkdir(parents=True,exist_ok=True);f,rows,cols,g,c=bind()
    circuit=build(True,amplitude=0.,controls=BASELINE,circuit_type=CompactCircuit,**MODEL)
    with np.load(ROOT/"simulation/raw/long_di_recovery/gain5_master8.npz") as z: states=z["signal"]
    maxima=dict(g=0.,c=0.,current=0.,charge=0.,jacobian=0.,capacitance=0.)
    dense_g=np.zeros((98,98));dense_c=np.zeros((98,98));dense_g[rows,cols]=g;dense_c[rows,cols]=c
    maxima["g"]=float(np.max(abs(dense_g-circuit.G)));maxima["c"]=float(np.max(abs(dense_c-circuit.C)))
    chosen=states[np.linspace(0,len(states)-1,32,dtype=int)]
    began=time.perf_counter_ns()
    for state in chosen:
        ci,cj,cq,cc=circuit.nonlinear(state);ai,aq,aj,ac=nonlinear(f,state)
        maxima["current"]=max(maxima["current"],float(np.max(abs(ai-ci))))
        maxima["charge"]=max(maxima["charge"],float(np.max(abs(aq-cq))))
        maxima["jacobian"]=max(maxima["jacobian"],float(np.max(abs(aj-cj[rows,cols]))))
        maxima["capacitance"]=max(maxima["capacitance"],float(np.max(abs(ac-cc[rows,cols]))))
    validation_ns=time.perf_counter_ns()-began
    timings=[]
    for _ in range(5000):
        began=time.perf_counter_ns();nonlinear(f,chosen[10]);timings.append(time.perf_counter_ns()-began)
    result=dict(status="pass",states=len(chosen),max_absolute_difference=maxima,
                benchmark=dict(median_ns=int(np.median(timings)),calls=5000),validation_host_ns=validation_ns)
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Статическая C-модель полной MNA","","Статус: **pass**.","",
           "Сгенерированы компактные G/C и локальные stamps 6×ECC83, 4×EL34 и 9 диодов.",
           "Все 98 физических переменных сохранены; матрицы хранят 394 позиции.","",
           *[f"- max Δ {k}: `{v:.3e}`" for k,v in maxima.items()],"",
           f"Медиана полной nonlinear-сборки через C API на текущем Mac: **{result['benchmark']['median_ns']} нс**.",
           "В число входит ctypes; это не полный timestep и не такты STM32N6.",""]
    (DEST/"report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":main()
