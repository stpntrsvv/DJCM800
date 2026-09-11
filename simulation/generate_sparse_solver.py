"""Сгенерировать фиксированный sparse LU 98×98 из принятой структуры MNA."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import MODEL

HEADER=ROOT/"csrc/jcm800_sparse98.h"
SOURCE=ROOT/"csrc/jcm800_sparse98.c"
META=ROOT/"csrc/jcm800_sparse98.json"
INITIAL=ROOT/"simulation/raw/linear_reduction/initial.npz"
N=98


def reference_matrix():
    circuit=build(True,amplitude=.1,circuit_type=CompactCircuit,**MODEL)
    with np.load(INITIAL) as stored: state=stored["state"]
    alpha=1/.625e-6; _,jac,_,cap=circuit.nonlinear(state)
    matrix=circuit.G+alpha*circuit.C+jac+alpha*cap
    scale=np.maximum(np.max(np.abs(matrix),axis=1),1e-15)
    return matrix/scale[:,None]


def structure(matrix):
    factor=splu(csc_matrix(matrix),permc_spec="MMD_AT_PLUS_A")
    rows=np.argsort(factor.perm_r); columns=np.argsort(factor.perm_c)
    original=matrix[rows][:,columns] != 0.; filled=original.copy()
    stages=[]
    for k in range(N):
        below=(np.flatnonzero(filled[k+1:,k])+k+1).tolist()
        right=(np.flatnonzero(filled[k,k+1:])+k+1).tolist()
        stages.append((below,right))
        for i in below: filled[i,right]=True
    positions=[(i,j) for i in range(N) for j in range(N) if filled[i,j]]
    slots={position:index for index,position in enumerate(positions)}
    return rows,columns,original,positions,slots,stages


def generate():
    matrix=reference_matrix(); rows,columns,original,positions,slots,stages=structure(matrix)
    original_positions=[(i,j) for i in range(N) for j in range(N) if matrix[i,j] != 0.]
    original_slots={position:index for index,position in enumerate(original_positions)}
    h=f'''#ifndef JCM800_SPARSE98_H\n#define JCM800_SPARSE98_H\n\n#define JCM800_SPARSE98_VALUES {len(original_positions)}\n/* A is scaled row-major 98x98; inputs are preserved, x receives the solution. */\nint jcm800_sparse98_solve(const double *a, const double *b, double *x);\n/* values follows matrix_positions in jcm800_sparse98.json. */\nint jcm800_sparse98_solve_values(const double *values, const double *b, double *x);\n\n#endif\n'''
    lines=['#include "jcm800_sparse98.h"','#include <float.h>','#include <math.h>','',
           'int jcm800_sparse98_solve_values(const double *values, const double *b, double *x) {',
           f'    double m[{len(positions)}]={{0.0}}, r[{N}];']
    for k,row in enumerate(rows): lines.append(f'    r[{k}]=b[{row}];')
    for i,j in positions:
        if original[i,j]: lines.append(f'    m[{slots[i,j]}]=values[{original_slots[rows[i],columns[j]]}];')
    for k,(below,right) in enumerate(stages):
        pivot=slots[k,k]; lines.append(f'    if (fabs(m[{pivot}])<=DBL_MIN) return {k+1};')
        for i in below:
            ik=slots[i,k]; lines.append(f'    m[{ik}]/=m[{pivot}];')
            for j in right: lines.append(f'    m[{slots[i,j]}]-=m[{ik}]*m[{slots[k,j]}];')
            lines.append(f'    r[{i}]-=m[{ik}]*r[{k}];')
    for i in range(N-1,-1,-1):
        expression=f'r[{i}]'
        for j in stages[i][1]: expression+=f'-m[{slots[i,j]}]*r[{j}]'
        lines.append(f'    r[{i}]=({expression})/m[{slots[i,i]}];')
    for k,column in enumerate(columns): lines.append(f'    x[{column}]=r[{k}];')
    lines += ['    return 0;','}','',
              'int jcm800_sparse98_solve(const double *a, const double *b, double *x) {',
              f'    double values[{len(original_positions)}];']
    for index,(i,j) in enumerate(original_positions): lines.append(f'    values[{index}]=a[{i*N+j}];')
    lines += ['    return jcm800_sparse98_solve_values(values,b,x);','}','']
    HEADER.write_text(h,encoding='utf-8'); SOURCE.write_text('\n'.join(lines),encoding='utf-8')
    meta=dict(size=N,stored_coefficients=len(positions),original_coefficients=int(np.count_nonzero(original)),
              elimination_multiply_subtract_pairs=sum(len(a)*len(b) for a,b in stages),
              elimination_divisions=sum(len(a) for a,_ in stages),
              back_substitution_terms=sum(len(b) for _,b in stages),rows=rows.tolist(),columns=columns.tolist(),
              matrix_positions=original_positions,
              initial_sha256=hashlib.sha256(INITIAL.read_bytes()).hexdigest())
    META.write_text(json.dumps(meta,indent=2)+"\n",encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('rows','columns','matrix_positions')},indent=2))


if __name__=='__main__': generate()
