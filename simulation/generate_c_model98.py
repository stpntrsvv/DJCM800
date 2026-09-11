"""Сгенерировать компактные G/C и локальные nonlinear stamps полной MNA."""
from __future__ import annotations

import json
import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import BASELINE, MODEL

META=ROOT/"csrc/jcm800_sparse98.json"
HEADER=ROOT/"csrc/jcm800_model98.h"
SOURCE=ROOT/"csrc/jcm800_model98.c"


def terms(vector): return [(int(i),float(vector[i])) for i in np.flatnonzero(vector)]
def number(value): return f"{float(value):.17g}"


def generate():
    circuit=build(True,amplitude=0.,controls=BASELINE,circuit_type=CompactCircuit,**MODEL)
    meta=json.loads(META.read_text()); positions=[tuple(x) for x in meta["matrix_positions"]]
    slot={position:i for i,position in enumerate(positions)}; count=len(positions)
    g=[circuit.G[i,j] for i,j in positions]; c=[circuit.C[i,j] for i,j in positions]
    header=f'''#ifndef JCM800_MODEL98_H\n#define JCM800_MODEL98_H\n#include <stddef.h>\n#define JCM800_MODEL98_SIZE 98\n#define JCM800_MODEL98_VALUES {count}\nextern const unsigned char jcm800_model98_row[JCM800_MODEL98_VALUES];\nextern const unsigned char jcm800_model98_col[JCM800_MODEL98_VALUES];\nextern const double jcm800_model98_g[JCM800_MODEL98_VALUES];\nextern const double jcm800_model98_c[JCM800_MODEL98_VALUES];\nvoid jcm800_model98_nonlinear(const double *x,double *current,double *charge,double *jacobian,double *capacitance,int derivatives);\nvoid jcm800_model98_rhs(double time,double vin,double *rhs);\n#endif\n'''
    lines=['#include "jcm800_model98.h"','#include "jcm800_nonlinear.h"','#include <math.h>','#include <string.h>','',
           f'const unsigned char jcm800_model98_row[{count}]={{'+','.join(str(i) for i,_ in positions)+'};',
           f'const unsigned char jcm800_model98_col[{count}]={{'+','.join(str(j) for _,j in positions)+'};',
           f'const double jcm800_model98_g[{count}]={{'+','.join(number(x) for x in g)+'};',
           f'const double jcm800_model98_c[{count}]={{'+','.join(number(x) for x in c)+'};','',
           'void jcm800_model98_nonlinear(const double *x,double *current,double *charge,double *jacobian,double *capacitance,int derivatives) {',
           '  memset(current,0,98*sizeof(double)); memset(charge,0,98*sizeof(double));',
           f'  if (derivatives) {{ memset(jacobian,0,{count}*sizeof(double)); memset(capacitance,0,{count}*sizeof(double)); }}']
    tube_number=diode_number=0
    for kind,control,output,parameters in circuit.nl:
        if kind=='T':
            tube_number+=1; controls=np.atleast_2d(control); outputs=np.atleast_2d(output)
            width=controls.shape[0]; lines.append(f'  {{ double u[3]={{0}}, y[3], d[9];')
            for k,row in enumerate(controls):
                expression='+'.join(f'({number(v)}*x[{i}])' for i,v in terms(row)) or '0.0'
                lines.append(f'    u[{k}]={expression};')
            function='jcm800_dempwolf_rsd1_batch' if str(parameters).startswith('dempwolf:') else 'jcm800_reefman_el34_batch'
            lines.append(f'    {function}(u,1,y,d,derivatives?JCM800_DERIVATIVES:0);')
            for k,row in enumerate(outputs):
                for i,v in terms(row): lines.append(f'    current[{i}]+={number(v)}*y[{k}];')
            lines.append('    if (derivatives) {')
            for a,outrow in enumerate(outputs):
                for i,ov in terms(outrow):
                    for b,inrow in enumerate(controls):
                        for j,iv in terms(inrow):
                            lines.append(f'      jacobian[{slot[i,j]}]+={number(ov*iv)}*d[{a*width+b}];')
            lines += ['    }','  }']
        else:
            diode_number+=1; inc=np.asarray(control); support=terms(inc); iss,cjo,tt=parameters
            expression='+'.join(f'({number(v)}*x[{i}])' for i,v in support)
            lines.append(f'  {{ double cur,cond,q,cap; jcm800_diode({expression},{number(iss)},{number(cjo)},{number(tt)},&cur,&cond,&q,&cap);')
            for i,v in support:
                lines.append(f'    current[{i}]+={number(v)}*cur; charge[{i}]+={number(v)}*q;')
            lines.append('    if (derivatives) {')
            for i,iv in support:
                for j,jv in support:
                    lines.append(f'      jacobian[{slot[i,j]}]+={number(iv*jv)}*cond; capacitance[{slot[i,j]}]+={number(iv*jv)}*cap;')
            lines += ['    }','  }']
    lines += ['}','', 'void jcm800_model98_rhs(double time,double vin,double *rhs) {',
              '  memset(rhs,0,98*sizeof(double)); rhs[87]=vin; rhs[88]=325.2691193458119*sin(314.15926535897932*time);','}','']
    HEADER.write_text(header,encoding='utf-8'); SOURCE.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(matrix_values=count,tubes=tube_number,diodes=diode_number,source_lines=len(lines)),indent=2))


if __name__=='__main__': generate()
