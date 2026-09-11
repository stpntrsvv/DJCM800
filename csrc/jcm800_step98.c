#include "jcm800_step98.h"
#include "jcm800_model98.h"
#include "jcm800_sparse98.h"
#include <math.h>
#include <string.h>

static void sparse_mv(const double *values,const double *x,double *out) {
    memset(out,0,98*sizeof(double));
    for (int k=0;k<JCM800_MODEL98_VALUES;++k) out[jcm800_model98_row[k]]+=values[k]*x[jcm800_model98_col[k]];
}
static double residual_norm(const double *r,double *kcl,double *voltage) {
    double norm=0.;*kcl=0.;*voltage=0.;
    for(int i=0;i<98;++i){double a=fabs(r[i]),tol=i<84?1e-10:1e-7;if(a/tol>norm)norm=a/tol;if(i<84){if(a>*kcl)*kcl=a;}else if(a>*voltage)*voltage=a;}
    return norm;
}
static void residual(const double *x,const double *current,const double *charge,const double *history,const double *rhs,double alpha,double *out) {
    double linear[98];
    memset(linear,0,sizeof(linear));
    for(int k=0;k<JCM800_MODEL98_VALUES;++k){int r=jcm800_model98_row[k],c=jcm800_model98_col[k];linear[r]+=(jcm800_model98_g[k]+alpha*jcm800_model98_c[k])*x[c];}
    for(int i=0;i<98;++i)out[i]=linear[i]+current[i]+alpha*charge[i]+history[i]-rhs[i];
}
int jcm800_step98(const double *previous,double t,double h,double vin,double *next,jcm800_step98_stats *stats) {
    double current[98],charge[98],jac[394],cap[394],history[98],rhs[98],r[98],delta[98],candidate[98];
    double cv[98]; const double alpha=1./h;
    jcm800_model98_nonlinear(previous,current,charge,jac,cap,0);
    sparse_mv(jcm800_model98_c,previous,cv);
    for(int i=0;i<98;++i)history[i]=-(cv[i]+charge[i])*alpha;
    jcm800_model98_rhs(t,vin,rhs);memcpy(next,previous,98*sizeof(double));
    for(int iteration=0;iteration<100;++iteration){
        jcm800_model98_nonlinear(next,current,charge,jac,cap,1);residual(next,current,charge,history,rhs,alpha,r);
        double kcl,voltage,norm=residual_norm(r,&kcl,&voltage);
        if(norm<=1.){if(stats){stats->iterations=iteration;stats->kcl_a=kcl;stats->voltage_v=voltage;}return 0;}
        double values[394],scale[98]={0},scaled_rhs[98];
        for(int k=0;k<394;++k){values[k]=jcm800_model98_g[k]+alpha*jcm800_model98_c[k]+jac[k]+alpha*cap[k];int row=jcm800_model98_row[k];double a=fabs(values[k]);if(a>scale[row])scale[row]=a;}
        for(int i=0;i<98;++i){if(scale[i]<1e-15)scale[i]=1e-15;scaled_rhs[i]=-r[i]/scale[i];}
        for(int k=0;k<394;++k)values[k]/=scale[jcm800_model98_row[k]];
        int status=jcm800_sparse98_solve_values(values,scaled_rhs,delta);if(status)return 100+status;
        int accepted=0;
        for(int power=0;power<28;++power){double factor=ldexp(1.,-power);for(int i=0;i<98;++i)candidate[i]=next[i]+factor*delta[i];
            jcm800_model98_nonlinear(candidate,current,charge,jac,cap,0);residual(candidate,current,charge,history,rhs,alpha,r);
            double ck,cvv,cn=residual_norm(r,&ck,&cvv);if(cn<norm||cn<=1.){memcpy(next,candidate,sizeof(candidate));accepted=1;break;}}
        if(!accepted)return 2;
    }
    return 1;
}
