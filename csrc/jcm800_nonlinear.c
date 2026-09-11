#include "jcm800_nonlinear.h"

#include <math.h>
#include <float.h>

typedef struct { double value, sigmoid; } softplus_result;

static inline softplus_result softplus_sigmoid(double x) {
    softplus_result r;
    if (x >= 0.0) {
        const double e = exp(-x);
        r.value = x + log1p(e);
        r.sigmoid = 1.0/(1.0+e);
    } else {
        const double e = exp(x);
        r.value = log1p(e);
        r.sigmoid = e/(1.0+e);
    }
    return r;
}

void jcm800_dempwolf_rsd1_batch(const double *va_vg, size_t count,
                                double *currents, double *jacobian, unsigned flags) {
    const double G=2.242e-3, mu=103.2, gamma=1.26, C=3.40;
    const double Gg=6.177e-4, xi=1.314, Cg=9.901, Ig0=8.025e-8;
    for (size_t n=0; n<count; ++n) {
        const double va=va_vg[2*n], vg=va_vg[2*n+1];
        const softplus_result a=softplus_sigmoid(C*(va/mu+vg));
        const softplus_result g=softplus_sigmoid(Cg*vg);
        const double f=a.value/C, fg=g.value/Cg;
        const double ik=G*pow(f,gamma), ig=Gg*pow(fg,xi)+Ig0;
        currents[3*n]=ik-ig; currents[3*n+1]=ig; currents[3*n+2]=ik;
        if ((flags&JCM800_DERIVATIVES) && jacobian) {
            const double dk=G*gamma*pow(f,gamma-1.0)*a.sigmoid;
            const double dg=Gg*xi*pow(fg,xi-1.0)*g.sigmoid;
            double *j=jacobian+6*n;
            j[0]=dk/mu; j[1]=dk-dg; j[2]=0.0; j[3]=dg; j[4]=dk/mu; j[5]=dk;
        }
    }
}

void jcm800_reefman_el34_batch(const double *va_vg_vs, size_t count,
                               double *currents, double *jacobian, unsigned flags) {
    const double mu=12.50, exponent=1.363, kg1=217.7, kp=50.5, kvb=1282.7;
    const double kg2=1950.2, secondary=.022, ap=.033, w=64., nu=2.91, lam=4.23;
    const double offset=.00408, slope=1.6e-6, knee=.00408, beta=.105, screen_knee=6.09;
    (void)kg1;
    for (size_t n=0; n<count; ++n) {
        const double va=va_vg_vs[3*n], vg=va_vg_vs[3*n+1], vs=va_vg_vs[3*n+2];
        const double root=sqrt(kvb+vs*vs), z=kp*(1.0/mu+vg/root);
        const softplus_result sp=softplus_sigmoid(z);
        const double e=vs*sp.value/kp, ip=pow(e,exponent);
        const double inv=1.0/(1.0+beta*va), vco=vs/lam-nu*vg-w;
        const double t=tanh(-ap*(va-vco));
        const double sec=secondary*va*(1.0+t)/kg2;
        const double fa=offset+slope*va-knee*inv-sec;
        const double fs=(1.0+screen_knee*inv)/kg2+sec;
        currents[3*n]=ip*fa; currents[3*n+1]=ip*fs; currents[3*n+2]=ip*(fa+fs);
        if ((flags&JCM800_DERIVATIVES) && jacobian) {
            const double de[3]={0.0,vs*sp.sigmoid/root,
                sp.value/kp-vs*vs*vg*sp.sigmoid/(root*root*root)};
            double dip[3]; for (int k=0;k<3;++k) dip[k]=exponent*pow(e,exponent-1.0)*de[k];
            const double dinv=-beta*inv*inv, common=secondary/kg2;
            const double one_t2=1.0-t*t;
            const double dsec[3]={common*((1.0+t)-va*ap*one_t2),
                common*va*one_t2*(-ap*nu), common*va*one_t2*(ap/lam)};
            const double dfa[3]={slope-knee*dinv-dsec[0],-dsec[1],-dsec[2]};
            const double dfs[3]={screen_knee/kg2*dinv+dsec[0],dsec[1],dsec[2]};
            double *j=jacobian+9*n;
            for (int k=0;k<3;++k) {
                j[k]=dip[k]*fa+ip*dfa[k];
                j[3+k]=dip[k]*fs+ip*dfs[k];
                j[6+k]=j[k]+j[3+k];
            }
        }
    }
}

int jcm800_dense_solve(double *a, double *b, size_t n) {
    for (size_t k=0; k<n; ++k) {
        size_t pivot=k; double largest=fabs(a[k*n+k]);
        for (size_t i=k+1; i<n; ++i) {
            const double candidate=fabs(a[i*n+k]);
            if (candidate>largest) { largest=candidate; pivot=i; }
        }
        if (largest<=DBL_MIN) return -1;
        if (pivot!=k) {
            for (size_t j=k; j<n; ++j) {
                const double temporary=a[k*n+j]; a[k*n+j]=a[pivot*n+j]; a[pivot*n+j]=temporary;
            }
            const double temporary=b[k]; b[k]=b[pivot]; b[pivot]=temporary;
        }
        const double diagonal=a[k*n+k];
        for (size_t i=k+1; i<n; ++i) {
            const double factor=a[i*n+k]/diagonal; a[i*n+k]=factor;
            for (size_t j=k+1; j<n; ++j) a[i*n+j]-=factor*a[k*n+j];
            b[i]-=factor*b[k];
        }
    }
    for (size_t ii=n; ii-- > 0;) {
        double value=b[ii];
        for (size_t j=ii+1; j<n; ++j) value-=a[ii*n+j]*b[j];
        b[ii]=value/a[ii*n+ii];
    }
    return 0;
}

void jcm800_diode(double v, double isat, double cjo, double tt,
                  double *current, double *conductance, double *charge, double *capacitance) {
    const double vt=8.617087e-5*300.15, z=v/vt;
    if (z<40.0) { const double e=exp(z); *current=isat*expm1(z); *conductance=isat*e/vt; }
    else { const double e=exp(40.0); *current=isat*(e*(1.0+z-40.0)-1.0); *conductance=isat*e/vt; }
    double junction_q,junction_c;
    if (v<.5) { junction_q=2*cjo*(1-sqrt(1-v)); junction_c=cjo/sqrt(1-v); }
    else { const double d=v-.5; junction_q=2*cjo*(1-sqrt(.5))+cjo*sqrt(2.)*(d+.5*d*d); junction_c=cjo*sqrt(2.)*(1+d); }
    *charge=junction_q+tt**current; *capacitance=junction_c+tt**conductance;
}
