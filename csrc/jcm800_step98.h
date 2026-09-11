#ifndef JCM800_STEP98_H
#define JCM800_STEP98_H
typedef struct { int iterations; double kcl_a; double voltage_v; } jcm800_step98_stats;
/* t is the end time of the implicit-Euler step; vin is the input source at t. */
int jcm800_step98(const double *previous,double t,double h,double vin,double *next,jcm800_step98_stats *stats);
#endif
