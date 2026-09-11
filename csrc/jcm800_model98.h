#ifndef JCM800_MODEL98_H
#define JCM800_MODEL98_H
#include <stddef.h>
#define JCM800_MODEL98_SIZE 98
#define JCM800_MODEL98_VALUES 394
extern const unsigned char jcm800_model98_row[JCM800_MODEL98_VALUES];
extern const unsigned char jcm800_model98_col[JCM800_MODEL98_VALUES];
extern const double jcm800_model98_g[JCM800_MODEL98_VALUES];
extern const double jcm800_model98_c[JCM800_MODEL98_VALUES];
void jcm800_model98_nonlinear(const double *x,double *current,double *charge,double *jacobian,double *capacitance,int derivatives);
void jcm800_model98_rhs(double time,double vin,double *rhs);
#endif
