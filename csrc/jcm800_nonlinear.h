#ifndef JCM800_NONLINEAR_H
#define JCM800_NONLINEAR_H

#include <stddef.h>

enum { JCM800_DERIVATIVES = 1 };

/* Interleaved inputs/outputs. Jacobian may be NULL in currents-only mode. */
void jcm800_dempwolf_rsd1_batch(const double *vg_va, size_t count,
                                double *currents, double *jacobian, unsigned flags);
void jcm800_reefman_el34_batch(const double *va_vg_vs, size_t count,
                              double *currents, double *jacobian, unsigned flags);

#endif
