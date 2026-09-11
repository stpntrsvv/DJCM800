#ifndef JCM800_SPARSE98_H
#define JCM800_SPARSE98_H

#define JCM800_SPARSE98_VALUES 394
/* A is scaled row-major 98x98; inputs are preserved, x receives the solution. */
int jcm800_sparse98_solve(const double *a, const double *b, double *x);
/* values follows matrix_positions in jcm800_sparse98.json. */
int jcm800_sparse98_solve_values(const double *values, const double *b, double *x);

#endif
