/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern void (*f_cblas_srotg)(float *a, float *b, float *c, float *s);
void cblas_srotg(float *a, float *b, float *c, float *s) {
  if (*f_cblas_srotg)
    (f_cblas_srotg)(a, b, c, s);
}

extern void (*f_cblas_drotg)(double *a, double *b, double *c, double *s);
void cblas_drotg(double *a, double *b, double *c, double *s) {
  if (*f_cblas_drotg)
    (f_cblas_drotg)(a, b, c, s);
}

extern void (*f_cblas_crotg)(void *a, void *b, float *c, void *s);
void cblas_crotg(void *a, void *b, float *c, void *s) {
  if (*f_cblas_crotg)
    (f_cblas_crotg)(a, b, c, s);
}

extern void (*f_cblas_zrotg)(void *a, void *b, double *c, void *s);
void cblas_zrotg(void *a, void *b, double *c, void *s) {
  if (*f_cblas_zrotg)
    (f_cblas_zrotg)(a, b, c, s);
}
