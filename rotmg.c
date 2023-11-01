/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern void (*f_cblas_srotmg)(float *d1, float *d2, float *b1, const float b2,
                              float *P);
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P) {
  if (*f_cblas_srotmg)
    (f_cblas_srotmg)(d1, d2, b1, b2, P);
}

extern void (*f_cblas_drotmg)(double *d1, double *d2, double *b1,
                              const double b2, double *P);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2,
                  double *P) {
  if (*f_cblas_drotmg)
    (f_cblas_drotmg)(d1, d2, b1, b2, P);
}

extern void (*f_cblas_crotmg)(void *d1, void *d2, void *b1, const float b2,
                              void *P);
void cblas_crotmg(void *d1, void *d2, void *b1, const float b2, void *P) {
  if (*f_cblas_crotmg)
    (f_cblas_crotmg)(d1, d2, b1, b2, P);
}

extern void (*f_cblas_zrotmg)(void *d1, void *d2, void *b1, const double b2,
                              void *P);
void cblas_zrotmg(void *d1, void *d2, void *b1, const double b2, void *P) {
  if (*f_cblas_zrotmg)
    (f_cblas_zrotmg)(d1, d2, b1, b2, P);
}
