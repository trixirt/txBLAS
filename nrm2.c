/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern float (*f_cblas_snrm2)(const CBLAS_INT N, const float *X,
                              const CBLAS_INT incX);
float cblas_snrm2(const CBLAS_INT N, const float *X, const CBLAS_INT incX) {
  if (*f_cblas_snrm2)
    return (f_cblas_snrm2)(N, X, incX);
  return 0.0;
}

extern double (*f_cblas_dnrm2)(const CBLAS_INT N, const double *X,
                               const CBLAS_INT incX);
double cblas_dnrm2(const CBLAS_INT N, const double *X, const CBLAS_INT incX) {
  if (*f_cblas_dnrm2)
    return (f_cblas_dnrm2)(N, X, incX);
  return 0.0;
}

extern float (*f_cblas_scnrm2)(const CBLAS_INT N, const void *X,
                               const CBLAS_INT incX);
float cblas_scnrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX) {
  if (*f_cblas_scnrm2)
    return (f_cblas_scnrm2)(N, X, incX);
  return 0.0;
}

extern double (*f_cblas_dznrm2)(const CBLAS_INT N, const void *X,
                                const CBLAS_INT incX);
double cblas_dznrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX) {
  if (*f_cblas_dznrm2)
    return (f_cblas_dznrm2)(N, X, incX);
  return 0.0;
}
