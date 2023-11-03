/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern float (*f_cblas_sasum)(const CBLAS_INT N, const float *X,
                              const CBLAS_INT incX);
float cblas_sasum(const CBLAS_INT N, const float *X, const CBLAS_INT incX) {
  if (*f_cblas_sasum)
    return (f_cblas_sasum)(N, X, incX);
  return 0.0;
}

extern double (*f_cblas_dasum)(const CBLAS_INT N, const double *X,
                               const CBLAS_INT incX);
double cblas_dasum(const CBLAS_INT N, const double *X, const CBLAS_INT incX) {
  if (*f_cblas_dasum)
    return (f_cblas_dasum)(N, X, incX);
  return 0.0;
}

extern float (*f_cblas_scasum)(const CBLAS_INT N, const void *X,
                               const CBLAS_INT incX);
float cblas_scasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX) {
  if (*f_cblas_scasum)
    return (f_cblas_scasum)(N, X, incX);
  return 0.0;
}

extern double (*f_cblas_dzasum)(const CBLAS_INT N, const void *X,
                                const CBLAS_INT incX);
double cblas_dzasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX) {
  if (*f_cblas_dzasum)
    return (f_cblas_dzasum)(N, X, incX);
  return 0.0;
}
