/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern float (*f_cblas_sdot)(const CBLAS_INT N, const float *X,
                             const CBLAS_INT incX, const float *Y,
                             const CBLAS_INT incY);
float cblas_sdot(const CBLAS_INT N, const float *X, const CBLAS_INT incX,
                 const float *Y, const CBLAS_INT incY) {
  if (*f_cblas_sdot)
    return (f_cblas_sdot)(N, X, incX, Y, incY);
  return 0.0;
}

extern double (*f_cblas_ddot)(const CBLAS_INT N, const double *X,
                              const CBLAS_INT incX, const double *Y,
                              const CBLAS_INT incY);
double cblas_ddot(const CBLAS_INT N, const double *X, const CBLAS_INT incX,
                  const double *Y, const CBLAS_INT incY) {
  if (*f_cblas_ddot)
    return (f_cblas_ddot)(N, X, incX, Y, incY);
  return 0.0;
}

extern float (*f_cblas_sdsdot)(const CBLAS_INT N, const float alpha,
                               const float *X, const CBLAS_INT incX,
                               const float *Y, const CBLAS_INT incY);
float cblas_sdsdot(const CBLAS_INT N, const float alpha, const float *X,
                   const CBLAS_INT incX, const float *Y, const CBLAS_INT incY) {
  if (*f_cblas_sdsdot)
    return (f_cblas_sdsdot)(N, alpha, X, incX, Y, incY);
  return 0.0;
}

extern double (*f_cblas_dsdot)(const CBLAS_INT N, const float *X,
                               const CBLAS_INT incX, const float *Y,
                               const CBLAS_INT incY);
double cblas_dsdot(const CBLAS_INT N, const float *X, const CBLAS_INT incX,
                   const float *Y, const CBLAS_INT incY) {
  if (*f_cblas_dsdot)
    return (f_cblas_dsdot)(N, X, incX, Y, incY);
  return 0.0;
}
