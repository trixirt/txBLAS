/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern CBLAS_INDEX (*f_cblas_isamax)(const CBLAS_INT N, const float *X,
                                     const CBLAS_INT incX);
CBLAS_INDEX cblas_isamax(const CBLAS_INT N, const float *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_isamax)
    return (f_cblas_isamax)(N, X, incX);
  return 0;
}

extern CBLAS_INDEX (*f_cblas_idamax)(const CBLAS_INT N, const double *X,
                                     const CBLAS_INT incX);
CBLAS_INDEX cblas_idamax(const CBLAS_INT N, const double *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_idamax)
    return (f_cblas_idamax)(N, X, incX);
  return 0;
}

extern CBLAS_INDEX (*f_cblas_icamax)(const CBLAS_INT N, const void *X,
                                     const CBLAS_INT incX);
CBLAS_INDEX cblas_icamax(const CBLAS_INT N, const void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_icamax)
    return (f_cblas_icamax)(N, X, incX);
  return 0;
}

extern CBLAS_INDEX (*f_cblas_izamax)(const CBLAS_INT N, const void *X,
                                     const CBLAS_INT incX);
CBLAS_INDEX cblas_izamax(const CBLAS_INT N, const void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_izamax)
    return (f_cblas_izamax)(N, X, incX);
  return 0;
}
