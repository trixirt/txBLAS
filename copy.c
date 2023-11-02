/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

extern void (*f_cblas_scopy)(const CBLAS_INT N, const float *X,
                             const CBLAS_INT incX, float *Y,
                             const CBLAS_INT incY);
static void _cblas_scopy(const CBLAS_INT N, const float *X,
                         const CBLAS_INT incX, float *Y, const CBLAS_INT incY) {
  if (*f_cblas_scopy)
    (f_cblas_scopy)(N, X, incX, Y, incY);
}

extern void (*f_cblas_dcopy)(const CBLAS_INT N, const float *X,
                             const CBLAS_INT incX, float *Y,
                             const CBLAS_INT incY);
static void _cblas_dcopy(const CBLAS_INT N, const float *X,
                         const CBLAS_INT incX, float *Y, const CBLAS_INT incY) {
  if (*f_cblas_dcopy)
    (f_cblas_dcopy)(N, X, incX, Y, incY);
}

extern void (*f_cblas_ccopy)(const CBLAS_INT N, const void *X,
                             const CBLAS_INT incX, void *Y,
                             const CBLAS_INT incY);
static void _cblas_ccopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_ccopy)
    (f_cblas_ccopy)(N, X, incX, Y, incY);
}

extern void (*f_cblas_zcopy)(const CBLAS_INT N, const void *X,
                             const CBLAS_INT incX, void *Y,
                             const CBLAS_INT incY);
static void _cblas_zcopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_zcopy)
    (f_cblas_zcopy)(N, X, incX, Y, incY);
}
