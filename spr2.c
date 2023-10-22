/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_spr2(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b,
                     size_t *size_c) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));
  *size_c = ((N * (N + 1)) / 2);

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sspr2)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const float *alpha, const float *x,
                                         rocblas_int incx, const float *y,
                                         rocblas_int incy, float *AP);
rocblas_status rocblas_sspr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const float *alpha, const float *x,
                             rocblas_int incx, const float *y, rocblas_int incy,
                             float *AP) {
  if (*rocblas_sspr2)
    return (f_rocblas_sspr2)(handle, uplo, n, alpha, x, incx, y, incy, AP);
  return -1;
}

extern void (*f_cblas_sspr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const float alpha,
                             const float *X, const CBLAS_INT incX,
                             const float *Y, const CBLAS_INT incY, float *A);

static void _cblas_sspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const float alpha, const float *X,
                         const CBLAS_INT incX, const float *Y,
                         const CBLAS_INT incY, float *A) {
  if (*f_cblas_sspr2)
    _cblas_sspr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}
void cblas_sspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const float alpha, const float *X, const CBLAS_INT incX,
                 const float *Y, const CBLAS_INT incY, float *A) {

  size_t size_a, size_b, size_c;
  int s = size_spr2(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_sspr2(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, incY,
                __C);

  hipMemcpy(A, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sspr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}

extern rocblas_status (*f_rocblas_dspr2)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const double *alpha, const double *x,
                                         rocblas_int incx, const double *y,
                                         rocblas_int incy, double *AP);
rocblas_status rocblas_dspr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const double *alpha,
                             const double *x, rocblas_int incx, const double *y,
                             rocblas_int incy, double *AP) {
  if (*rocblas_dspr2)
    return (f_rocblas_dspr2)(handle, uplo, n, alpha, x, incx, y, incy, AP);
  return -1;
}

extern void (*f_cblas_dspr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const double alpha,
                             const double *X, const CBLAS_INT incX,
                             const double *Y, const CBLAS_INT incY, double *A);

static void _cblas_dspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const double alpha, const double *X,
                         const CBLAS_INT incX, const double *Y,
                         const CBLAS_INT incY, double *A) {
  if (*f_cblas_dspr2)
    _cblas_dspr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}
void cblas_dspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const double alpha, const double *X, const CBLAS_INT incX,
                 const double *Y, const CBLAS_INT incY, double *A) {

  size_t size_a, size_b, size_c;
  int s = size_spr2(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_dspr2(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, incY,
                __C);

  hipMemcpy(A, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dspr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}
