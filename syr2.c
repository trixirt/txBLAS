/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_syr2(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, const CBLAS_INT lda, size_t *size_a,
                     size_t *size_b, size_t *size_c) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));
  *size_c = lda * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_ssyr2)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const float *alpha, const float *x,
                                         rocblas_int incx, const float *y,
                                         rocblas_int incy, float *A,
                                         rocblas_int lda);
rocblas_status rocblas_ssyr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const float *alpha, const float *x,
                             rocblas_int incx, const float *y, rocblas_int incy,
                             float *A, rocblas_int lda) {
  if (*f_rocblas_ssyr2)
    return (f_rocblas_ssyr2)(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_ssyr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const float alpha,
                             const float *X, const CBLAS_INT incX,
                             const float *Y, const CBLAS_INT incY, float *A,
                             const CBLAS_INT lda);
static void _cblas_ssyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const float alpha, const float *X,
                         const CBLAS_INT incX, const float *Y,
                         const CBLAS_INT incY, float *A, const CBLAS_INT lda) {
  if (*f_cblas_ssyr2)
    (f_cblas_ssyr2)(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_ssyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const float alpha, const float *X, const CBLAS_INT incX,
                 const float *Y, const CBLAS_INT incY, float *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s = size_syr2(Uplo, N, incX, incY, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_ssyr2(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, incY,
                __C, lda);

  hipMemcpy(A, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssyr2(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

extern rocblas_status (*f_rocblas_dsyr2)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const double *alpha, const double *x,
                                         rocblas_int incx, const double *y,
                                         rocblas_int incy, double *A,
                                         rocblas_int lda);
rocblas_status rocblas_dsyr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const double *alpha,
                             const double *x, rocblas_int incx, const double *y,
                             rocblas_int incy, double *A, rocblas_int lda) {
  if (*f_rocblas_dsyr2)
    return (f_rocblas_dsyr2)(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_dsyr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const double alpha,
                             const double *X, const CBLAS_INT incX,
                             const double *Y, const CBLAS_INT incY, double *A,
                             const CBLAS_INT lda);
static void _cblas_dsyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const double alpha, const double *X,
                         const CBLAS_INT incX, const double *Y,
                         const CBLAS_INT incY, double *A, const CBLAS_INT lda) {
  if (*f_cblas_dsyr2)
    (f_cblas_dsyr2)(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_dsyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const double alpha, const double *X, const CBLAS_INT incX,
                 const double *Y, const CBLAS_INT incY, double *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s = size_syr2(Uplo, N, incX, incY, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_dsyr2(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, incY,
                __C, lda);

  hipMemcpy(A, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsyr2(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}
