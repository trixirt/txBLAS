/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_sbmv(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT K,
                     const CBLAS_INT lda, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b,
                     size_t *size_c) {

  *size_a = lda * N;
  *size_b = (1 + (N - 1) * abs(incX));
  *size_c = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_ssbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
    const float *alpha, const float *A, rocblas_int lda, const float *x,
    rocblas_int incx, const float *beta, float *y, rocblas_int incy);

rocblas_status rocblas_ssbmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, rocblas_int k, const float *alpha,
                             const float *A, rocblas_int lda, const float *x,
                             rocblas_int incx, const float *beta, float *y,
                             rocblas_int incy) {
  if (f_rocblas_ssbmv)
    return (f_rocblas_ssbmv)(handle, uplo, n, k, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_ssbmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const float alpha, const float *A,
                             const CBLAS_INT lda, const float *X,
                             const CBLAS_INT incX, const float beta, float *Y,
                             const CBLAS_INT incY);
static void _cblas_ssbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         const float *X, const CBLAS_INT incX, const float beta,
                         float *Y, const CBLAS_INT incY) {
  if (f_cblas_ssbmv)
    (f_cblas_ssbmv)(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}
void cblas_ssbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_sbmv(Uplo, N, K, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_ssbmv(__handle, (rocblas_fill)Uplo, N, K, &alpha, __A, lda, __B, incX,
                &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssbmv(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_dsbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
    const double *alpha, const double *A, rocblas_int lda, const double *x,
    rocblas_int incx, const double *beta, double *y, rocblas_int incy);

rocblas_status rocblas_dsbmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, rocblas_int k, const double *alpha,
                             const double *A, rocblas_int lda, const double *x,
                             rocblas_int incx, const double *beta, double *y,
                             rocblas_int incy) {
  if (f_rocblas_dsbmv)
    return (f_rocblas_dsbmv)(handle, uplo, n, k, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_dsbmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const double alpha, const double *A,
                             const CBLAS_INT lda, const double *X,
                             const CBLAS_INT incX, const double beta, double *Y,
                             const CBLAS_INT incY);
static void _cblas_dsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, const double *X,
                         const CBLAS_INT incX, const double beta, double *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_dsbmv)
    (f_cblas_dsbmv)(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}
void cblas_dsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_sbmv(Uplo, N, K, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_dsbmv(__handle, (rocblas_fill)Uplo, N, K, &alpha, __A, lda, __B, incX,
                &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsbmv(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_csbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *beta, rocblas_float_complex *y,
    rocblas_int incy);

rocblas_status rocblas_csbmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, rocblas_int k,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (f_rocblas_csbmv)
    return (f_rocblas_csbmv)(handle, uplo, n, k, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_csbmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);
static void _cblas_csbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (f_cblas_csbmv)
    (f_cblas_csbmv)(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}
void cblas_csbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_sbmv(Uplo, N, K, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_csbmv(__handle, (rocblas_fill)Uplo, N, K, alpha, __A, lda, __B, incX,
                beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_csbmv(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_zsbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *beta, rocblas_double_complex *y,
    rocblas_int incy);

rocblas_status rocblas_zsbmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, rocblas_int k,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (f_rocblas_zsbmv)
    return (f_rocblas_zsbmv)(handle, uplo, n, k, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_zsbmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);
static void _cblas_zsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (f_cblas_zsbmv)
    (f_cblas_zsbmv)(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}
void cblas_zsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_sbmv(Uplo, N, K, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zsbmv(__handle, (rocblas_fill)Uplo, N, K, alpha, __A, lda, __B, incX,
                beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsbmv(layout, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
}
