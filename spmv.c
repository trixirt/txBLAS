/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_spmv(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b,
                     size_t *size_c) {

  *size_a = (N * (N + 1)) / 2;
  *size_b = (1 + (N - 1) * abs(incX));
  *size_c = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sspmv)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const float *alpha, const float *A,
                                         const float *x, rocblas_int incx,
                                         const float *beta, float *y,
                                         rocblas_int incy);
rocblas_status rocblas_sspmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const float *alpha, const float *A,
                             const float *x, rocblas_int incx,
                             const float *beta, float *y, rocblas_int incy) {
  if (*f_rocblas_sspmv)
    return (f_rocblas_sspmv)(handle, uplo, n, alpha, A, x, incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_sspmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const float alpha,
                             const float *Ap, const float *X,
                             const CBLAS_INT incX, const float beta, float *Y,
                             const CBLAS_INT incY);
static void _cblas_sspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const float alpha, const float *Ap,
                         const float *X, const CBLAS_INT incX, const float beta,
                         float *Y, const CBLAS_INT incY) {
  if (*f_cblas_sspmv)
    (f_cblas_sspmv)(layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

void cblas_sspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const float alpha, const float *A, const float *X,
                 const CBLAS_INT incX, const float beta, float *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s = size_spmv(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_sspmv(__handle, (rocblas_fill)Uplo, N, &alpha, __A, __B, incX, &beta,
                __C, incY);

  hipMemcpy(Y, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sspmv(layout, Uplo, N, alpha, A, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_dspmv)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const double *alpha, const double *A,
                                         const double *x, rocblas_int incx,
                                         const double *beta, double *y,
                                         rocblas_int incy);
rocblas_status rocblas_dspmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const double *alpha,
                             const double *A, const double *x, rocblas_int incx,
                             const double *beta, double *y, rocblas_int incy) {
  if (*f_rocblas_dspmv)
    return (f_rocblas_dspmv)(handle, uplo, n, alpha, A, x, incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_dspmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const double alpha,
                             const double *Ap, const double *X,
                             const CBLAS_INT incX, const double beta, double *Y,
                             const CBLAS_INT incY);
static void _cblas_dspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const double alpha,
                         const double *Ap, const double *X,
                         const CBLAS_INT incX, const double beta, double *Y,
                         const CBLAS_INT incY) {
  if (*f_cblas_dspmv)
    (f_cblas_dspmv)(layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

void cblas_dspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const double alpha, const double *A, const double *X,
                 const CBLAS_INT incX, const double beta, double *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s = size_spmv(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_dspmv(__handle, (rocblas_fill)Uplo, N, &alpha, __A, __B, incX, &beta,
                __C, incY);

  hipMemcpy(Y, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dspmv(layout, Uplo, N, alpha, A, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_cspmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *beta, rocblas_float_complex *y,
    rocblas_int incy);
rocblas_status rocblas_cspmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (*f_rocblas_cspmv)
    return (f_rocblas_cspmv)(handle, uplo, n, alpha, A, x, incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_cspmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *Ap, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);
static void _cblas_cspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *Ap,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_cspmv)
    (f_cblas_cspmv)(layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

void cblas_cspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *A, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s = size_spmv(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_cspmv(__handle, (rocblas_fill)Uplo, N, alpha, __A, __B, incX, beta,
                __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cspmv(layout, Uplo, N, alpha, A, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_zspmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *beta, rocblas_double_complex *y,
    rocblas_int incy);
rocblas_status rocblas_zspmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (*f_rocblas_zspmv)
    return (f_rocblas_zspmv)(handle, uplo, n, alpha, A, x, incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_zspmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *Ap, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);
static void _cblas_zspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *Ap,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_zspmv)
    (f_cblas_zspmv)(layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
}

void cblas_zspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *A, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s = size_spmv(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zspmv(__handle, (rocblas_fill)Uplo, N, alpha, __A, __B, incX, beta,
                __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zspmv(layout, Uplo, N, alpha, A, X, incX, beta, Y, incY);
}
