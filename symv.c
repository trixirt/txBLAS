/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_symv(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT lda,
                     const CBLAS_INT incX, const CBLAS_INT incY, size_t *size_a,
                     size_t *size_b, size_t *size_c) {

  *size_a = lda * N;
  *size_b = (1 + (N - 1) * abs(incX));
  *size_c = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}
extern rocblas_status (*f_rocblas_ssymv)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const float *alpha, const float *A,
                                         rocblas_int lda, const float *x,
                                         rocblas_int incx, const float *beta,
                                         float *y, rocblas_int incy);

rocblas_status rocblas_ssymv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const float *alpha, const float *A,
                             rocblas_int lda, const float *x, rocblas_int incx,
                             const float *beta, float *y, rocblas_int incy) {
  if (f_rocblas_ssymv)
    return rocblas_ssymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y,
                         incy);
  return -1;
}

extern void (*f_cblas_ssymv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const float alpha,
                             const float *A, const CBLAS_INT lda,
                             const float *X, const CBLAS_INT incX,
                             const float beta, float *Y, const CBLAS_INT incY);

static void _cblas_ssymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const float alpha, const float *A,
                         const CBLAS_INT lda, const float *X,
                         const CBLAS_INT incX, const float beta, float *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_ssymv)
    (f_cblas_ssymv)(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void cblas_ssymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *X, const CBLAS_INT incX, const float beta,
                 float *Y, const CBLAS_INT incY) {
  size_t size_a, size_b, size_c;
  int s;
  s = size_symv(Uplo, N, lda, incX, incY, &size_a, &size_b, &size_c);

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_ssymv(__handle, (rocblas_fill)Uplo, N, &alpha, __A, lda, __B, incX,
                &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_dsymv)(rocblas_handle handle,
                                         rocblas_fill uplo, rocblas_int n,
                                         const double *alpha, const double *A,
                                         rocblas_int lda, const double *x,
                                         rocblas_int incx, const double *beta,
                                         double *y, rocblas_int incy);

rocblas_status rocblas_dsymv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, const double *x,
                             rocblas_int incx, const double *beta, double *y,
                             rocblas_int incy) {
  if (f_rocblas_dsymv)
    return rocblas_dsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y,
                         incy);
  return -1;
}

extern void (*f_cblas_dsymv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const double alpha,
                             const double *A, const CBLAS_INT lda,
                             const double *X, const CBLAS_INT incX,
                             const double beta, double *Y,
                             const CBLAS_INT incY);

static void _cblas_dsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const double alpha, const double *A,
                         const CBLAS_INT lda, const double *X,
                         const CBLAS_INT incX, const double beta, double *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_dsymv)
    (f_cblas_dsymv)(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void cblas_dsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *X, const CBLAS_INT incX, const double beta,
                 double *Y, const CBLAS_INT incY) {
  size_t size_a, size_b, size_c;
  int s;
  s = size_symv(Uplo, N, lda, incX, incY, &size_a, &size_b, &size_c);

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_dsymv(__handle, (rocblas_fill)Uplo, N, &alpha, __A, lda, __B, incX,
                &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_csymv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *beta, rocblas_float_complex *y,
    rocblas_int incy);

rocblas_status rocblas_csymv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (f_rocblas_csymv)
    return rocblas_csymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y,
                         incy);
  return -1;
}

extern void (*f_cblas_csymv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_csymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *X,
                         const CBLAS_INT incX, const void *beta, void *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_csymv)
    (f_cblas_csymv)(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void cblas_csymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {
  size_t size_a, size_b, size_c;
  int s;
  s = size_symv(Uplo, N, lda, incX, incY, &size_a, &size_b, &size_c);

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_csymv(__handle, (rocblas_fill)Uplo, N, alpha, __A, lda, __B, incX,
                beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_csymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_zsymv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *beta, rocblas_double_complex *y,
    rocblas_int incy);

rocblas_status rocblas_zsymv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (f_rocblas_zsymv)
    return rocblas_zsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y,
                         incy);
  return -1;
}

extern void (*f_cblas_zsymv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_zsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *X,
                         const CBLAS_INT incX, const void *beta, void *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_zsymv)
    (f_cblas_zsymv)(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void cblas_zsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {
  size_t size_a, size_b, size_c;
  int s;
  s = size_symv(Uplo, N, lda, incX, incY, &size_a, &size_b, &size_c);

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zsymv(__handle, (rocblas_fill)Uplo, N, alpha, __A, lda, __B, incX,
                beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
}
