/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_gemv(const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                     const CBLAS_INT N, const CBLAS_INT lda,
                     const CBLAS_INT incX, const CBLAS_INT incY, size_t *size_a,
                     size_t *size_b, size_t *size_c) {

  *size_a = N;
  if (TransA == CblasNoTrans) {
    *size_b = (1 + (N - 1) * abs(incX));
    *size_c = (1 + (M - 1) * abs(incX));
  } else {
    *size_b = (1 + (M - 1) * abs(incX));
    *size_c = (1 + (N - 1) * abs(incX));
  }

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sgemv)(rocblas_handle handle,
                                         rocblas_operation trans, rocblas_int m,
                                         rocblas_int n, const float *alpha,
                                         const float *A, rocblas_int lda,
                                         const float *x, rocblas_int incx,
                                         const float *beta, float *y,
                                         rocblas_int incy);
rocblas_status rocblas_sgemv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, const float *alpha,
                             const float *A, rocblas_int lda, const float *x,
                             rocblas_int incx, const float *beta, float *y,
                             rocblas_int incy) {
  if (f_rocblas_sgemv)
    return (f_rocblas_sgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_sgemv)(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                             const CBLAS_INT N, const float alpha,
                             const float *A, const CBLAS_INT lda,
                             const float *X, const CBLAS_INT incX,
                             const float beta, float *Y, const CBLAS_INT incY);

static void _cblas_sgemv(const CBLAS_LAYOUT layout,
                         const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                         const CBLAS_INT N, const float alpha, const float *A,
                         const CBLAS_INT lda, const float *X,
                         const CBLAS_INT incX, const float beta, float *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_sgemv)
    (f_cblas_sgemv)(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
}

void cblas_sgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_INT M, const CBLAS_INT N, const float alpha,
                 const float *A, const CBLAS_INT lda, const float *X,
                 const CBLAS_INT incX, const float beta, float *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gemv(TransA, M, N, lda, incX, incY, &size_a, &size_b, &size_c);
  else
    s = size_gemv(TransA, N, M, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_sgemv(__handle, (rocblas_operation)TransA, M, N, &alpha, __A, lda,
                  __B, incX, &beta, __C, incY);
  else
    rocblas_sgemv(__handle, (rocblas_operation)TransA, N, M, &alpha, __A, lda,
                  __B, incX, &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_dgemv)(rocblas_handle handle,
                                         rocblas_operation trans, rocblas_int m,
                                         rocblas_int n, const double *alpha,
                                         const double *A, rocblas_int lda,
                                         const double *x, rocblas_int incx,
                                         const double *beta, double *y,
                                         rocblas_int incy);
rocblas_status rocblas_dgemv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, const double *x,
                             rocblas_int incx, const double *beta, double *y,
                             rocblas_int incy) {
  if (f_rocblas_dgemv)
    return (f_rocblas_dgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_dgemv)(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                             const CBLAS_INT N, const double alpha,
                             const double *A, const CBLAS_INT lda,
                             const double *X, const CBLAS_INT incX,
                             const double beta, double *Y,
                             const CBLAS_INT incY);

static void _cblas_dgemv(const CBLAS_LAYOUT layout,
                         const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                         const CBLAS_INT N, const double alpha, const double *A,
                         const CBLAS_INT lda, const double *X,
                         const CBLAS_INT incX, const double beta, double *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_dgemv)
    (f_cblas_dgemv)(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
}

void cblas_dgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_INT M, const CBLAS_INT N, const double alpha,
                 const double *A, const CBLAS_INT lda, const double *X,
                 const CBLAS_INT incX, const double beta, double *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gemv(TransA, M, N, lda, incX, incY, &size_a, &size_b, &size_c);
  else
    s = size_gemv(TransA, N, M, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dgemv(__handle, (rocblas_operation)TransA, M, N, &alpha, __A, lda,
                  __B, incX, &beta, __C, incY);
  else
    rocblas_dgemv(__handle, (rocblas_operation)TransA, N, M, &alpha, __A, lda,
                  __B, incX, &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_cgemv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *beta, rocblas_float_complex *y,
    rocblas_int incy);
rocblas_status rocblas_cgemv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (f_rocblas_cgemv)
    return (f_rocblas_cgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_cgemv)(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_cgemv(const CBLAS_LAYOUT layout,
                         const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                         const CBLAS_INT N, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *X,
                         const CBLAS_INT incX, const void *beta, void *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_cgemv)
    (f_cblas_cgemv)(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
}

void cblas_cgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gemv(TransA, M, N, lda, incX, incY, &size_a, &size_b, &size_c);
  else
    s = size_gemv(TransA, N, M, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_cgemv(__handle, (rocblas_operation)TransA, M, N, alpha, __A, lda,
                  __B, incX, beta, __C, incY);
  else
    rocblas_cgemv(__handle, (rocblas_operation)TransA, N, M, alpha, __A, lda,
                  __B, incX, beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

extern rocblas_status (*f_rocblas_zgemv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *beta, rocblas_double_complex *y,
    rocblas_int incy);
rocblas_status rocblas_zgemv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (f_rocblas_zgemv)
    return (f_rocblas_zgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                             y, incy);
  return -1;
}

extern void (*f_cblas_zgemv)(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_zgemv(const CBLAS_LAYOUT layout,
                         const CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                         const CBLAS_INT N, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *X,
                         const CBLAS_INT incX, const void *beta, void *Y,
                         const CBLAS_INT incY) {
  if (f_cblas_zgemv)
    (f_cblas_zgemv)(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
}

void cblas_zgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gemv(TransA, M, N, lda, incX, incY, &size_a, &size_b, &size_c);
  else
    s = size_gemv(TransA, N, M, lda, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_zgemv(__handle, (rocblas_operation)TransA, M, N, alpha, __A, lda,
                  __B, incX, beta, __C, incY);
  else
    rocblas_zgemv(__handle, (rocblas_operation)TransA, N, M, alpha, __A, lda,
                  __B, incX, beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}
