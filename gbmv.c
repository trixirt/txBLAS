/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_gbmv(CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                     const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU,
                     const CBLAS_INT lda, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b,
                     size_t *size_c) {

  *size_a = lda * N;
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

extern rocblas_status (*f_rocblas_sgbmv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, rocblas_int kl, rocblas_int ku, const float *alpha,
    const float *A, rocblas_int lda, const float *x, rocblas_int incx,
    const float *beta, float *y, rocblas_int incy);

rocblas_status rocblas_sgbmv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, rocblas_int kl,
                             rocblas_int ku, const float *alpha, const float *A,
                             rocblas_int lda, const float *x, rocblas_int incx,
                             const float *beta, float *y, rocblas_int incy) {
  if (*f_rocblas_sgbmv)
    return (f_rocblas_sgbmv)(handle, trans, m, n, kl, ku, alpha, A, lda, x,
                             incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_sgbmv)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             const CBLAS_INT M, const CBLAS_INT N,
                             const CBLAS_INT KL, const CBLAS_INT KU,
                             const float alpha, const float *A,
                             const CBLAS_INT lda, const float *X,
                             const CBLAS_INT incX, const float beta, float *Y,
                             const CBLAS_INT incY);

static void _cblas_sgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const CBLAS_INT KL, const CBLAS_INT KU,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         const float *X, const CBLAS_INT incX, const float beta,
                         float *Y, const CBLAS_INT incY) {
  if (*f_cblas_sgbmv)
    (f_cblas_sgbmv)(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void cblas_sgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                 const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *X, const CBLAS_INT incX, const float beta,
                 float *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gbmv(TransA, M, N, KL, KU, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  else
    s = size_gbmv(TransA, N, M, KU, KL, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_sgbmv(__handle, (rocblas_operation)TransA, M, N, KL, KU, &alpha,
                  __A, lda, __B, incX, &beta, __C, incY);
  else
    rocblas_sgbmv(__handle, (rocblas_operation)TransA, N, M, KU, KL, &alpha,
                  __A, lda, __B, incX, &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sgbmv(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y,
               incY);
}

extern rocblas_status (*f_rocblas_dgbmv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, rocblas_int kl, rocblas_int ku, const double *alpha,
    const double *A, rocblas_int lda, const double *x, rocblas_int incx,
    const double *beta, double *y, rocblas_int incy);

rocblas_status rocblas_dgbmv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, rocblas_int kl,
                             rocblas_int ku, const double *alpha,
                             const double *A, rocblas_int lda, const double *x,
                             rocblas_int incx, const double *beta, double *y,
                             rocblas_int incy) {
  if (*f_rocblas_dgbmv)
    return (f_rocblas_dgbmv)(handle, trans, m, n, kl, ku, alpha, A, lda, x,
                             incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_dgbmv)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             const CBLAS_INT M, const CBLAS_INT N,
                             const CBLAS_INT KL, const CBLAS_INT KU,
                             const double alpha, const double *A,
                             const CBLAS_INT lda, const double *X,
                             const CBLAS_INT incX, const double beta, double *Y,
                             const CBLAS_INT incY);

static void _cblas_dgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const CBLAS_INT KL, const CBLAS_INT KU,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, const double *X,
                         const CBLAS_INT incX, const double beta, double *Y,
                         const CBLAS_INT incY) {
  if (*f_cblas_dgbmv)
    (f_cblas_dgbmv)(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void cblas_dgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                 const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *X, const CBLAS_INT incX, const double beta,
                 double *Y, const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gbmv(TransA, M, N, KL, KU, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  else
    s = size_gbmv(TransA, N, M, KU, KL, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dgbmv(__handle, (rocblas_operation)TransA, M, N, KL, KU, &alpha,
                  __A, lda, __B, incX, &beta, __C, incY);
  else
    rocblas_dgbmv(__handle, (rocblas_operation)TransA, N, M, KU, KL, &alpha,
                  __A, lda, __B, incX, &beta, __C, incY);

  hipMemcpy(Y, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dgbmv(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y,
               incY);
}

extern rocblas_status (*f_rocblas_cgbmv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, rocblas_int kl, rocblas_int ku,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *beta, rocblas_float_complex *y,
    rocblas_int incy);

rocblas_status rocblas_cgbmv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, rocblas_int kl,
                             rocblas_int ku, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (*f_rocblas_cgbmv)
    return (f_rocblas_cgbmv)(handle, trans, m, n, kl, ku, alpha, A, lda, x,
                             incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_cgbmv)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             const CBLAS_INT M, const CBLAS_INT N,
                             const CBLAS_INT KL, const CBLAS_INT KU,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_cgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const CBLAS_INT KL, const CBLAS_INT KU,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_cgbmv)
    (f_cblas_cgbmv)(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void cblas_cgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                 const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gbmv(TransA, M, N, KL, KU, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  else
    s = size_gbmv(TransA, N, M, KU, KL, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_cgbmv(__handle, (rocblas_operation)TransA, M, N, KL, KU, alpha, __A,
                  lda, __B, incX, beta, __C, incY);
  else
    rocblas_cgbmv(__handle, (rocblas_operation)TransA, N, M, KU, KL, alpha, __A,
                  lda, __B, incX, beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cgbmv(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y,
               incY);
}

extern rocblas_status (*f_rocblas_zgbmv)(
    rocblas_handle handle, rocblas_operation trans, rocblas_int m,
    rocblas_int n, rocblas_int kl, rocblas_int ku,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *beta, rocblas_double_complex *y,
    rocblas_int incy);

rocblas_status rocblas_zgbmv(rocblas_handle handle, rocblas_operation trans,
                             rocblas_int m, rocblas_int n, rocblas_int kl,
                             rocblas_int ku,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (*f_rocblas_zgbmv)
    return (f_rocblas_zgbmv)(handle, trans, m, n, kl, ku, alpha, A, lda, x,
                             incx, beta, y, incy);
  return -1;
}

extern void (*f_cblas_zgbmv)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             const CBLAS_INT M, const CBLAS_INT N,
                             const CBLAS_INT KL, const CBLAS_INT KU,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *X,
                             const CBLAS_INT incX, const void *beta, void *Y,
                             const CBLAS_INT incY);

static void _cblas_zgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const CBLAS_INT KL, const CBLAS_INT KU,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *X, const CBLAS_INT incX, const void *beta,
                         void *Y, const CBLAS_INT incY) {
  if (*f_cblas_zgbmv)
    (f_cblas_zgbmv)(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void cblas_zgbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
                 const CBLAS_INT N, const CBLAS_INT KL, const CBLAS_INT KU,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta, void *Y,
                 const CBLAS_INT incY) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gbmv(TransA, M, N, KL, KU, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  else
    s = size_gbmv(TransA, N, M, KU, KL, lda, incX, incY, &size_a, &size_b,
                  &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, Y, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_zgbmv(__handle, (rocblas_operation)TransA, M, N, KL, KU, alpha, __A,
                  lda, __B, incX, beta, __C, incY);
  else
    rocblas_zgbmv(__handle, (rocblas_operation)TransA, N, M, KU, KL, alpha, __A,
                  lda, __B, incX, beta, __C, incY);

  hipMemcpy(Y, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zgbmv(layout, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y,
               incY);
}
