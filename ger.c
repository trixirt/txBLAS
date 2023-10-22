/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_ger(const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT incX,
                    const CBLAS_INT incY, const CBLAS_INT lda, size_t *size_a,
                    size_t *size_b, size_t *size_c) {
  *size_a = (1 + (M - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));
  *size_c = lda * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sger)(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, const float *alpha,
                                        const float *x, rocblas_int incx,
                                        const float *y, rocblas_int incy,
                                        float *A, rocblas_int lda);
rocblas_status rocblas_sger(rocblas_handle handle, rocblas_int m, rocblas_int n,
                            const float *alpha, const float *x,
                            rocblas_int incx, const float *y, rocblas_int incy,
                            float *A, rocblas_int lda) {
  if (*f_rocblas_sger)
    return (f_rocblas_sger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_sger)(CBLAS_LAYOUT layout, const CBLAS_INT M,
                            const CBLAS_INT N, const float alpha,
                            const float *X, const CBLAS_INT incX,
                            const float *Y, const CBLAS_INT incY, float *A,
                            const CBLAS_INT lda);

static void _cblas_sger(CBLAS_LAYOUT layout, const CBLAS_INT M,
                        const CBLAS_INT N, const float alpha, const float *X,
                        const CBLAS_INT incX, const float *Y,
                        const CBLAS_INT incY, float *A, const CBLAS_INT lda) {
  if (*f_cblas_sger)
    (f_cblas_sger)(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_sger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const float alpha, const float *X, const CBLAS_INT incX,
                const float *Y, const CBLAS_INT incY, float *A,
                const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_ger(M, N, incX, incY, lda, &size_a, &size_b, &size_c);
  else
    s = size_ger(N, M, incY, incX, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_sger(__handle, M, N, &alpha, __A, incX, __B, incY, __C, lda);
  else
    rocblas_sger(__handle, N, M, &alpha, __B, incY, __A, incX, __C, lda);

  hipMemcpy(A, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sger(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

extern rocblas_status (*f_rocblas_dger)(rocblas_handle handle, rocblas_int m,
                                        rocblas_int n, const double *alpha,
                                        const double *x, rocblas_int incx,
                                        const double *y, rocblas_int incy,
                                        double *A, rocblas_int lda);
rocblas_status rocblas_dger(rocblas_handle handle, rocblas_int m, rocblas_int n,
                            const double *alpha, const double *x,
                            rocblas_int incx, const double *y, rocblas_int incy,
                            double *A, rocblas_int lda) {
  if (*f_rocblas_dger)
    return (f_rocblas_dger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_dger)(CBLAS_LAYOUT layout, const CBLAS_INT M,
                            const CBLAS_INT N, const double alpha,
                            const double *X, const CBLAS_INT incX,
                            const double *Y, const CBLAS_INT incY, double *A,
                            const CBLAS_INT lda);

static void _cblas_dger(CBLAS_LAYOUT layout, const CBLAS_INT M,
                        const CBLAS_INT N, const double alpha, const double *X,
                        const CBLAS_INT incX, const double *Y,
                        const CBLAS_INT incY, double *A, const CBLAS_INT lda) {
  if (*f_cblas_dger)
    (f_cblas_dger)(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_dger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const double alpha, const double *X, const CBLAS_INT incX,
                const double *Y, const CBLAS_INT incY, double *A,
                const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_ger(M, N, incX, incY, lda, &size_a, &size_b, &size_c);
  else
    s = size_ger(N, M, incY, incX, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dger(__handle, M, N, &alpha, __A, incX, __B, incY, __C, lda);
  else
    rocblas_dger(__handle, N, M, &alpha, __B, incY, __A, incX, __C, lda);

  hipMemcpy(A, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dger(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

extern rocblas_status (*f_rocblas_cgeru)(
    rocblas_handle handle, rocblas_int m, rocblas_int n,
    const rocblas_float_complex *alpha, const rocblas_float_complex *x,
    rocblas_int incx, const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *A, rocblas_int lda);
rocblas_status rocblas_cgeru(rocblas_handle handle, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy,
                             rocblas_float_complex *A, rocblas_int lda) {
  if (*f_rocblas_cgeru)
    return (f_rocblas_cgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_cgeru)(CBLAS_LAYOUT layout, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *A,
                             const CBLAS_INT lda);

static void _cblas_cgeru(CBLAS_LAYOUT layout, const CBLAS_INT M,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *A, const CBLAS_INT lda) {
  if (*f_cblas_cgeru)
    (f_cblas_cgeru)(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_cgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_ger(M, N, incX, incY, lda, &size_a, &size_b, &size_c);
  else
    s = size_ger(N, M, incY, incX, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_cgeru(__handle, M, N, alpha, __A, incX, __B, incY, __C, lda);
  else
    rocblas_cgeru(__handle, N, M, alpha, __B, incY, __A, incX, __C, lda);

  hipMemcpy(A, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cgeru(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

extern rocblas_status (*f_rocblas_zgeru)(
    rocblas_handle handle, rocblas_int m, rocblas_int n,
    const rocblas_double_complex *alpha, const rocblas_double_complex *x,
    rocblas_int incx, const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_zgeru(rocblas_handle handle, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy,
                             rocblas_double_complex *A, rocblas_int lda) {
  if (*f_rocblas_zgeru)
    return (f_rocblas_zgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
  return -1;
}

extern void (*f_cblas_zgeru)(CBLAS_LAYOUT layout, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *A,
                             const CBLAS_INT lda);

static void _cblas_zgeru(CBLAS_LAYOUT layout, const CBLAS_INT M,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *A, const CBLAS_INT lda) {
  if (*f_cblas_zgeru)
    (f_cblas_zgeru)(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}

void cblas_zgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_ger(M, N, incX, incY, lda, &size_a, &size_b, &size_c);
  else
    s = size_ger(N, M, incY, incX, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_zgeru(__handle, M, N, alpha, __A, incX, __B, incY, __C, lda);
  else
    rocblas_zgeru(__handle, N, M, alpha, __B, incY, __A, incX, __C, lda);

  hipMemcpy(A, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zgeru(layout, M, N, alpha, X, incX, Y, incY, A, lda);
}
