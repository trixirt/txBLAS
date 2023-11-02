/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_axpy(const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_saxpy)(rocblas_handle handle, rocblas_int n,
                                         const float *alpha, const float *x,
                                         rocblas_int incx, float *y,
                                         rocblas_int incy);
rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n,
                             const float *alpha, const float *x,
                             rocblas_int incx, float *y, rocblas_int incy) {
  if (*f_rocblas_saxpy)
    return (f_rocblas_saxpy)(handle, n, alpha, x, incx, y, incy);
  return -1;
}

extern void (*f_cblas_saxpy)(const CBLAS_INT N, const float alpha,
                             const float *X, const CBLAS_INT incX, float *Y,
                             const CBLAS_INT incY);
static void _cblas_saxpy(const CBLAS_INT N, const float alpha, const float *X,
                         const CBLAS_INT incX, float *Y, const CBLAS_INT incY) {
  if (*f_cblas_saxpy)
    (f_cblas_saxpy)(N, alpha, X, incX, Y, incY);
}
void cblas_saxpy(const CBLAS_INT N, const float alpha, const float *X,
                 const CBLAS_INT incX, float *Y, const CBLAS_INT incY) {
  size_t size_a, size_b;
  int s = size_axpy(N, incX, incY, &size_a, &size_b);

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_saxpy(__handle, N, &alpha, __A, incX, __B, incY);

  hipMemcpy(Y, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_saxpy(N, alpha, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_daxpy)(rocblas_handle handle, rocblas_int n,
                                         const double *alpha, const double *x,
                                         rocblas_int incx, double *y,
                                         rocblas_int incy);
rocblas_status rocblas_daxpy(rocblas_handle handle, rocblas_int n,
                             const double *alpha, const double *x,
                             rocblas_int incx, double *y, rocblas_int incy) {
  if (*f_rocblas_daxpy)
    return (f_rocblas_daxpy)(handle, n, alpha, x, incx, y, incy);
  return -1;
}

extern void (*f_cblas_daxpy)(const CBLAS_INT N, const double alpha,
                             const double *X, const CBLAS_INT incX, double *Y,
                             const CBLAS_INT incY);
static void _cblas_daxpy(const CBLAS_INT N, const double alpha, const double *X,
                         const CBLAS_INT incX, double *Y,
                         const CBLAS_INT incY) {
  if (*f_cblas_daxpy)
    (f_cblas_daxpy)(N, alpha, X, incX, Y, incY);
}
void cblas_daxpy(const CBLAS_INT N, const double alpha, const double *X,
                 const CBLAS_INT incX, double *Y, const CBLAS_INT incY) {
  size_t size_a, size_b;
  int s = size_axpy(N, incX, incY, &size_a, &size_b);

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_daxpy(__handle, N, &alpha, __A, incX, __B, incY);

  hipMemcpy(Y, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_daxpy(N, alpha, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_caxpy)(rocblas_handle handle, rocblas_int n,
                                         const rocblas_float_complex *alpha,
                                         const rocblas_float_complex *x,
                                         rocblas_int incx,
                                         rocblas_float_complex *y,
                                         rocblas_int incy);
rocblas_status rocblas_caxpy(rocblas_handle handle, rocblas_int n,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (*f_rocblas_caxpy)
    return (f_rocblas_caxpy)(handle, n, alpha, x, incx, y, incy);
  return -1;
}

extern void (*f_cblas_caxpy)(const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, void *Y,
                             const CBLAS_INT incY);
static void _cblas_caxpy(const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, void *Y, const CBLAS_INT incY) {
  if (*f_cblas_caxpy)
    (f_cblas_caxpy)(N, alpha, X, incX, Y, incY);
}
void cblas_caxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY) {
  size_t size_a, size_b;
  int s = size_axpy(N, incX, incY, &size_a, &size_b);

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_caxpy(__handle, N, alpha, __A, incX, __B, incY);

  hipMemcpy(Y, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_caxpy(N, alpha, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_zaxpy)(rocblas_handle handle, rocblas_int n,
                                         const rocblas_double_complex *alpha,
                                         const rocblas_double_complex *x,
                                         rocblas_int incx,
                                         rocblas_double_complex *y,
                                         rocblas_int incy);
rocblas_status rocblas_zaxpy(rocblas_handle handle, rocblas_int n,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy) {
  if (*f_rocblas_zaxpy)
    return (f_rocblas_zaxpy)(handle, n, alpha, x, incx, y, incy);
  return -1;
}

extern void (*f_cblas_zaxpy)(const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, void *Y,
                             const CBLAS_INT incY);
static void _cblas_zaxpy(const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, void *Y, const CBLAS_INT incY) {
  if (*f_cblas_zaxpy)
    (f_cblas_zaxpy)(N, alpha, X, incX, Y, incY);
}
void cblas_zaxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY) {
  size_t size_a, size_b;
  int s = size_axpy(N, incX, incY, &size_a, &size_b);

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zaxpy(__handle, N, alpha, __A, incX, __B, incY);

  hipMemcpy(Y, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_zaxpy(N, alpha, X, incX, Y, incY);
}
