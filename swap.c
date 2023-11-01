/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_swap(const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sswap)(rocblas_handle handle, rocblas_int n,
                                         float *x, rocblas_int incx, float *y,
                                         rocblas_int incy);
rocblas_status rocblas_sswap(rocblas_handle handle, rocblas_int n, float *x,
                             rocblas_int incx, float *y, rocblas_int incy) {
  if (*f_rocblas_sswap)
    return (f_rocblas_sswap)(handle, n, x, incx, y, incy);

  return -1;
}

extern void (*f_cblas_sswap)(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                             float *Y, const CBLAS_INT incY);
static void _cblas_sswap(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                         float *Y, const CBLAS_INT incY) {
  if (*f_cblas_sswap)
    (f_cblas_sswap)(N, X, incX, Y, incY);
}
void cblas_sswap(const CBLAS_INT N, float *X, const CBLAS_INT incX, float *Y,
                 const CBLAS_INT incY) {
  size_t size_a, size_b;
  int r = size_swap(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_sswap(__handle, N, __A, incX, __B, incY);

  hipMemcpy(X, __A, sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sswap(N, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_dswap)(rocblas_handle handle, rocblas_int n,
                                         double *x, rocblas_int incx, double *y,
                                         rocblas_int incy);
rocblas_status rocblas_dswap(rocblas_handle handle, rocblas_int n, double *x,
                             rocblas_int incx, double *y, rocblas_int incy) {
  if (*f_rocblas_dswap)
    return (f_rocblas_dswap)(handle, n, x, incx, y, incy);

  return -1;
}

extern void (*f_cblas_dswap)(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                             double *Y, const CBLAS_INT incY);
static void _cblas_dswap(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                         double *Y, const CBLAS_INT incY) {
  if (*f_cblas_dswap)
    (f_cblas_dswap)(N, X, incX, Y, incY);
}
void cblas_dswap(const CBLAS_INT N, double *X, const CBLAS_INT incX, double *Y,
                 const CBLAS_INT incY) {
  size_t size_a, size_b;
  int r = size_swap(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dswap(__handle, N, __A, incX, __B, incY);

  hipMemcpy(X, __A, sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dswap(N, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_cswap)(rocblas_handle handle, rocblas_int n,
                                         rocblas_float_complex *x,
                                         rocblas_int incx,
                                         rocblas_float_complex *y,
                                         rocblas_int incy);
rocblas_status rocblas_cswap(rocblas_handle handle, rocblas_int n,
                             rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy) {
  if (*f_rocblas_cswap)
    return (f_rocblas_cswap)(handle, n, x, incx, y, incy);

  return -1;
}

extern void (*f_cblas_cswap)(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                             float *Y, const CBLAS_INT incY);
static void _cblas_cswap(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                         float *Y, const CBLAS_INT incY) {
  if (*f_cblas_cswap)
    (f_cblas_cswap)(N, X, incX, Y, incY);
}
void cblas_cswap(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                 const CBLAS_INT incY) {
  size_t size_a, size_b;
  int r = size_swap(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_cswap(__handle, N, __A, incX, __B, incY);

  hipMemcpy(X, __A, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cswap(N, X, incX, Y, incY);
}

extern rocblas_status (*f_rocblas_zsswap)(rocblas_handle handle, rocblas_int n,
                                          rocblas_double_complex *x,
                                          rocblas_int incx,
                                          rocblas_double_complex *y,
                                          rocblas_int incy);
rocblas_status rocblas_zsswap(rocblas_handle handle, rocblas_int n,
                              rocblas_double_complex *x, rocblas_int incx,
                              rocblas_double_complex *y, rocblas_int incy) {
  if (*f_rocblas_zsswap)
    return (f_rocblas_zsswap)(handle, n, x, incx, y, incy);

  return -1;
}

extern void (*f_cblas_zsswap)(const CBLAS_INT N, double *X,
                              const CBLAS_INT incX, double *Y,
                              const CBLAS_INT incY);
static void _cblas_zsswap(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                          double *Y, const CBLAS_INT incY) {
  if (*f_cblas_zsswap)
    (f_cblas_zsswap)(N, X, incX, Y, incY);
}
void cblas_zsswap(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                  const CBLAS_INT incY, const double c, const double s) {
  size_t size_a, size_b;
  int r = size_swap(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zsswap(__handle, N, __A, incX, __B, incY);

  hipMemcpy(X, __A, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsswap(N, X, incX, Y, incY);
}
