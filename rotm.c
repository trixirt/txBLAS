/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_rotm(const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_srotm)(rocblas_handle handle, rocblas_int n,
                                         float *x, rocblas_int incx, float *y,
                                         rocblas_int incy, const float *param);
rocblas_status rocblas_srotm(rocblas_handle handle, rocblas_int n, float *x,
                             rocblas_int incx, float *y, rocblas_int incy,
                             const float *param) {
  if (*f_rocblas_srotm)
    return (f_rocblas_srotm)(handle, n, x, incx, y, incy, param);
  return -1;
}

extern void (*f_cblas_srotm)(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                             float *Y, const CBLAS_INT incY, const float *P);
static void _cblas_srotm(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                         float *Y, const CBLAS_INT incY, const float *P) {
  if (*f_cblas_srotm)
    (f_cblas_srotm)(N, X, incX, Y, incY, P);
}
void cblas_srotm(const CBLAS_INT N, float *X, const CBLAS_INT incX, float *Y,
                 const CBLAS_INT incY, const float *P) {
  size_t size_a, size_b;
  int r = size_rotm(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_srotm(__handle, N, __A, incX, __B, incY, P);

  hipMemcpy(X, __A, sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_srotm(N, X, incX, Y, incY, P);
}

extern rocblas_status (*f_rocblas_drotm)(rocblas_handle handle, rocblas_int n,
                                         double *x, rocblas_int incx, double *y,
                                         rocblas_int incy, const double *param);
rocblas_status rocblas_drotm(rocblas_handle handle, rocblas_int n, double *x,
                             rocblas_int incx, double *y, rocblas_int incy,
                             const double *param) {
  if (*f_rocblas_drotm)
    return (f_rocblas_drotm)(handle, n, x, incx, y, incy, param);
  return -1;
}

extern void (*f_cblas_drotm)(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                             double *Y, const CBLAS_INT incY, const double *P);
static void _cblas_drotm(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                         double *Y, const CBLAS_INT incY, const double *P) {
  if (*f_cblas_drotm)
    (f_cblas_drotm)(N, X, incX, Y, incY, P);
}
void cblas_drotm(const CBLAS_INT N, double *X, const CBLAS_INT incX, double *Y,
                 const CBLAS_INT incY, const double *P) {
  size_t size_a, size_b;
  int r = size_rotm(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_drotm(__handle, N, __A, incX, __B, incY, P);

  hipMemcpy(X, __A, sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_drotm(N, X, incX, Y, incY, P);
}

extern rocblas_status (*f_rocblas_crotm)(rocblas_handle handle, rocblas_int n,
                                         rocblas_float_complex *x,
                                         rocblas_int incx,
                                         rocblas_float_complex *y,
                                         rocblas_int incy, const float *param);
rocblas_status rocblas_crotm(rocblas_handle handle, rocblas_int n,
                             rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy,
                             const float *param) {
  if (*f_rocblas_crotm)
    return (f_rocblas_crotm)(handle, n, x, incx, y, incy, param);
  return -1;
}

extern void (*f_cblas_crotm)(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                             void *Y, const CBLAS_INT incY, const float *P);
static void _cblas_crotm(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                         void *Y, const CBLAS_INT incY, const float *P) {
  if (*f_cblas_crotm)
    (f_cblas_crotm)(N, X, incX, Y, incY, P);
}
void cblas_crotm(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                 const CBLAS_INT incY, const float *P) {
  size_t size_a, size_b;
  int r = size_rotm(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_crotm(__handle, N, __A, incX, __B, incY, P);

  hipMemcpy(X, __A, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_crotm(N, X, incX, Y, incY, P);
}

extern rocblas_status (*f_rocblas_zrotm)(rocblas_handle handle, rocblas_int n,
                                         rocblas_double_complex *x,
                                         rocblas_int incx,
                                         rocblas_double_complex *y,
                                         rocblas_int incy, const double *param);
rocblas_status rocblas_zrotm(rocblas_handle handle, rocblas_int n,
                             rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy,
                             const double *param) {
  if (*f_rocblas_zrotm)
    return (f_rocblas_zrotm)(handle, n, x, incx, y, incy, param);
  return -1;
}

extern void (*f_cblas_zrotm)(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                             void *Y, const CBLAS_INT incY, const double *P);
static void _cblas_zrotm(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                         void *Y, const CBLAS_INT incY, const double *P) {
  if (*f_cblas_zrotm)
    (f_cblas_zrotm)(N, X, incX, Y, incY, P);
}
void cblas_zrotm(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                 const CBLAS_INT incY, const double *P) {
  size_t size_a, size_b;
  int r = size_rotm(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zrotm(__handle, N, __A, incX, __B, incY, P);

  hipMemcpy(X, __A, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zrotm(N, X, incX, Y, incY, P);
}
