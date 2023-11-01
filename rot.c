/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_rot(const CBLAS_INT N, const CBLAS_INT incX,
                    const CBLAS_INT incY, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_srot)(rocblas_handle handle, rocblas_int n,
                                        float *x, rocblas_int incx, float *y,
                                        rocblas_int incy, const float *c,
                                        const float *s);
rocblas_status rocblas_srot(rocblas_handle handle, rocblas_int n, float *x,
                            rocblas_int incx, float *y, rocblas_int incy,
                            const float *c, const float *s) {
  if (*f_rocblas_srot)
    return (f_rocblas_srot)(handle, n, x, incx, y, incy, c, s);

  return -1;
}

extern void (*f_cblas_srot)(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                            float *Y, const CBLAS_INT incY, const float c,
                            const float s);
static void _cblas_srot(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                        float *Y, const CBLAS_INT incY, const float c,
                        const float s) {
  if (*f_cblas_srot)
    (f_cblas_srot)(N, X, incX, Y, incY, c, s);
}
void cblas_srot(const CBLAS_INT N, float *X, const CBLAS_INT incX, float *Y,
                const CBLAS_INT incY, const float c, const float s) {
  size_t size_a, size_b;
  int r = size_rot(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_srot(__handle, N, __A, incX, __B, incY, &c, &s);

  hipMemcpy(X, __A, sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_srot(N, X, incX, Y, incY, c, s);
}

extern rocblas_status (*f_rocblas_drot)(rocblas_handle handle, rocblas_int n,
                                        double *x, rocblas_int incx, double *y,
                                        rocblas_int incy, const double *c,
                                        const double *s);
rocblas_status rocblas_drot(rocblas_handle handle, rocblas_int n, double *x,
                            rocblas_int incx, double *y, rocblas_int incy,
                            const double *c, const double *s) {
  if (*f_rocblas_drot)
    return (f_rocblas_drot)(handle, n, x, incx, y, incy, c, s);

  return -1;
}

extern void (*f_cblas_drot)(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                            double *Y, const CBLAS_INT incY, const double c,
                            const double s);
static void _cblas_drot(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                        double *Y, const CBLAS_INT incY, const double c,
                        const double s) {
  if (*f_cblas_drot)
    (f_cblas_drot)(N, X, incX, Y, incY, c, s);
}
void cblas_drot(const CBLAS_INT N, double *X, const CBLAS_INT incX, double *Y,
                const CBLAS_INT incY, const double c, const double s) {
  size_t size_a, size_b;
  int r = size_rot(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_drot(__handle, N, __A, incX, __B, incY, &c, &s);

  hipMemcpy(X, __A, sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_drot(N, X, incX, Y, incY, c, s);
}

extern rocblas_status (*f_rocblas_csrot)(rocblas_handle handle, rocblas_int n,
                                         rocblas_float_complex *x,
                                         rocblas_int incx,
                                         rocblas_float_complex *y,
                                         rocblas_int incy, const float *c,
                                         const float *s);
rocblas_status rocblas_csrot(rocblas_handle handle, rocblas_int n,
                             rocblas_float_complex *x, rocblas_int incx,
                             rocblas_float_complex *y, rocblas_int incy,
                             const float *c, const float *s) {
  if (*f_rocblas_csrot)
    return (f_rocblas_csrot)(handle, n, x, incx, y, incy, c, s);

  return -1;
}

extern void (*f_cblas_csrot)(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                             float *Y, const CBLAS_INT incY, const float c,
                             const float s);
static void _cblas_csrot(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                         float *Y, const CBLAS_INT incY, const float c,
                         const float s) {
  if (*f_cblas_csrot)
    (f_cblas_csrot)(N, X, incX, Y, incY, c, s);
}
void cblas_csrot(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                 const CBLAS_INT incY, const float c, const float s) {
  size_t size_a, size_b;
  int r = size_rot(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_csrot(__handle, N, __A, incX, __B, incY, &c, &s);

  hipMemcpy(X, __A, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_csrot(N, X, incX, Y, incY, c, s);
}

extern rocblas_status (*f_rocblas_zsrot)(rocblas_handle handle, rocblas_int n,
                                         rocblas_double_complex *x,
                                         rocblas_int incx,
                                         rocblas_double_complex *y,
                                         rocblas_int incy, const double *c,
                                         const double *s);
rocblas_status rocblas_zsrot(rocblas_handle handle, rocblas_int n,
                             rocblas_double_complex *x, rocblas_int incx,
                             rocblas_double_complex *y, rocblas_int incy,
                             const double *c, const double *s) {
  if (*f_rocblas_zsrot)
    return (f_rocblas_zsrot)(handle, n, x, incx, y, incy, c, s);

  return -1;
}

extern void (*f_cblas_zsrot)(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                             double *Y, const CBLAS_INT incY, const double c,
                             const double s);
static void _cblas_zsrot(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                         double *Y, const CBLAS_INT incY, const double c,
                         const double s) {
  if (*f_cblas_zsrot)
    (f_cblas_zsrot)(N, X, incX, Y, incY, c, s);
}
void cblas_zsrot(const CBLAS_INT N, void *X, const CBLAS_INT incX, void *Y,
                 const CBLAS_INT incY, const double c, const double s) {
  size_t size_a, size_b;
  int r = size_rot(N, incX, incY, &size_a, &size_b);

  if (r)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zsrot(__handle, N, __A, incX, __B, incY, &c, &s);

  hipMemcpy(X, __A, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);
  hipMemcpy(Y, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsrot(N, X, incX, Y, incY, c, s);
}
