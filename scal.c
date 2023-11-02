/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_scal(const CBLAS_INT N, const CBLAS_INT incX, size_t *size_a) {

  *size_a = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sscal)(rocblas_handle handle, rocblas_int n,
                                         const float *alpha, float *x,
                                         rocblas_int incx);
rocblas_status rocblas_sscal(rocblas_handle handle, rocblas_int n,
                             const float *alpha, float *x, rocblas_int incx) {
  if (*f_rocblas_sscal)
    return (f_rocblas_sscal)(handle, n, alpha, x, incx);
  return -1;
}

extern void (*f_cblas_sscal)(const CBLAS_INT N, const float alpha, float *X,
                             const CBLAS_INT incX);
static void _cblas_sscal(const CBLAS_INT N, const float alpha, float *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_sscal)
    (f_cblas_sscal)(N, alpha, X, incX);
}
void cblas_sscal(const CBLAS_INT N, const float alpha, float *X,
                 const CBLAS_INT incX) {
  size_t size_a;
  int s = size_scal(N, incX, &size_a);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  rocblas_sscal(__handle, N, &alpha, __A, incX);
  hipMemcpy(X, __A, sizeof(float) * size_a, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sscal(N, alpha, X, incX);
}

extern rocblas_status (*f_rocblas_dscal)(rocblas_handle handle, rocblas_int n,
                                         const double *alpha, double *x,
                                         rocblas_int incx);
rocblas_status rocblas_dscal(rocblas_handle handle, rocblas_int n,
                             const double *alpha, double *x, rocblas_int incx) {
  if (*f_rocblas_dscal)
    return (f_rocblas_dscal)(handle, n, alpha, x, incx);
  return -1;
}

extern void (*f_cblas_dscal)(const CBLAS_INT N, const double alpha, double *X,
                             const CBLAS_INT incX);
static void _cblas_dscal(const CBLAS_INT N, const double alpha, double *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_dscal)
    (f_cblas_dscal)(N, alpha, X, incX);
}
void cblas_dscal(const CBLAS_INT N, const double alpha, double *X,
                 const CBLAS_INT incX) {
  size_t size_a;
  int s = size_scal(N, incX, &size_a);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  rocblas_dscal(__handle, N, &alpha, __A, incX);
  hipMemcpy(X, __A, sizeof(double) * size_a, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dscal(N, alpha, X, incX);
}

extern rocblas_status (*f_rocblas_cscal)(rocblas_handle handle, rocblas_int n,
                                         const rocblas_float_complex *alpha,
                                         rocblas_float_complex *x,
                                         rocblas_int incx);
rocblas_status rocblas_cscal(rocblas_handle handle, rocblas_int n,
                             const rocblas_float_complex *alpha,
                             rocblas_float_complex *x, rocblas_int incx) {
  if (*f_rocblas_cscal)
    return (f_rocblas_cscal)(handle, n, alpha, x, incx);
  return -1;
}

extern void (*f_cblas_cscal)(const CBLAS_INT N, const void *alpha, void *X,
                             const CBLAS_INT incX);
static void _cblas_cscal(const CBLAS_INT N, const void *alpha, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_cscal)
    (f_cblas_cscal)(N, alpha, X, incX);
}
void cblas_cscal(const CBLAS_INT N, const void *alpha, void *X,
                 const CBLAS_INT incX) {
  size_t size_a;
  int s = size_scal(N, incX, &size_a);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  rocblas_cscal(__handle, N, alpha, __A, incX);
  hipMemcpy(X, __A, 2 * sizeof(float) * size_a, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cscal(N, alpha, X, incX);
}

extern rocblas_status (*f_rocblas_zscal)(rocblas_handle handle, rocblas_int n,
                                         const rocblas_double_complex *alpha,
                                         rocblas_double_complex *x,
                                         rocblas_int incx);
rocblas_status rocblas_zscal(rocblas_handle handle, rocblas_int n,
                             const rocblas_double_complex *alpha,
                             rocblas_double_complex *x, rocblas_int incx) {
  if (*f_rocblas_zscal)
    return (f_rocblas_zscal)(handle, n, alpha, x, incx);
  return -1;
}

extern void (*f_cblas_zscal)(const CBLAS_INT N, const void *alpha, void *X,
                             const CBLAS_INT incX);
static void _cblas_zscal(const CBLAS_INT N, const void *alpha, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_zscal)
    (f_cblas_zscal)(N, alpha, X, incX);
}
void cblas_zscal(const CBLAS_INT N, const void *alpha, void *X,
                 const CBLAS_INT incX) {
  size_t size_a;
  int s = size_scal(N, incX, &size_a);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  rocblas_zscal(__handle, N, alpha, __A, incX);
  hipMemcpy(X, __A, 2 * sizeof(double) * size_a, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zscal(N, alpha, X, incX);
}
