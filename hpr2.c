/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_hpr2(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, size_t *size_a, size_t *size_b,
                     size_t *size_c) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));
  *size_c = ((N * (N + 1)) / 2);

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_chpr2)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_float_complex *alpha, const rocblas_float_complex *x,
    rocblas_int incx, const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *AP);
rocblas_status rocblas_chpr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy,
                             rocblas_float_complex *AP) {
  if (*f_rocblas_chpr2)
    return (f_rocblas_chpr2)(handle, uplo, n, alpha, x, incx, y, incy, AP);
  return -1;
}

extern void (*f_cblas_chpr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *Ap);
static void _cblas_chpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *Ap) {
  if (*f_cblas_chpr2)
    (f_cblas_chpr2)(layout, Uplo, N, alpha, X, incX, Y, incY, Ap);
}

void cblas_chpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A) {

  size_t size_a, size_b, size_c;
  int s = size_hpr2(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_chpr2(__handle, (rocblas_fill)Uplo, N, alpha, __A, incX, __B, incY,
                __C);

  hipMemcpy(A, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_chpr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}

extern rocblas_status (*f_rocblas_zhpr2)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_double_complex *alpha, const rocblas_double_complex *x,
    rocblas_int incx, const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *AP);
rocblas_status rocblas_zhpr2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy,
                             rocblas_double_complex *AP) {
  if (*f_rocblas_zhpr2)
    return (f_rocblas_zhpr2)(handle, uplo, n, alpha, x, incx, y, incy, AP);
  return -1;
}

extern void (*f_cblas_zhpr2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *Ap);
static void _cblas_zhpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *Ap) {
  if (*f_cblas_zhpr2)
    (f_cblas_zhpr2)(layout, Uplo, N, alpha, X, incX, Y, incY, Ap);
}

void cblas_zhpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A) {

  size_t size_a, size_b, size_c;
  int s = size_hpr2(Uplo, N, incX, incY, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zhpr2(__handle, (rocblas_fill)Uplo, N, alpha, __A, incX, __B, incY,
                __C);

  hipMemcpy(A, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zhpr2(layout, Uplo, N, alpha, X, incX, Y, incY, A);
}
