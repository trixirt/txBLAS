/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_her2(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                     const CBLAS_INT incY, const CBLAS_INT lda, size_t *size_a,
                     size_t *size_b, size_t *size_c) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = (1 + (N - 1) * abs(incY));
  *size_c = lda * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_cher2)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_float_complex *alpha, const rocblas_float_complex *x,
    rocblas_int incx, const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *A, rocblas_int lda);
rocblas_status rocblas_cher2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *x, rocblas_int incx,
                             const rocblas_float_complex *y, rocblas_int incy,
                             rocblas_float_complex *A, rocblas_int lda) {
  if (*f_rocblas_cher2)
    (f_rocblas_cher2)(handle, uplo, n, alpha, x, incx, y, incy, A, lda);

  return -1;
}

extern void (*f_cblas_cher2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *A,
                             const CBLAS_INT lda);
static void _cblas_cher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *A, const CBLAS_INT lda) {
  if (*f_cblas_cher2)
    (f_cblas_cher2)(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}
void cblas_cher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s = size_her2(Uplo, N, incX, incY, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_cher2(__handle, (rocblas_fill)Uplo, N, alpha, __A, incX, __B, incY,
                __C, lda);

  hipMemcpy(A, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cher2(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}

extern rocblas_status (*f_rocblas_zher2)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const rocblas_double_complex *alpha, const rocblas_double_complex *x,
    rocblas_int incx, const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_zher2(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *x, rocblas_int incx,
                             const rocblas_double_complex *y, rocblas_int incy,
                             rocblas_double_complex *A, rocblas_int lda) {
  if (*f_rocblas_zher2)
    (f_rocblas_zher2)(handle, uplo, n, alpha, x, incx, y, incy, A, lda);

  return -1;
}

extern void (*f_cblas_zher2)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             const CBLAS_INT N, const void *alpha,
                             const void *X, const CBLAS_INT incX, const void *Y,
                             const CBLAS_INT incY, void *A,
                             const CBLAS_INT lda);
static void _cblas_zher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         const CBLAS_INT N, const void *alpha, const void *X,
                         const CBLAS_INT incX, const void *Y,
                         const CBLAS_INT incY, void *A, const CBLAS_INT lda) {
  if (*f_cblas_zher2)
    (f_cblas_zher2)(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}
void cblas_zher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A,
                 const CBLAS_INT lda) {

  size_t size_a, size_b, size_c;
  int s = size_her2(Uplo, N, incX, incY, lda, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, Y, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, A, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zher2(__handle, (rocblas_fill)Uplo, N, alpha, __A, incX, __B, incY,
                __C, lda);

  hipMemcpy(A, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zher2(layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
}
