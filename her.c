/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_her(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                    const CBLAS_INT lda, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = lda * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_cher)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float *alpha,
    const rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex *A,
    rocblas_int lda);
rocblas_status rocblas_cher(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const float *alpha,
                            const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *A, rocblas_int lda) {
  if (*f_rocblas_cher)
    return (f_rocblas_cher)(handle, uplo, n, alpha, x, incx, A, lda);
  return -1;
}

extern void (*f_cblas_cher)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const float alpha, const void *X,
                            const CBLAS_INT incX, void *A, const CBLAS_INT lda);
static void _cblas_cher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const float alpha, const void *X, const CBLAS_INT incX,
                        void *A, const CBLAS_INT lda) {
  if (*f_cblas_cher)
    (f_cblas_cher)(layout, Uplo, N, alpha, X, incX, A, lda);
}

void cblas_cher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const float alpha, const void *X, const CBLAS_INT incX, void *A,
                const CBLAS_INT lda) {

  size_t size_a, size_b;
  int s = size_her(Uplo, N, incX, lda, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_cher(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, lda);

  hipMemcpy(A, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cher(layout, Uplo, N, alpha, X, incX, A, lda);
}

extern rocblas_status (*f_rocblas_zher)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    const double *alpha, const rocblas_double_complex *x, rocblas_int incx,
    rocblas_double_complex *A, rocblas_int lda);
rocblas_status rocblas_zher(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const double *alpha,
                            const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *A, rocblas_int lda) {
  if (*f_rocblas_zher)
    return (f_rocblas_zher)(handle, uplo, n, alpha, x, incx, A, lda);
  return -1;
}

extern void (*f_cblas_zher)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const double alpha,
                            const void *X, const CBLAS_INT incX, void *A,
                            const CBLAS_INT lda);
static void _cblas_zher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const double alpha, const void *X, const CBLAS_INT incX,
                        void *A, const CBLAS_INT lda) {
  if (*f_cblas_zher)
    (f_cblas_zher)(layout, Uplo, N, alpha, X, incX, A, lda);
}

void cblas_zher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const double alpha, const void *X, const CBLAS_INT incX,
                void *A, const CBLAS_INT lda) {

  size_t size_a, size_b;
  int s = size_her(Uplo, N, incX, lda, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zher(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, lda);

  hipMemcpy(A, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zher(layout, Uplo, N, alpha, X, incX, A, lda);
}
