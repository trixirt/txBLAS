/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_syr(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                    const CBLAS_INT lda, size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = lda * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_ssyr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const float *alpha, const float *x,
                                        rocblas_int incx, float *A,
                                        rocblas_int lda);
rocblas_status rocblas_ssyr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const float *alpha, const float *x,
                            rocblas_int incx, float *A, rocblas_int lda) {
  if (*f_rocblas_ssyr)
    return (f_rocblas_ssyr)(handle, uplo, n, alpha, x, incx, A, lda);
  return -1;
}

extern void (*f_cblas_ssyr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const float alpha,
                            const float *X, const CBLAS_INT incX, float *A,
                            const CBLAS_INT lda);
static void _cblas_ssyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const float alpha, const float *X, const CBLAS_INT incX,
                        float *A, const CBLAS_INT lda) {
  if (*f_cblas_ssyr)
    (f_cblas_ssyr)(layout, Uplo, N, alpha, X, incX, A, lda);
}

void cblas_ssyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const float alpha, const float *X, const CBLAS_INT incX,
                float *A, const CBLAS_INT lda) {

  size_t size_a, size_b;
  int s = size_syr(Uplo, N, incX, lda, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ssyr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, lda);

  hipMemcpy(A, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssyr(layout, Uplo, N, alpha, X, incX, A, lda);
}

extern rocblas_status (*f_rocblas_dsyr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const double *alpha, const double *x,
                                        rocblas_int incx, double *A,
                                        rocblas_int lda);
rocblas_status rocblas_dsyr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const double *alpha, const double *x,
                            rocblas_int incx, double *A, rocblas_int lda) {
  if (*f_rocblas_dsyr)
    return (f_rocblas_dsyr)(handle, uplo, n, alpha, x, incx, A, lda);
  return -1;
}

extern void (*f_cblas_dsyr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const double alpha,
                            const double *X, const CBLAS_INT incX, double *A,
                            const CBLAS_INT lda);
static void _cblas_dsyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const double alpha, const double *X,
                        const CBLAS_INT incX, double *A, const CBLAS_INT lda) {
  if (*f_cblas_dsyr)
    (f_cblas_dsyr)(layout, Uplo, N, alpha, X, incX, A, lda);
}

void cblas_dsyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const double alpha, const double *X, const CBLAS_INT incX,
                double *A, const CBLAS_INT lda) {

  size_t size_a, size_b;
  int s = size_syr(Uplo, N, incX, lda, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dsyr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B, lda);

  hipMemcpy(A, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsyr(layout, Uplo, N, alpha, X, incX, A, lda);
}

