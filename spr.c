/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_spr(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                    size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = ((N * (N + 1)) / 2);

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sspr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const float *alpha, const float *x,
                                        rocblas_int incx, float *A);

rocblas_status rocblas_sspr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const float *alpha, const float *x,
                            rocblas_int incx, float *A) {
  if (*f_rocblas_sspr)
    return (f_rocblas_sspr)(handle, uplo, n, alpha, x, incx, A);
  return -1;
}

extern void (*f_cblas_sspr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const float alpha,
                            const float *X, const CBLAS_INT incX, float *A);
static void _cblas_sspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const float alpha, const float *X, const CBLAS_INT incX,
                        float *A) {
  if (*f_cblas_sspr)
    (f_cblas_sspr)(layout, Uplo, N, alpha, X, incX, A);
}

void cblas_sspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const float alpha, const float *X, const CBLAS_INT incX,
                float *A) {

  size_t size_a, size_b;
  int s = size_spr(Uplo, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_sspr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B);

  hipMemcpy(A, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sspr(layout, Uplo, N, alpha, X, incX, A);
}

extern rocblas_status (*f_rocblas_dspr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const double *alpha, const double *x,
                                        rocblas_int incx, double *A);
rocblas_status rocblas_dspr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const double *alpha, const double *x,
                            rocblas_int incx, double *A) {
  if (*f_rocblas_dspr)
    return (f_rocblas_dspr)(handle, uplo, n, alpha, x, incx, A);
  return -1;
}

extern void (*f_cblas_dspr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const double alpha,
                            const double *X, const CBLAS_INT incX, double *A);
static void _cblas_dspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const double alpha, const double *X,
                        const CBLAS_INT incX, double *A) {
  if (*f_cblas_dspr)
    (f_cblas_dspr)(layout, Uplo, N, alpha, X, incX, A);
}

void cblas_dspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const double alpha, const double *X, const CBLAS_INT incX,
                double *A) {

  size_t size_a, size_b;
  int s = size_spr(Uplo, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dspr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B);

  hipMemcpy(A, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dspr(layout, Uplo, N, alpha, X, incX, A);
}
