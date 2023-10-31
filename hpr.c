/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_hpr(CBLAS_UPLO Uplo, const CBLAS_INT N, const CBLAS_INT incX,
                    size_t *size_a, size_t *size_b) {

  *size_a = (1 + (N - 1) * abs(incX));
  *size_b = ((N * (N + 1)) / 2);

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_chpr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const float *alpha,
                                        const rocblas_float_complex *x,
                                        rocblas_int incx,
                                        rocblas_float_complex *AP);
rocblas_status rocblas_chpr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const float *alpha,
                            const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *AP) {
  if (*f_rocblas_chpr)
    return (f_rocblas_chpr)(handle, uplo, n, alpha, x, incx, AP);
  return -1;
}

extern void (*f_cblas_chpr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const float alpha, const void *X,
                            const CBLAS_INT incX, void *A);
static void _cblas_chpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const float alpha, const void *X, const CBLAS_INT incX,
                        void *A) {
  if (*f_cblas_chpr)
    (f_cblas_chpr)(layout, Uplo, N, alpha, X, incX, A);
}
void cblas_chpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const float alpha, const void *X, const CBLAS_INT incX,
                void *A) {
  size_t size_a, size_b;
  int s = size_hpr(Uplo, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_chpr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B);

  hipMemcpy(A, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;

fail:
  _cblas_chpr(layout, Uplo, N, alpha, X, incX, A);
}

extern rocblas_status (*f_rocblas_zhpr)(rocblas_handle handle,
                                        rocblas_fill uplo, rocblas_int n,
                                        const double *alpha,
                                        const rocblas_double_complex *x,
                                        rocblas_int incx,
                                        rocblas_double_complex *AP);
rocblas_status rocblas_zhpr(rocblas_handle handle, rocblas_fill uplo,
                            rocblas_int n, const double *alpha,
                            const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *AP) {
  if (*f_rocblas_zhpr)
    return (f_rocblas_zhpr)(handle, uplo, n, alpha, x, incx, AP);
  return -1;
}

extern void (*f_cblas_zhpr)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                            const CBLAS_INT N, const double alpha,
                            const void *X, const CBLAS_INT incX, void *A);
static void _cblas_zhpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                        const double alpha, const void *X, const CBLAS_INT incX,
                        void *A) {
  if (*f_cblas_zhpr)
    (f_cblas_zhpr)(layout, Uplo, N, alpha, X, incX, A);
}
void cblas_zhpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const double alpha, const void *X, const CBLAS_INT incX,
                void *A) {
  size_t size_a, size_b;
  int s = size_hpr(Uplo, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, X, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, A, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_zhpr(__handle, (rocblas_fill)Uplo, N, &alpha, __A, incX, __B);

  hipMemcpy(A, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;

fail:
  _cblas_zhpr(layout, Uplo, N, alpha, X, incX, A);
}
