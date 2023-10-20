/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_her2k(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                      const CBLAS_INT K, const CBLAS_INT lda,
                      const CBLAS_INT ldb, const CBLAS_INT ldc, size_t *size_a,
                      size_t *size_b, size_t *size_c) {

  if (Trans == CblasNoTrans) {
    *size_a = K * lda;
    *size_b = K * ldb;
  } else {
    *size_a = N * lda;
    *size_b = N * ldb;
  }

  *size_c = N * ldc;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_cher2k)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_int n, rocblas_int k, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb, const float *beta,
    rocblas_float_complex *C, rocblas_int ldc);

rocblas_status rocblas_cher2k(rocblas_handle handle, rocblas_fill uplo,
                              rocblas_operation trans, rocblas_int n,
                              rocblas_int k, const rocblas_float_complex *alpha,
                              const rocblas_float_complex *A, rocblas_int lda,
                              const rocblas_float_complex *B, rocblas_int ldb,
                              const float *beta, rocblas_float_complex *C,
                              rocblas_int ldc) {
  if (f_rocblas_cher2k)
    return (f_rocblas_cher2k)(handle, uplo, trans, n, k, alpha, A, lda, B, ldb,
                              beta, C, ldc);
  return -1;
}

extern void (*f_cblas_cher2k)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                              CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                              const CBLAS_INT K, const void *alpha,
                              const void *A, const CBLAS_INT lda, const void *B,
                              const CBLAS_INT ldb, const float beta, void *C,
                              const CBLAS_INT ldc);

static void _cblas_cher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                          CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                          const CBLAS_INT K, const void *alpha, const void *A,
                          const CBLAS_INT lda, const void *B,
                          const CBLAS_INT ldb, const float beta, void *C,
                          const CBLAS_INT ldc) {
  if (f_cblas_cher2k)
    (f_cblas_cher2k)(layout, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

/* no difference wrt layout in cblas_cher2k.c */
void cblas_cher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                  const CBLAS_INT N, const CBLAS_INT K, const void *alpha,
                  const void *A, const CBLAS_INT lda, const void *B,
                  const CBLAS_INT ldb, const float beta, void *C,
                  const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_her2k(Uplo, Trans, N, K, lda, ldb, ldc, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  rocblas_cher2k(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                 alpha, __A, lda, __B, ldb, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cher2k(layout, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_zher2k)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_int n, rocblas_int k, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb, const double *beta,
    rocblas_double_complex *C, rocblas_int ldc);

rocblas_status rocblas_zher2k(rocblas_handle handle, rocblas_fill uplo,
                              rocblas_operation trans, rocblas_int n,
                              rocblas_int k,
                              const rocblas_double_complex *alpha,
                              const rocblas_double_complex *A, rocblas_int lda,
                              const rocblas_double_complex *B, rocblas_int ldb,
                              const double *beta, rocblas_double_complex *C,
                              rocblas_int ldc) {
  if (f_rocblas_zher2k)
    return (f_rocblas_zher2k)(handle, uplo, trans, n, k, alpha, A, lda, B, ldb,
                              beta, C, ldc);
  return -1;
}

extern void (*f_cblas_zher2k)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                              CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                              const CBLAS_INT K, const void *alpha,
                              const void *A, const CBLAS_INT lda, const void *B,
                              const CBLAS_INT ldb, const double beta, void *C,
                              const CBLAS_INT ldc);

static void _cblas_zher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                          CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                          const CBLAS_INT K, const void *alpha, const void *A,
                          const CBLAS_INT lda, const void *B,
                          const CBLAS_INT ldb, const double beta, void *C,
                          const CBLAS_INT ldc) {
  if (f_cblas_zher2k)
    (f_cblas_zher2k)(layout, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

/* no difference wrt layout in cblas_zher2k.c */
void cblas_zher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                  const CBLAS_INT N, const CBLAS_INT K, const void *alpha,
                  const void *A, const CBLAS_INT lda, const void *B,
                  const CBLAS_INT ldb, const double beta, void *C,
                  const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  s = size_her2k(Uplo, Trans, N, K, lda, ldb, ldc, &size_a, &size_b, &size_c);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  rocblas_zher2k(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                 alpha, __A, lda, __B, ldb, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zher2k(layout, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
