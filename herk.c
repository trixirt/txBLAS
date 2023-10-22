/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_herk(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                     const CBLAS_INT K, const CBLAS_INT lda,
                     const CBLAS_INT ldc, size_t *size_a, size_t *size_c) {

  if (Trans == CblasNoTrans)
    *size_a = K * lda;
  else
    *size_a = N * lda;
  *size_c = N * ldc;

  if (*size_a > __mem_max_dim || *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_cherk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const float *alpha,
    const rocblas_float_complex *A, rocblas_int lda, const float *beta,
    rocblas_float_complex *C, rocblas_int ldc);
rocblas_status rocblas_cherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const float *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const float *beta, rocblas_float_complex *C,
                             rocblas_int ldc) {
  if (*f_rocblas_cherk)
    return (f_rocblas_cherk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_cherk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const float alpha,
                             const void *A, const CBLAS_INT lda,
                             const float beta, void *C, const CBLAS_INT ldc);
static void _cblas_cherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const float alpha, const void *A,
                         const CBLAS_INT lda, const float beta, void *C,
                         const CBLAS_INT ldc) {
  if (*f_cblas_cherk)
    (f_cblas_cherk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_cherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const float alpha,
                 const void *A, const CBLAS_INT lda, const float beta, void *C,
                 const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_herk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_herk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_cherk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);
  else
    rocblas_cherk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);

  hipMemcpy(C, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cherk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_zherk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const double *alpha,
    const rocblas_double_complex *A, rocblas_int lda, const double *beta,
    rocblas_double_complex *C, rocblas_int ldc);
rocblas_status rocblas_zherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const double *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const double *beta, rocblas_double_complex *C,
                             rocblas_int ldc) {
  if (*f_rocblas_zherk)
    return (f_rocblas_zherk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_zherk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const double alpha,
                             const void *A, const CBLAS_INT lda,
                             const double beta, void *C, const CBLAS_INT ldc);
static void _cblas_zherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const double alpha, const void *A,
                         const CBLAS_INT lda, const double beta, void *C,
                         const CBLAS_INT ldc) {
  if (*f_cblas_zherk)
    (f_cblas_zherk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_zherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const double alpha,
                 const void *A, const CBLAS_INT lda, const double beta, void *C,
                 const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_herk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_herk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_zherk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);
  else
    rocblas_zherk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);

  hipMemcpy(C, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zherk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}
