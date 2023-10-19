/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                     const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K,
                     const CBLAS_INT lda, const CBLAS_INT ldb,
                     const CBLAS_INT ldc, size_t *size_a, size_t *size_b,
                     size_t *size_c) {
  if (TransA == CblasNoTrans)
    *size_a = K * lda;
  else
    *size_a = M * lda;

  if (TransB == CblasNoTrans)
    *size_b = N * ldb;
  else
    *size_b = K * ldb;

  *size_c = N * ldc;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_sgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const float *alpha,
    const float *A, rocblas_int lda, const float *B, rocblas_int ldb,
    const float *beta, float *C, rocblas_int ldc);

rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k, const float *alpha,
                             const float *A, rocblas_int lda, const float *B,
                             rocblas_int ldb, const float *beta, float *C,
                             rocblas_int ldc) {
  if (f_rocblas_sgemm)
    return (*f_rocblas_sgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}

extern void (*f_cblas_sgemm)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const float alpha, const float *A,
                             const CBLAS_INT lda, const float *B,
                             const CBLAS_INT ldb, const float beta, float *C,
                             const CBLAS_INT ldc);
static void _cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         const float *B, const CBLAS_INT ldb, const float beta,
                         float *C, const CBLAS_INT ldc) {
  if (f_cblas_sgemm)
    (f_cblas_sgemm)(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
                 const float beta, float *C, const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_gemm(TransA, TransB, M, N, K, lda, ldb, ldc, &size_a, &size_b,
                  &size_c);
  else
    s = size_gemm(TransA, TransB, N, M, K, ldb, lda, ldc, &size_b, &size_a,
                  &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(float) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_sgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, M, N, K, &alpha, __A, lda, __B,
                  ldb, &beta, __C, ldc);
  else
    rocblas_sgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, N, M, K, &alpha, __B, ldb, __A,
                  lda, &beta, __C, ldc);
  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
}

extern rocblas_status (*f_rocblas_dgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const double *alpha,
    const double *A, rocblas_int lda, const double *B, rocblas_int ldb,
    const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k, const double *alpha,
                             const double *A, rocblas_int lda, const double *B,
                             rocblas_int ldb, const double *beta, double *C,
                             rocblas_int ldc) {
  if (f_rocblas_dgemm)
    return (*f_rocblas_dgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}

extern void (*f_cblas_dgemm)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const double alpha, const double *A,
                             const CBLAS_INT lda, const double *B,
                             const CBLAS_INT ldb, const double beta, double *C,
                             const CBLAS_INT ldc);
static void _cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, const double *B,
                         const CBLAS_INT ldb, const double beta, double *C,
                         const CBLAS_INT ldc) {
  if (f_cblas_dgemm)
    (f_cblas_dgemm)(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  if (size_gemm(TransA, TransB, M, N, K, lda, ldb, ldc, &size_a, &size_b,
                &size_c))
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(double) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_dgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, M, N, K, &alpha, __A, lda, __B,
                  ldb, &beta, __C, ldc);
  else
    rocblas_dgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, N, M, K, &alpha, __B, ldb, __A,
                  lda, &beta, __C, ldc);
  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
}

extern rocblas_status (*f_rocblas_cgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta, rocblas_float_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_cgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C, rocblas_int ldc) {
  if (f_rocblas_cgemm)
    (*f_rocblas_cgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
  return -1;
}

extern void (*f_cblas_cgemm)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (f_cblas_cgemm)
    (f_cblas_cgemm)(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  if (size_gemm(TransA, TransB, M, N, K, lda, ldb, ldc, &size_a, &size_b,
                &size_c))
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_cgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, M, N, K, alpha, __A, lda, __B, ldb,
                  beta, __C, ldc);
  else
    rocblas_cgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, N, M, K, alpha, __B, ldb, __A, lda,
                  beta, __C, ldc);
  hipMemcpy(C, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_cgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
}

extern rocblas_status (*f_rocblas_zgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta, rocblas_double_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_zgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C, rocblas_int ldc) {
  if (f_rocblas_zgemm)
    return (*f_rocblas_zgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}

extern void (*f_cblas_zgemm)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *alpha, const void *A,
                             const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                         CBLAS_TRANSPOSE TransB, const CBLAS_INT M,
                         const CBLAS_INT N, const CBLAS_INT K,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (f_cblas_zgemm)
    (f_cblas_zgemm)(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

void cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  if (size_gemm(TransA, TransB, M, N, K, lda, ldb, ldc, &size_a, &size_b,
                &size_c))
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_zgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, M, N, K, alpha, __A, lda, __B, ldb,
                  beta, __C, ldc);
  else
    rocblas_zgemm(__handle, (rocblas_operation)TransA,
                  (rocblas_operation)TransB, N, M, K, alpha, __B, ldb, __A, lda,
                  beta, __C, ldc);
  hipMemcpy(C, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
}
