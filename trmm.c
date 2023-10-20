/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_trmm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                     CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                     const CBLAS_INT lda, const CBLAS_INT ldb, size_t *size_a,
                     size_t *size_b) {

  if (Side == CblasLeft)
    *size_a = M * lda;
  else
    *size_a = N * lda;
  *size_b = N * ldb;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_strmm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         rocblas_int n, const float *alpha,
                                         const float *A, rocblas_int lda,
                                         float *B, rocblas_int ldb);

rocblas_status rocblas_strmm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const float *alpha, const float *A,
                             rocblas_int lda, float *B, rocblas_int ldb) {
  if (f_rocblas_strmm)
    return rocblas_strmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_strmm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const float alpha,
                             const float *A, const CBLAS_INT lda, float *B,
                             const CBLAS_INT ldb);

static void _cblas_strmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         float *B, const CBLAS_INT ldb) {
  if (f_cblas_strmm)
    (f_cblas_strmm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_strmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const float alpha, const float *A,
                 const CBLAS_INT lda, float *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trmm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trmm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(float) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_strmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  &alpha, __A, lda, __B, ldb);
  else
    rocblas_strmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  &alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_strmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_dtrmm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         rocblas_int n, const double *alpha,
                                         const double *A, rocblas_int lda,
                                         double *B, rocblas_int ldb);

rocblas_status rocblas_dtrmm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, double *B,
                             rocblas_int ldb) {
  if (f_rocblas_dtrmm)
    return rocblas_dtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_dtrmm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const double alpha,
                             const double *A, const CBLAS_INT lda, double *B,
                             const CBLAS_INT ldb);

static void _cblas_dtrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, double *B, const CBLAS_INT ldb) {
  if (f_cblas_dtrmm)
    (f_cblas_dtrmm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_dtrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const double alpha, const double *A,
                 const CBLAS_INT lda, double *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trmm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trmm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(double) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dtrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  &alpha, __A, lda, __B, ldb);
  else
    rocblas_dtrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  &alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_dtrmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_ctrmm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *B,
    rocblas_int ldb);

rocblas_status rocblas_ctrmm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *B, rocblas_int ldb) {
  if (f_rocblas_ctrmm)
    return rocblas_ctrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_ctrmm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, void *B,
                             const CBLAS_INT ldb);

static void _cblas_ctrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         void *B, const CBLAS_INT ldb) {
  if (f_cblas_ctrmm)
    (f_cblas_ctrmm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_ctrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, void *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trmm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trmm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_ctrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  alpha, __A, lda, __B, ldb);
  else
    rocblas_ctrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_ctrmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_ztrmm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *B,
    rocblas_int ldb);

rocblas_status rocblas_ztrmm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *B, rocblas_int ldb) {
  if (f_rocblas_ztrmm)
    return rocblas_ztrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_ztrmm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, void *B,
                             const CBLAS_INT ldb);

static void _cblas_ztrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         void *B, const CBLAS_INT ldb) {
  if (f_cblas_ztrmm)
    (f_cblas_ztrmm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_ztrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, void *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trmm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trmm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_ztrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  alpha, __A, lda, __B, ldb);
  else
    rocblas_ztrmm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_ztrmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}
