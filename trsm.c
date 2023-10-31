/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_trsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
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

extern rocblas_status (*f_rocblas_strsm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         rocblas_int n, const float *alpha,
                                         const float *A, rocblas_int lda,
                                         float *B, rocblas_int ldb);

rocblas_status rocblas_strsm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const float *alpha, const float *A,
                             rocblas_int lda, float *B, rocblas_int ldb) {
  if (*f_rocblas_strsm)
    return rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_strsm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const float alpha,
                             const float *A, const CBLAS_INT lda, float *B,
                             const CBLAS_INT ldb);

static void _cblas_strsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         float *B, const CBLAS_INT ldb) {
  if (*f_cblas_strsm)
    (f_cblas_strsm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_strsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const float alpha, const float *A,
                 const CBLAS_INT lda, float *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trsm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trsm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(float) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_strsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  &alpha, __A, lda, __B, ldb);
  else
    rocblas_strsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  &alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_strsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_dtrsm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         rocblas_int n, const double *alpha,
                                         const double *A, rocblas_int lda,
                                         double *B, rocblas_int ldb);

rocblas_status rocblas_dtrsm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const double *alpha,
                             const double *A, rocblas_int lda, double *B,
                             rocblas_int ldb) {
  if (*f_rocblas_dtrsm)
    return rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_dtrsm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const double alpha,
                             const double *A, const CBLAS_INT lda, double *B,
                             const CBLAS_INT ldb);

static void _cblas_dtrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, double *B, const CBLAS_INT ldb) {
  if (*f_cblas_dtrsm)
    (f_cblas_dtrsm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_dtrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const double alpha, const double *A,
                 const CBLAS_INT lda, double *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trsm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trsm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(double) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dtrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  &alpha, __A, lda, __B, ldb);
  else
    rocblas_dtrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  &alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_ctrsm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *B,
    rocblas_int ldb);

rocblas_status rocblas_ctrsm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *B, rocblas_int ldb) {
  if (*f_rocblas_ctrsm)
    return rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_ctrsm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, void *B,
                             const CBLAS_INT ldb);

static void _cblas_ctrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         void *B, const CBLAS_INT ldb) {
  if (*f_cblas_ctrsm)
    (f_cblas_ctrsm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_ctrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, void *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trsm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trsm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_ctrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  alpha, __A, lda, __B, ldb);
  else
    rocblas_ctrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_ctrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

extern rocblas_status (*f_rocblas_ztrsm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *B,
    rocblas_int ldb);

rocblas_status rocblas_ztrsm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_operation transA,
                             rocblas_diagonal diag, rocblas_int m,
                             rocblas_int n, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *B, rocblas_int ldb) {
  if (*f_rocblas_ztrsm)
    return rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda,
                         B, ldb);
  return -1;
}

extern void (*f_cblas_ztrsm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                             CBLAS_DIAG Diag, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, void *B,
                             const CBLAS_INT ldb);

static void _cblas_ztrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         void *B, const CBLAS_INT ldb) {
  if (*f_cblas_ztrsm)
    (f_cblas_ztrsm)(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B,
                    ldb);
}

void cblas_ztrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, const CBLAS_INT M,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, void *B, const CBLAS_INT ldb) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_trsm(Side, Uplo, TransA, Diag, M, N, lda, ldb, &size_a, &size_b);
  else
    s = size_trsm(Side, Uplo, TransA, Diag, N, M, lda, ldb, &size_a, &size_b);

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_ztrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, M, N,
                  alpha, __A, lda, __B, ldb);
  else
    rocblas_ztrsm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo,
                  (rocblas_operation)TransA, (rocblas_diagonal)Diag, N, M,
                  alpha, __A, lda, __B, ldb);

  hipMemcpy(B, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  if (s)
    goto fail;

  return;
fail:
  _cblas_ztrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}
