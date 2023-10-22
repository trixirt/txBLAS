/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_tbsv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                     const CBLAS_INT N, const CBLAS_INT K, const CBLAS_INT lda,
                     const CBLAS_INT incX, size_t *size_a, size_t *size_b) {

  *size_a = lda * N;
  *size_b = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_stbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, rocblas_int m, rocblas_int k, const float *A,
    rocblas_int lda, float *x, rocblas_int incx);

rocblas_status rocblas_stbsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int m, rocblas_int k, const float *A,
                             rocblas_int lda, float *x, rocblas_int incx) {
  if (*f_rocblas_stbsv)
    return (f_rocblas_stbsv)(handle, uplo, trans, diag, m, k, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_stbsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const float *A, const CBLAS_INT lda, float *X,
                             const CBLAS_INT incX);

static void _cblas_stbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const CBLAS_INT K, const float *A,
                         const CBLAS_INT lda, float *X, const CBLAS_INT incX) {
  if (*f_cblas_stbsv)
    (f_cblas_stbsv)(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

void cblas_stbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K,
                 const float *A, const CBLAS_INT lda, float *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tbsv(Uplo, TransA, Diag, N, K, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_stbsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, K, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_stbsv(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_dtbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, rocblas_int m, rocblas_int k, const double *A,
    rocblas_int lda, double *x, rocblas_int incx);

rocblas_status rocblas_dtbsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int m, rocblas_int k, const double *A,
                             rocblas_int lda, double *x, rocblas_int incx) {
  if (*f_rocblas_dtbsv)
    return (f_rocblas_dtbsv)(handle, uplo, trans, diag, m, k, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_dtbsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const double *A, const CBLAS_INT lda, double *X,
                             const CBLAS_INT incX);

static void _cblas_dtbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const CBLAS_INT K, const double *A,
                         const CBLAS_INT lda, double *X, const CBLAS_INT incX) {
  if (*f_cblas_dtbsv)
    (f_cblas_dtbsv)(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

void cblas_dtbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K,
                 const double *A, const CBLAS_INT lda, double *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tbsv(Uplo, TransA, Diag, N, K, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dtbsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, K, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dtbsv(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ctbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, rocblas_int m, rocblas_int k,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *x,
    rocblas_int incx);

rocblas_status rocblas_ctbsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int m, rocblas_int k,
                             const rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *x, rocblas_int incx) {
  if (*f_rocblas_ctbsv)
    return (f_rocblas_ctbsv)(handle, uplo, trans, diag, m, k, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ctbsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *A, const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);

static void _cblas_ctbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const CBLAS_INT K, const void *A,
                         const CBLAS_INT lda, void *X, const CBLAS_INT incX) {
  if (*f_cblas_ctbsv)
    (f_cblas_ctbsv)(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

void cblas_ctbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K,
                 const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tbsv(Uplo, TransA, Diag, N, K, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ctbsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, K, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ctbsv(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ztbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, rocblas_int m, rocblas_int k,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *x,
    rocblas_int incx);

rocblas_status rocblas_ztbsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, rocblas_diagonal diag,
                             rocblas_int m, rocblas_int k,
                             const rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *x, rocblas_int incx) {
  if (*f_rocblas_ztbsv)
    return (f_rocblas_ztbsv)(handle, uplo, trans, diag, m, k, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ztbsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const CBLAS_INT K,
                             const void *A, const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);

static void _cblas_ztbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const CBLAS_INT K, const void *A,
                         const CBLAS_INT lda, void *X, const CBLAS_INT incX) {
  if (*f_cblas_ztbsv)
    (f_cblas_ztbsv)(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

void cblas_ztbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const CBLAS_INT K,
                 const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tbsv(Uplo, TransA, Diag, N, K, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_ztbsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, K, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ztbsv(layout, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}
