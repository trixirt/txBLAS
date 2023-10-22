/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_trsv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                     const CBLAS_INT N, const CBLAS_INT lda,
                     const CBLAS_INT incX, size_t *size_a, size_t *size_b) {

  *size_a = lda * N;
  *size_b = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_strsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const float *A, rocblas_int lda,
                                         float *x, rocblas_int incx);
rocblas_status rocblas_strsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const float *A, rocblas_int lda,
                             float *x, rocblas_int incx) {
  if (*f_rocblas_strsv)
    return (f_rocblas_strsv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_strsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const float *A,
                             const CBLAS_INT lda, float *X,
                             const CBLAS_INT incX);
static void _cblas_strsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const float *A, const CBLAS_INT lda,
                         float *X, const CBLAS_INT incX) {
  if (*f_cblas_strsv)
    (f_cblas_strsv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_strsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const float *A,
                 const CBLAS_INT lda, float *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trsv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_strsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_strsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_dtrsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const double *A, rocblas_int lda,
                                         double *x, rocblas_int incx);
rocblas_status rocblas_dtrsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const double *A, rocblas_int lda,
                             double *x, rocblas_int incx) {
  if (*f_rocblas_dtrsv)
    return (f_rocblas_dtrsv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_dtrsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const double *A,
                             const CBLAS_INT lda, double *X,
                             const CBLAS_INT incX);
static void _cblas_dtrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const double *A,
                         const CBLAS_INT lda, double *X, const CBLAS_INT incX) {
  if (*f_cblas_dtrsv)
    (f_cblas_dtrsv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_dtrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const double *A,
                 const CBLAS_INT lda, double *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trsv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dtrsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ctrsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *x, rocblas_int incx);
rocblas_status rocblas_ctrsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_float_complex *A,
                             rocblas_int lda, rocblas_float_complex *x,
                             rocblas_int incx) {
  if (*f_rocblas_ctrsv)
    return (f_rocblas_ctrsv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ctrsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *A,
                             const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);
static void _cblas_ctrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                         void *X, const CBLAS_INT incX) {
  if (*f_cblas_ctrsv)
    (f_cblas_ctrsv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_ctrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A,
                 const CBLAS_INT lda, void *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trsv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ctrsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ctrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ztrsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_ztrsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_double_complex *A,
                             rocblas_int lda, rocblas_double_complex *x,
                             rocblas_int incx) {
  if (*f_rocblas_ztrsv)
    return (f_rocblas_ztrsv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ztrsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *A,
                             const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);
static void _cblas_ztrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                         void *X, const CBLAS_INT incX) {
  if (*f_cblas_ztrsv)
    (f_cblas_ztrsv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_ztrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A,
                 const CBLAS_INT lda, void *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trsv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_ztrsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ztrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}
