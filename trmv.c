/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_trmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                     const CBLAS_INT N, const CBLAS_INT lda,
                     const CBLAS_INT incX, size_t *size_a, size_t *size_b) {

  *size_a = lda * N;
  *size_b = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_strmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const float *A, rocblas_int lda,
                                         float *x, rocblas_int incx);
rocblas_status rocblas_strmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const float *A, rocblas_int lda,
                             float *x, rocblas_int incx) {
  if (*f_rocblas_strmv)
    return (f_rocblas_strmv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_strmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const float *A,
                             const CBLAS_INT lda, float *X,
                             const CBLAS_INT incX);
static void _cblas_strmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const float *A, const CBLAS_INT lda,
                         float *X, const CBLAS_INT incX) {
  if (*f_cblas_strmv)
    (f_cblas_strmv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_strmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const float *A,
                 const CBLAS_INT lda, float *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trmv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_strmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_strmv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_dtrmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const double *A, rocblas_int lda,
                                         double *x, rocblas_int incx);
rocblas_status rocblas_dtrmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const double *A, rocblas_int lda,
                             double *x, rocblas_int incx) {
  if (*f_rocblas_dtrmv)
    return (f_rocblas_dtrmv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_dtrmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const double *A,
                             const CBLAS_INT lda, double *X,
                             const CBLAS_INT incX);
static void _cblas_dtrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const double *A,
                         const CBLAS_INT lda, double *X, const CBLAS_INT incX) {
  if (*f_cblas_dtrmv)
    (f_cblas_dtrmv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_dtrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const double *A,
                 const CBLAS_INT lda, double *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trmv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dtrmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dtrmv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ctrmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *x, rocblas_int incx);
rocblas_status rocblas_ctrmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_float_complex *A,
                             rocblas_int lda, rocblas_float_complex *x,
                             rocblas_int incx) {
  if (*f_rocblas_ctrmv)
    return (f_rocblas_ctrmv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ctrmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *A,
                             const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);
static void _cblas_ctrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                         void *X, const CBLAS_INT incX) {
  if (*f_cblas_ctrmv)
    (f_cblas_ctrmv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_ctrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A,
                 const CBLAS_INT lda, void *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trmv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ctrmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ctrmv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

extern rocblas_status (*f_rocblas_ztrmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *x, rocblas_int incx);
rocblas_status rocblas_ztrmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_double_complex *A,
                             rocblas_int lda, rocblas_double_complex *x,
                             rocblas_int incx) {
  if (*f_rocblas_ztrmv)
    return (f_rocblas_ztrmv)(handle, uplo, transA, diag, m, A, lda, x, incx);
  return -1;
}

extern void (*f_cblas_ztrmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *A,
                             const CBLAS_INT lda, void *X,
                             const CBLAS_INT incX);
static void _cblas_ztrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                         void *X, const CBLAS_INT incX) {
  if (*f_cblas_ztrmv)
    (f_cblas_ztrmv)(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void cblas_ztrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A,
                 const CBLAS_INT lda, void *X, const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_trmv(Uplo, TransA, Diag, N, lda, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_ztrmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, lda, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ztrmv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);
}
