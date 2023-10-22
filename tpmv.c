/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_tpmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                     const CBLAS_INT N, const CBLAS_INT incX, size_t *size_a,
                     size_t *size_b) {

  *size_a = ((N * (N + 1)) / 2);
  *size_b = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_stpmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const float *A, float *x,
                                         rocblas_int incx);

rocblas_status rocblas_stpmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const float *A, float *x,
                             rocblas_int incx) {
  if (*f_rocblas_stpmv)
    return (f_rocblas_stpmv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_stpmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const float *Ap, float *X,
                             const CBLAS_INT incX);

static void _cblas_stpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const float *Ap, float *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_stpmv)
    (f_cblas_stpmv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_stpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const float *A, float *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpmv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_stpmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_stpmv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_dtpmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const double *A, double *x,
                                         rocblas_int incx);

rocblas_status rocblas_dtpmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const double *A, double *x,
                             rocblas_int incx) {
  if (*f_rocblas_dtpmv)
    return (f_rocblas_dtpmv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_dtpmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const double *Ap, double *X,
                             const CBLAS_INT incX);

static void _cblas_dtpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const double *Ap, double *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_dtpmv)
    (f_cblas_dtpmv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_dtpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const double *A, double *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpmv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dtpmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dtpmv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_ctpmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex *A,
    rocblas_float_complex *x, rocblas_int incx);

rocblas_status rocblas_ctpmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_float_complex *A,
                             rocblas_float_complex *x, rocblas_int incx) {
  if (*f_rocblas_ctpmv)
    return (f_rocblas_ctpmv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_ctpmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *Ap, void *X,
                             const CBLAS_INT incX);

static void _cblas_ctpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *Ap, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_ctpmv)
    (f_cblas_ctpmv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_ctpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpmv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ctpmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ctpmv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_ztpmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex *A,
    rocblas_double_complex *x, rocblas_int incx);

rocblas_status rocblas_ztpmv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_double_complex *A,
                             rocblas_double_complex *x, rocblas_int incx) {
  if (*f_rocblas_ztpmv)
    return (f_rocblas_ztpmv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_ztpmv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *Ap, void *X,
                             const CBLAS_INT incX);

static void _cblas_ztpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *Ap, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_ztpmv)
    (f_cblas_ztpmv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_ztpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpmv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_ztpmv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ztpmv(layout, Uplo, TransA, Diag, N, A, X, incX);
}
