/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_tpsv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                     const CBLAS_INT N, const CBLAS_INT incX, size_t *size_a,
                     size_t *size_b) {

  *size_a = ((N * (N + 1)) / 2);
  *size_b = (1 + (N - 1) * abs(incX));

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim)
    return 1;
  return 0;
}

extern rocblas_status (*f_rocblas_stpsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const float *A, float *x,
                                         rocblas_int incx);

rocblas_status rocblas_stpsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const float *A, float *x,
                             rocblas_int incx) {
  if (*f_rocblas_stpsv)
    return (f_rocblas_stpsv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_stpsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const float *Ap, float *X,
                             const CBLAS_INT incX);

static void _cblas_stpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const float *Ap, float *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_stpsv)
    (f_cblas_stpsv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_stpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const float *A, float *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpsv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_stpsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_stpsv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_dtpsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal diag, rocblas_int m,
                                         const double *A, double *x,
                                         rocblas_int incx);

rocblas_status rocblas_dtpsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const double *A, double *x,
                             rocblas_int incx) {
  if (*f_rocblas_dtpsv)
    return (f_rocblas_dtpsv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_dtpsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const double *Ap, double *X,
                             const CBLAS_INT incX);

static void _cblas_dtpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const double *Ap, double *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_dtpsv)
    (f_cblas_dtpsv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_dtpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const double *A, double *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpsv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_dtpsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dtpsv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_ctpsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex *A,
    rocblas_float_complex *x, rocblas_int incx);

rocblas_status rocblas_ctpsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_float_complex *A,
                             rocblas_float_complex *x, rocblas_int incx) {
  if (*f_rocblas_ctpsv)
    return (f_rocblas_ctpsv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_ctpsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *Ap, void *X,
                             const CBLAS_INT incX);

static void _cblas_ctpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *Ap, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_ctpsv)
    (f_cblas_ctpsv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_ctpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpsv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);

  rocblas_ctpsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(float) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ctpsv(layout, Uplo, TransA, Diag, N, A, X, incX);
}

extern rocblas_status (*f_rocblas_ztpsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex *A,
    rocblas_double_complex *x, rocblas_int incx);

rocblas_status rocblas_ztpsv(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_diagonal diag,
                             rocblas_int m, const rocblas_double_complex *A,
                             rocblas_double_complex *x, rocblas_int incx) {
  if (*f_rocblas_ztpsv)
    return (f_rocblas_ztpsv)(handle, uplo, transA, diag, m, A, x, incx);

  return -1;
}

extern void (*f_cblas_ztpsv)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                             const CBLAS_INT N, const void *Ap, void *X,
                             const CBLAS_INT incX);

static void _cblas_ztpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                         const CBLAS_INT N, const void *Ap, void *X,
                         const CBLAS_INT incX) {
  if (*f_cblas_ztpsv)
    (f_cblas_ztpsv)(layout, Uplo, TransA, Diag, N, Ap, X, incX);
}

void cblas_ztpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT N, const void *A, void *X,
                 const CBLAS_INT incX) {

  size_t size_a, size_b;
  int s = size_tpsv(Uplo, TransA, Diag, N, incX, &size_a, &size_b);

  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, X, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);

  rocblas_ztpsv(__handle, (rocblas_fill)Uplo, (rocblas_operation)TransA,
                (rocblas_diagonal)Diag, N, __A, __B, incX);

  hipMemcpy(X, __B, 2 * sizeof(double) * size_b, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ztpsv(layout, Uplo, TransA, Diag, N, A, X, incX);
}
