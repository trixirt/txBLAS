/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_syrk(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
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

extern rocblas_status (*f_rocblas_ssyrk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const float *alpha, const float *A,
    rocblas_int lda, const float *beta, float *C, rocblas_int ldc);
rocblas_status rocblas_ssyrk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const float *alpha, const float *A,
                             rocblas_int lda, const float *beta, float *C,
                             rocblas_int ldc) {
  if (f_rocblas_ssyrk)
    return (f_rocblas_ssyrk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_ssyrk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const float alpha,
                             const float *A, const CBLAS_INT lda,
                             const float beta, float *C, const CBLAS_INT ldc);
static void _cblas_ssyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const float alpha, const float *A,
                         const CBLAS_INT lda, const float beta, float *C,
                         const CBLAS_INT ldc) {
  if (f_cblas_ssyrk)
    (f_cblas_ssyrk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_ssyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const float alpha,
                 const float *A, const CBLAS_INT lda, const float beta,
                 float *C, const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(float) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_ssyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);
  else
    rocblas_ssyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_dsyrk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const double *alpha, const double *A,
    rocblas_int lda, const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_dsyrk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const double *alpha,
                             const double *A, rocblas_int lda,
                             const double *beta, double *C, rocblas_int ldc) {
  if (f_rocblas_dsyrk)
    return (f_rocblas_dsyrk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_dsyrk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const double alpha,
                             const double *A, const CBLAS_INT lda,
                             const double beta, double *C, const CBLAS_INT ldc);
static void _cblas_dsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const double alpha, const double *A,
                         const CBLAS_INT lda, const double beta, double *C,
                         const CBLAS_INT ldc) {
  if (f_cblas_dsyrk)
    (f_cblas_dsyrk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_dsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const double alpha,
                 const double *A, const CBLAS_INT lda, const double beta,
                 double *C, const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(double) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_dsyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);
  else
    rocblas_dsyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  &alpha, __A, lda, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_csyrk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *beta, rocblas_float_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_csyrk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C, rocblas_int ldc) {
  if (f_rocblas_csyrk)
    return (f_rocblas_csyrk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_csyrk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const void *alpha,
                             const void *A, const CBLAS_INT lda,
                             const void *beta, void *C, const CBLAS_INT ldc);
static void _cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *beta, void *C,
                         const CBLAS_INT ldc) {
  if (f_cblas_csyrk)
    (f_cblas_csyrk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *beta, void *C,
                 const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_csyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  alpha, __A, lda, beta, __C, ldc);
  else
    rocblas_csyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  alpha, __A, lda, beta, __C, ldc);

  hipMemcpy(C, __C, 2 * sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_csyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_zsyrk)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA,
    rocblas_int n, rocblas_int k, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *beta, rocblas_double_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_zsyrk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation transA, rocblas_int n,
                             rocblas_int k, const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C, rocblas_int ldc) {
  if (f_rocblas_zsyrk)
    return (f_rocblas_zsyrk)(handle, uplo, transA, n, k, alpha, A, lda, beta, C,
                             ldc);
  return -1;
}

extern void (*f_cblas_zsyrk)(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                             const CBLAS_INT K, const void *alpha,
                             const void *A, const CBLAS_INT lda,
                             const void *beta, void *C, const CBLAS_INT ldc);
static void _cblas_zsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, const CBLAS_INT N,
                         const CBLAS_INT K, const void *alpha, const void *A,
                         const CBLAS_INT lda, const void *beta, void *C,
                         const CBLAS_INT ldc) {
  if (f_cblas_zsyrk)
    (f_cblas_zsyrk)(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

void cblas_zsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *beta, void *C,
                 const CBLAS_INT ldc) {

  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  else
    s = size_syrk(Uplo, Trans, N, K, lda, ldc, &size_a, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);
  if (layout == CblasColMajor)
    rocblas_zsyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  alpha, __A, lda, beta, __C, ldc);
  else
    rocblas_zsyrk(__handle, (rocblas_fill)Uplo, (rocblas_operation)Trans, N, K,
                  alpha, __A, lda, beta, __C, ldc);

  hipMemcpy(C, __C, 2 * sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}
