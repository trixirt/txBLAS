/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_symm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, const CBLAS_INT M,
                     const CBLAS_INT N, const CBLAS_INT lda,
                     const CBLAS_INT ldb, const CBLAS_INT ldc, size_t *size_a,
                     size_t *size_b, size_t *size_c) {

  if (Side == CblasLeft)
    *size_a = lda * M;
  else
    *size_a = lda * N;
  *size_b = ldb * N;
  *size_c = ldc * N;

  if (*size_a > __mem_max_dim || *size_b > __mem_max_dim ||
      *size_c > __mem_max_dim)
    return 1;

  return 0;
}

extern rocblas_status (*f_rocblas_ssymm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_int m, rocblas_int n,
                                         const float *alpha, const float *A,
                                         rocblas_int lda, const float *B,
                                         rocblas_int ldb, const float *beta,
                                         float *C, rocblas_int ldc);
rocblas_status rocblas_ssymm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const float *alpha, const float *A,
                             rocblas_int lda, const float *B, rocblas_int ldb,
                             const float *beta, float *C, rocblas_int ldc) {
  if (*f_rocblas_ssymm)
    return (f_rocblas_ssymm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_ssymm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const float alpha,
                             const float *A, const CBLAS_INT lda,
                             const float *B, const CBLAS_INT ldb,
                             const float beta, float *C, const CBLAS_INT ldc);
static void _cblas_ssymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const float alpha, const float *A, const CBLAS_INT lda,
                         const float *B, const CBLAS_INT ldb, const float beta,
                         float *C, const CBLAS_INT ldc) {
  if (*f_cblas_ssymm)
    (f_cblas_ssymm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void cblas_ssymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const float alpha,
                 const float *A, const CBLAS_INT lda, const float *B,
                 const CBLAS_INT ldb, const float beta, float *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_symm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_symm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_ssymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N,
                  &alpha, __A, lda, __B, ldb, &beta, __C, ldc);
  else
    rocblas_ssymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M,
                  &alpha, __A, lda, __B, ldb, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_ssymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_dsymm)(rocblas_handle handle,
                                         rocblas_side side, rocblas_fill uplo,
                                         rocblas_int m, rocblas_int n,
                                         const double *alpha, const double *A,
                                         rocblas_int lda, const double *B,
                                         rocblas_int ldb, const double *beta,
                                         double *C, rocblas_int ldc);
rocblas_status rocblas_dsymm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const double *alpha, const double *A,
                             rocblas_int lda, const double *B, rocblas_int ldb,
                             const double *beta, double *C, rocblas_int ldc) {
  if (*f_rocblas_dsymm)
    return (f_rocblas_dsymm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_dsymm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const double alpha,
                             const double *A, const CBLAS_INT lda,
                             const double *B, const CBLAS_INT ldb,
                             const double beta, double *C, const CBLAS_INT ldc);
static void _cblas_dsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const double alpha, const double *A,
                         const CBLAS_INT lda, const double *B,
                         const CBLAS_INT ldb, const double beta, double *C,
                         const CBLAS_INT ldc) {
  if (*f_cblas_dsymm)
    (f_cblas_dsymm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void rblas_dsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const double alpha,
                 const double *A, const CBLAS_INT lda, const double *B,
                 const CBLAS_INT ldb, const double beta, double *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_symm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_symm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_dsymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N,
                  &alpha, __A, lda, __B, ldb, &beta, __C, ldc);
  else
    rocblas_dsymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M,
                  &alpha, __A, lda, __B, ldb, &beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_dsymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_csymm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
    rocblas_int n, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta, rocblas_float_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_csymm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C, rocblas_int ldc) {
  if (*f_rocblas_csymm)
    return (f_rocblas_csymm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_csymm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_csymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (*f_cblas_csymm)
    (f_cblas_csymm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void cblas_csymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *B,
                 const CBLAS_INT ldb, const void *beta, void *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_symm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_symm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_csymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);
  else
    rocblas_csymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_csymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_zsymm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
    rocblas_int n, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta, rocblas_double_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_zsymm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C, rocblas_int ldc) {
  if (*f_rocblas_zsymm)
    return (f_rocblas_zsymm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_zsymm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_zsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (*f_cblas_zsymm)
    (f_cblas_zsymm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void cblas_zsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *B,
                 const CBLAS_INT ldb, const void *beta, void *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_symm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_symm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_zsymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);
  else
    rocblas_zsymm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zsymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}
