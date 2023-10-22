/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/mem.h"
#include "internal/roc.h"

static int size_hemm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, const CBLAS_INT M,
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

extern rocblas_status (*f_rocblas_chemm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
    rocblas_int n, const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta, rocblas_float_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_chemm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C, rocblas_int ldc) {
  if (*f_rocblas_chemm)
    return (f_rocblas_chemm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_chemm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_chemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (*f_cblas_chemm)
    (f_cblas_chemm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void cblas_chemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *B,
                 const CBLAS_INT ldb, const void *beta, void *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_hemm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_hemm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(float) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(float) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(float) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_chemm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);
  else
    rocblas_chemm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(float) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_chemm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern rocblas_status (*f_rocblas_zhemm)(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m,
    rocblas_int n, const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta, rocblas_double_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_zhemm(rocblas_handle handle, rocblas_side side,
                             rocblas_fill uplo, rocblas_int m, rocblas_int n,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C, rocblas_int ldc) {
  if (*f_rocblas_zhemm)
    return (f_rocblas_zhemm)(handle, side, uplo, m, n, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  return -1;
}

extern void (*f_cblas_zhemm)(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                             CBLAS_UPLO Uplo, const CBLAS_INT M,
                             const CBLAS_INT N, const void *alpha,
                             const void *A, const CBLAS_INT lda, const void *B,
                             const CBLAS_INT ldb, const void *beta, void *C,
                             const CBLAS_INT ldc);
static void _cblas_zhemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         const CBLAS_INT M, const CBLAS_INT N,
                         const void *alpha, const void *A, const CBLAS_INT lda,
                         const void *B, const CBLAS_INT ldb, const void *beta,
                         void *C, const CBLAS_INT ldc) {
  if (*f_cblas_zhemm)
    (f_cblas_zhemm)(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C,
                    ldc);
}

void cblas_zhemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 const CBLAS_INT M, const CBLAS_INT N, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *B,
                 const CBLAS_INT ldb, const void *beta, void *C,
                 const CBLAS_INT ldc) {
  size_t size_a, size_b, size_c;
  int s;
  if (layout == CblasColMajor)
    s = size_hemm(Side, Uplo, M, N, lda, ldb, ldc, &size_a, &size_b, &size_c);
  else
    s = size_hemm(Side, Uplo, N, M, lda, ldb, ldc, &size_a, &size_b, &size_c);
  if (s)
    goto fail;

  hipMemcpy(__A, A, 2 * sizeof(double) * size_a, hipMemcpyHostToDevice);
  hipMemcpy(__B, B, 2 * sizeof(double) * size_b, hipMemcpyHostToDevice);
  hipMemcpy(__C, C, 2 * sizeof(double) * size_c, hipMemcpyHostToDevice);

  if (layout == CblasColMajor)
    rocblas_zhemm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, M, N, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);
  else
    rocblas_zhemm(__handle, (rocblas_side)Side, (rocblas_fill)Uplo, N, M, alpha,
                  __A, lda, __B, ldb, beta, __C, ldc);

  hipMemcpy(C, __C, sizeof(double) * size_c, hipMemcpyDeviceToHost);

  return;
fail:
  _cblas_zhemm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}
