#include <rocblas/rocblas.h>

extern rocblas_status (*f_rocblas_sgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const float *alpha,
    const float *A, rocblas_int lda, const float *B, rocblas_int ldb,
    const float *beta, float *C, rocblas_int ldc);

rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k, const float *alpha,
                             const float *A, rocblas_int lda, const float *B,
                             rocblas_int ldb, const float *beta, float *C,
                             rocblas_int ldc) {
  if (f_rocblas_sgemm)
    return (*f_rocblas_sgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}

extern rocblas_status (*f_rocblas_dgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const double *alpha,
    const double *A, rocblas_int lda, const double *B, rocblas_int ldb,
    const double *beta, double *C, rocblas_int ldc);
rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k, const double *alpha,
                             const double *A, rocblas_int lda, const double *B,
                             rocblas_int ldb, const double *beta, double *C,
                             rocblas_int ldc) {
  if (f_rocblas_dgemm)
    return (*f_rocblas_dgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}

extern rocblas_status (*f_rocblas_cgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha, const rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta, rocblas_float_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_cgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A, rocblas_int lda,
                             const rocblas_float_complex *B, rocblas_int ldb,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C, rocblas_int ldc) {
  if (f_rocblas_cgemm)
    (*f_rocblas_cgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
  return -1;
}

extern rocblas_status (*f_rocblas_zgemm)(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha, const rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta, rocblas_double_complex *C,
    rocblas_int ldc);
rocblas_status rocblas_zgemm(rocblas_handle handle, rocblas_operation transA,
                             rocblas_operation transB, rocblas_int m,
                             rocblas_int n, rocblas_int k,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A, rocblas_int lda,
                             const rocblas_double_complex *B, rocblas_int ldb,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C, rocblas_int ldc) {
  if (f_rocblas_zgemm)
    return (*f_rocblas_zgemm)(handle, transA, transB, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  return -1;
}
