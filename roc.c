#include "internal/roc.h"

extern rocblas_status (*f_rocblas_create_handle)(rocblas_handle *handle);
rocblas_status rocblas_create_handle(rocblas_handle *handle) {
  if (*f_rocblas_create_handle)
    return (*f_rocblas_create_handle)(handle);
  return -1;
}

extern rocblas_status (*f_rocblas_destroy_handle)(rocblas_handle handle);
rocblas_status rocblas_destroy_handle(rocblas_handle handle) {
  if (*f_rocblas_destroy_handle)
    return (*f_rocblas_destroy_handle)(handle);
  return -1;
}

rocblas_handle __handle;
void roc_init() { rocblas_create_handle(&__handle); }
