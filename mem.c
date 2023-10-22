#include "internal/mem.h"
#include <hip/hip_runtime_api.h>

void *__A = NULL;
void *__B = NULL;
void *__C = NULL;
/* 4k x 4k */
const size_t __mem_max_dim = 16777216;
/* 4x x 4k * 2 * sizeof(double) */
const size_t __mem_max_size = 268435456;

extern hipError_t (*f_hipFree)(void *ptr);
hipError_t hipFree(void *ptr) {
  if (*f_hipFree)
    return (f_hipFree)(ptr);
  return -1;
}

extern hipError_t (*f_hipMalloc)(void **, size_t);
hipError_t hipMalloc(void **p, size_t a) {
  if (*f_hipMalloc)
    return (f_hipMalloc)(p, a);
  return -1;
}

extern hipError_t (*f_hipMemcpy)(void *dst, const void *src, size_t sizeBytes,
                                 hipMemcpyKind kind);
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  if (*f_hipMemcpy)
    return (f_hipMemcpy)(dst, src, sizeBytes, kind);
  return -1;
}

extern hipError_t (*f_hipMemset)(void *dst, int value, size_t sizeBytes);
hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
  if (*f_hipMemset)
    return (f_hipMemset)(dst, value, sizeBytes);
  return -1;
}

void mem_init() {
  hipMalloc(&__A, __mem_max_size);
  hipMalloc(&__B, __mem_max_size);
  hipMalloc(&__C, __mem_max_size);
}
