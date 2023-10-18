#include <hip/hip_runtime_api.h>
#include <stddef.h>

extern hipError_t (*f_hipFree)(void *ptr);
hipError_t hipFree(void *ptr) {
  if (f_hipFree)
    return (f_hipFree)(ptr);
  return -1;
}

extern hipError_t (*f_hipMalloc)(void **, size_t);
hipError_t hipMalloc(void **p, size_t a) {
  if (f_hipMalloc)
    return (f_hipMalloc)(p, a);
  return -1;
}

extern hipError_t (*f_hipMemcpy)(void *dst, const void *src, size_t sizeBytes,
                                 hipMemcpyKind kind);
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  if (f_hipMemcpy)
    return (f_hipMemcpy)(dst, src, sizeBytes, kind);
  return -1;
}
