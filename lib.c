/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */

#include "internal/lib.h"

void *__amdhip64_library = NULL;
void *__cblas_library = NULL;
void *__rocblas_library = NULL;

void lib_init() {
  __amdhip64_library = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
  __rocblas_library = dlopen("librocblas.so", RTLD_NOW | RTLD_GLOBAL);
  __cblas_library = dlopen("libcblas.so", RTLD_NOW | RTLD_GLOBAL);
}

void lib_fini() {
  dlclose(__amdhip64_library);
  dlclose(__rocblas_library);
  dlclose(__cblas_library);
}
