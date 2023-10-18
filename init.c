/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/func.h"
#include "internal/lib.h"

extern void roc_init();

__attribute__((constructor)) void rblas_init() {
  lib_init();
  func_init();
  roc_init();
}

__attribute__((destructor)) void rblas_fini() { lib_fini(); }
