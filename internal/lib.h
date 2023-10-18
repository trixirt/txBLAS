/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#ifndef RBLAS_LIB_H
#define RBLAS_LIB_H

#include <dlfcn.h>
#include <stddef.h>

/* interface */
void lib_init();
void lib_fini();

extern void * __rocblas_library;
extern void * __amdhip64_library;

#endif
