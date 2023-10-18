/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#ifndef RBLAS_MEM_H
#define RBLAS_MEM_H

#include <stddef.h>
#include <stdint.h>
#include <hip/hip_runtime_api.h>

/* interface */
void mem_init();

extern const size_t __mem_max_dim;
extern const size_t __mem_max_size;

extern void * __A;
extern void * __B;
extern void * __C;

#endif
