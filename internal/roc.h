/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#ifndef RBLAS_ROC_H
#define RBLAS_ROC_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <rocblas/rocblas.h>
#include <cblas.h>

/* interface */
void roc_init();

extern rocblas_handle __handle;

#endif
