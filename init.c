/*
 * Copyright 2023 Tom Rix
 *
 * SPDX BSD-3-Clause
 */
#include "internal/func.h"
#include "internal/lib.h"
#include <dlfcn.h>

__attribute__((constructor)) void rblas_init() {
  /* libamdhip64 */
  __amdhip64_library = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);

  hipExtMallocWithFlags = dlsym(__amdhip64_library, "hipExtMallocWithFlags");
  hipFree = dlsym(__amdhip64_library, "hipFree");
  hipFreeArray = dlsym(__amdhip64_library, "hipFreeArray");
  hipFreeAsync = dlsym(__amdhip64_library, "hipFreeAsync");
  hipFreeHost = dlsym(__amdhip64_library, "hipFreeHost");
  hipFreeMipmappedArray = dlsym(__amdhip64_library, "hipFreeMipmappedArray");
  hipGraphAddMemFreeNode = dlsym(__amdhip64_library, "hipGraphAddMemFreeNode");
  hipGraphMemFreeNodeGetParams =
      dlsym(__amdhip64_library, "hipGraphMemFreeNodeGetParams");
  hipHostFree = dlsym(__amdhip64_library, "hipHostFree");
  hipHostMalloc = dlsym(__amdhip64_library, "hipHostMalloc");
  hipMalloc = dlsym(__amdhip64_library, "hipMalloc");
  hipMalloc3D = dlsym(__amdhip64_library, "hipMalloc3D");
  hipMalloc3DArray = dlsym(__amdhip64_library, "hipMalloc3DArray");
  hipMallocArray = dlsym(__amdhip64_library, "hipMallocArray");
  hipMallocAsync = dlsym(__amdhip64_library, "hipMallocAsync");
  hipMallocFromPoolAsync = dlsym(__amdhip64_library, "hipMallocFromPoolAsync");
  hipMallocHost = dlsym(__amdhip64_library, "hipMallocHost");
  hipMallocManaged = dlsym(__amdhip64_library, "hipMallocManaged");
  hipMallocMipmappedArray =
      dlsym(__amdhip64_library, "hipMallocMipmappedArray");
  hipMallocPitch = dlsym(__amdhip64_library, "hipMallocPitch");
  hipMemAddressFree = dlsym(__amdhip64_library, "hipMemAddressFree");

  /* librocblas */
  __rocblas_library = dlopen("rocblas.so", RTLD_NOW | RTLD_GLOBAL);

  rocblas_caxpy = dlsym(__rocblas_library, "rocblas_caxpy");
  rocblas_ccopy = dlsym(__rocblas_library, "rocblas_ccopy");
  rocblas_cdotc = dlsym(__rocblas_library, "rocblas_cdotc");
  rocblas_cdotu = dlsym(__rocblas_library, "rocblas_cdotu");
  rocblas_cgbmv = dlsym(__rocblas_library, "rocblas_cgbmv");
  rocblas_cgemm = dlsym(__rocblas_library, "rocblas_cgemm");
  rocblas_cgemv = dlsym(__rocblas_library, "rocblas_cgemv");
  rocblas_cgerc = dlsym(__rocblas_library, "rocblas_cgerc");
  rocblas_cgeru = dlsym(__rocblas_library, "rocblas_cgeru");
  rocblas_chbmv = dlsym(__rocblas_library, "rocblas_chbmv");
  rocblas_chemm = dlsym(__rocblas_library, "rocblas_chemm");
  rocblas_chemv = dlsym(__rocblas_library, "rocblas_chemv");
  rocblas_cher = dlsym(__rocblas_library, "rocblas_cher");
  rocblas_cher2 = dlsym(__rocblas_library, "rocblas_cher2");
  rocblas_cher2k = dlsym(__rocblas_library, "rocblas_cher2k");
  rocblas_cherk = dlsym(__rocblas_library, "rocblas_cherk");
  rocblas_chpmv = dlsym(__rocblas_library, "rocblas_chpmv");
  rocblas_chpr = dlsym(__rocblas_library, "rocblas_chpr");
  rocblas_chpr2 = dlsym(__rocblas_library, "rocblas_chpr2");
  rocblas_crotg = dlsym(__rocblas_library, "rocblas_crotg");
  rocblas_cscal = dlsym(__rocblas_library, "rocblas_cscal");
  rocblas_csrot = dlsym(__rocblas_library, "rocblas_csrot");
  rocblas_csscal = dlsym(__rocblas_library, "rocblas_csscal");
  rocblas_cswap = dlsym(__rocblas_library, "rocblas_cswap");
  rocblas_csymm = dlsym(__rocblas_library, "rocblas_csymm");
  rocblas_csyr2k = dlsym(__rocblas_library, "rocblas_csyr2k");
  rocblas_csyrk = dlsym(__rocblas_library, "rocblas_csyrk");
  rocblas_ctbmv = dlsym(__rocblas_library, "rocblas_ctbmv");
  rocblas_ctbsv = dlsym(__rocblas_library, "rocblas_ctbsv");
  rocblas_ctpmv = dlsym(__rocblas_library, "rocblas_ctpmv");
  rocblas_ctpsv = dlsym(__rocblas_library, "rocblas_ctpsv");
  rocblas_ctrmm = dlsym(__rocblas_library, "rocblas_ctrmm");
  rocblas_ctrmv = dlsym(__rocblas_library, "rocblas_ctrmv");
  rocblas_ctrsm = dlsym(__rocblas_library, "rocblas_ctrsm");
  rocblas_ctrsv = dlsym(__rocblas_library, "rocblas_ctrsv");
  rocblas_dasum = dlsym(__rocblas_library, "rocblas_dasum");
  rocblas_daxpy = dlsym(__rocblas_library, "rocblas_daxpy");
  rocblas_dcabs1 = dlsym(__rocblas_library, "rocblas_dcabs1");
  rocblas_dcopy = dlsym(__rocblas_library, "rocblas_dcopy");
  rocblas_ddot = dlsym(__rocblas_library, "rocblas_ddot");
  rocblas_dgbmv = dlsym(__rocblas_library, "rocblas_dgbmv");
  rocblas_dgemm = dlsym(__rocblas_library, "rocblas_dgemm");
  rocblas_dgemv = dlsym(__rocblas_library, "rocblas_dgemv");
  rocblas_dger = dlsym(__rocblas_library, "rocblas_dger");
  rocblas_dnrm2 = dlsym(__rocblas_library, "rocblas_dnrm2");
  rocblas_drot = dlsym(__rocblas_library, "rocblas_drot");
  rocblas_drotg = dlsym(__rocblas_library, "rocblas_drotg");
  rocblas_drotm = dlsym(__rocblas_library, "rocblas_drotm");
  rocblas_drotmg = dlsym(__rocblas_library, "rocblas_drotmg");
  rocblas_dsbmv = dlsym(__rocblas_library, "rocblas_dsbmv");
  rocblas_dscal = dlsym(__rocblas_library, "rocblas_dscal");
  rocblas_dsdot = dlsym(__rocblas_library, "rocblas_dsdot");
  rocblas_dspmv = dlsym(__rocblas_library, "rocblas_dspmv");
  rocblas_dspr = dlsym(__rocblas_library, "rocblas_dspr");
  rocblas_dspr2 = dlsym(__rocblas_library, "rocblas_dspr2");
  rocblas_dswap = dlsym(__rocblas_library, "rocblas_dswap");
  rocblas_dsymm = dlsym(__rocblas_library, "rocblas_dsymm");
  rocblas_dsymv = dlsym(__rocblas_library, "rocblas_dsymv");
  rocblas_dsyr = dlsym(__rocblas_library, "rocblas_dsyr");
  rocblas_dsyr2 = dlsym(__rocblas_library, "rocblas_dsyr2");
  rocblas_dsyr2k = dlsym(__rocblas_library, "rocblas_dsyr2k");
  rocblas_dsyrk = dlsym(__rocblas_library, "rocblas_dsyrk");
  rocblas_dtbmv = dlsym(__rocblas_library, "rocblas_dtbmv");
  rocblas_dtbsv = dlsym(__rocblas_library, "rocblas_dtbsv");
  rocblas_dtpmv = dlsym(__rocblas_library, "rocblas_dtpmv");
  rocblas_dtpsv = dlsym(__rocblas_library, "rocblas_dtpsv");
  rocblas_dtrmm = dlsym(__rocblas_library, "rocblas_dtrmm");
  rocblas_dtrmv = dlsym(__rocblas_library, "rocblas_dtrmv");
  rocblas_dtrsm = dlsym(__rocblas_library, "rocblas_dtrsm");
  rocblas_dtrsv = dlsym(__rocblas_library, "rocblas_dtrsv");
  rocblas_dzasum = dlsym(__rocblas_library, "rocblas_dzasum");
  rocblas_dznrm2 = dlsym(__rocblas_library, "rocblas_dznrm2");
  rocblas_icamax = dlsym(__rocblas_library, "rocblas_icamax");
  rocblas_idamax = dlsym(__rocblas_library, "rocblas_idamax");
  rocblas_isamax = dlsym(__rocblas_library, "rocblas_isamax");
  rocblas_izamax = dlsym(__rocblas_library, "rocblas_izamax");
  rocblas_lsame = dlsym(__rocblas_library, "rocblas_lsame");
  rocblas_sasum = dlsym(__rocblas_library, "rocblas_sasum");
  rocblas_saxpy = dlsym(__rocblas_library, "rocblas_saxpy");
  rocblas_scabs1 = dlsym(__rocblas_library, "rocblas_scabs1");
  rocblas_scasum = dlsym(__rocblas_library, "rocblas_scasum");
  rocblas_scnrm2 = dlsym(__rocblas_library, "rocblas_scnrm2");
  rocblas_scopy = dlsym(__rocblas_library, "rocblas_scopy");
  rocblas_sdot = dlsym(__rocblas_library, "rocblas_sdot");
  rocblas_sdsdot = dlsym(__rocblas_library, "rocblas_sdsdot");
  rocblas_sgbmv = dlsym(__rocblas_library, "rocblas_sgbmv");
  rocblas_sgemm = dlsym(__rocblas_library, "rocblas_sgemm");
  rocblas_sgemv = dlsym(__rocblas_library, "rocblas_sgemv");
  rocblas_sger = dlsym(__rocblas_library, "rocblas_sger");
  rocblas_snrm2 = dlsym(__rocblas_library, "rocblas_snrm2");
  rocblas_srot = dlsym(__rocblas_library, "rocblas_srot");
  rocblas_srotg = dlsym(__rocblas_library, "rocblas_srotg");
  rocblas_srotm = dlsym(__rocblas_library, "rocblas_srotm");
  rocblas_srotmg = dlsym(__rocblas_library, "rocblas_srotmg");
  rocblas_ssbmv = dlsym(__rocblas_library, "rocblas_ssbmv");
  rocblas_sscal = dlsym(__rocblas_library, "rocblas_sscal");
  rocblas_sspmv = dlsym(__rocblas_library, "rocblas_sspmv");
  rocblas_sspr = dlsym(__rocblas_library, "rocblas_sspr");
  rocblas_sspr2 = dlsym(__rocblas_library, "rocblas_sspr2");
  rocblas_sswap = dlsym(__rocblas_library, "rocblas_sswap");
  rocblas_ssymm = dlsym(__rocblas_library, "rocblas_ssymm");
  rocblas_ssymv = dlsym(__rocblas_library, "rocblas_ssymv");
  rocblas_ssyr = dlsym(__rocblas_library, "rocblas_ssyr");
  rocblas_ssyr2 = dlsym(__rocblas_library, "rocblas_ssyr2");
  rocblas_ssyr2k = dlsym(__rocblas_library, "rocblas_ssyr2k");
  rocblas_ssyrk = dlsym(__rocblas_library, "rocblas_ssyrk");
  rocblas_stbmv = dlsym(__rocblas_library, "rocblas_stbmv");
  rocblas_stbsv = dlsym(__rocblas_library, "rocblas_stbsv");
  rocblas_stpmv = dlsym(__rocblas_library, "rocblas_stpmv");
  rocblas_stpsv = dlsym(__rocblas_library, "rocblas_stpsv");
  rocblas_strmm = dlsym(__rocblas_library, "rocblas_strmm");
  rocblas_strmv = dlsym(__rocblas_library, "rocblas_strmv");
  rocblas_strsm = dlsym(__rocblas_library, "rocblas_strsm");
  rocblas_strsv = dlsym(__rocblas_library, "rocblas_strsv");
  rocblas_xerbla = dlsym(__rocblas_library, "rocblas_xerbla");
  rocblas_xerbla_array = dlsym(__rocblas_library, "rocblas_xerbla_array");
  rocblas_zaxpy = dlsym(__rocblas_library, "rocblas_zaxpy");
  rocblas_zcopy = dlsym(__rocblas_library, "rocblas_zcopy");
  rocblas_zdotc = dlsym(__rocblas_library, "rocblas_zdotc");
  rocblas_zdotu = dlsym(__rocblas_library, "rocblas_zdotu");
  rocblas_zdrot = dlsym(__rocblas_library, "rocblas_zdrot");
  rocblas_zdscal = dlsym(__rocblas_library, "rocblas_zdscal");
  rocblas_zgbmv = dlsym(__rocblas_library, "rocblas_zgbmv");
  rocblas_zgemm = dlsym(__rocblas_library, "rocblas_zgemm");
  rocblas_zgemv = dlsym(__rocblas_library, "rocblas_zgemv");
  rocblas_zgerc = dlsym(__rocblas_library, "rocblas_zgerc");
  rocblas_zgeru = dlsym(__rocblas_library, "rocblas_zgeru");
  rocblas_zhbmv = dlsym(__rocblas_library, "rocblas_zhbmv");
  rocblas_zhemm = dlsym(__rocblas_library, "rocblas_zhemm");
  rocblas_zhemv = dlsym(__rocblas_library, "rocblas_zhemv");
  rocblas_zher = dlsym(__rocblas_library, "rocblas_zher");
  rocblas_zher2 = dlsym(__rocblas_library, "rocblas_zher2");
  rocblas_zher2k = dlsym(__rocblas_library, "rocblas_zher2k");
  rocblas_zherk = dlsym(__rocblas_library, "rocblas_zherk");
  rocblas_zhpmv = dlsym(__rocblas_library, "rocblas_zhpmv");
  rocblas_zhpr = dlsym(__rocblas_library, "rocblas_zhpr");
  rocblas_zhpr2 = dlsym(__rocblas_library, "rocblas_zhpr2");
  rocblas_zrotg = dlsym(__rocblas_library, "rocblas_zrotg");
  rocblas_zscal = dlsym(__rocblas_library, "rocblas_zscal");
  rocblas_zswap = dlsym(__rocblas_library, "rocblas_zswap");
  rocblas_zsymm = dlsym(__rocblas_library, "rocblas_zsymm");
  rocblas_zsyr2k = dlsym(__rocblas_library, "rocblas_zsyr2k");
  rocblas_zsyrk = dlsym(__rocblas_library, "rocblas_zsyrk");
  rocblas_ztbmv = dlsym(__rocblas_library, "rocblas_ztbmv");
  rocblas_ztbsv = dlsym(__rocblas_library, "rocblas_ztbsv");
  rocblas_ztpmv = dlsym(__rocblas_library, "rocblas_ztpmv");
  rocblas_ztpsv = dlsym(__rocblas_library, "rocblas_ztpsv");
  rocblas_ztrmm = dlsym(__rocblas_library, "rocblas_ztrmm");
  rocblas_ztrmv = dlsym(__rocblas_library, "rocblas_ztrmv");
  rocblas_ztrsm = dlsym(__rocblas_library, "rocblas_ztrsm");
  rocblas_ztrsv = dlsym(__rocblas_library, "rocblas_ztrsv");
}

__attribute__((destructor)) void rblas_exit() {
  dlclose(__amdhip64_library);
  dlclose(__rocblas_library);
}
