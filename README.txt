txBLAS

A drop in replacement for cblas that has a first pass of using rocblas.
The real cblas, hip and rocblas routines are dlsym-ed.  The real headers
cblas.h, rocblas.h provide the external interfaces.
