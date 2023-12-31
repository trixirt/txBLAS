#!/bin/sh

S="init func lib mem roc gemv her2k herk hemm trsm trmm syr2k syrk symm gemm gbmv symv sbmv spmv trmv tbmv tpmv trsv tbsv tpsv ger syr spr syr2 spr2 her hpr her2 hpr2 rotg rotmg rot rotm swap scal copy axpy dot nrm2 asum amax"
for s in $S; do
    if [ -f ${s}.o ]; then
	rm ${s}.o
    fi
    clang-format -i ${s}.c
    hipcc -c -D__HIP_PLATFORM_AMD__ -fPIC -O0 -g ${s}.c
    if [ ! -f ${s}.o ]; then
	exit
    fi
    O="$O ${s}.o"
done

gcc -shared -o libtxblas.so $O 

