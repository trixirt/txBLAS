#!/bin/sh

S="init func lib mem roc gemv her2k herk hemm trsm trmm syr2k syrk symm gemm gbmv symv sbmv"
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

