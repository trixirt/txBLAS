#!/bin/sh
S="init func lib"
O=
for s in $S; do
    if [ -f ${s}.o ]; then
	rm ${s}.o
    fi
    clang-format -i ${s}.c
    gcc -c -fPIC -O0 -g ${s}.c
    if [ ! -f ${s}.o ]; then
	exit
    fi
    O="$O ${s}.o"
done

S="mem roc gemm"
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

gcc -shared -o librblas.so $O

