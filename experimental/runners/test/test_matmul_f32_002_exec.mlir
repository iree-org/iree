// RUN: export M=128 && export N=128 && export K=128 && export ITERS=10 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=4,8,16 pad hoist-padding=1" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul vectorize vectorize-padding" |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: tee | FileCheck %s

// CHECK: ( ( 256 ) )
