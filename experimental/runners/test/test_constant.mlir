// RUN: mlir-proto-opt %s -linalg-comprehensive-bufferize-inplace | FileCheck %s

// RUN: mlir-proto-opt %s -linalg-comprehensive-bufferize-inplace |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee | FileCheck %s --check-prefix=EXEC

// CHECK: memref.global "private" constant @__constant_1x1xf32_1 : memref<1x1xf32> = dense<1.000000e+00>
// CHECK: memref.global "private" constant @__constant_1x1xf32_0 : memref<1x1xf32> = dense<3.000000e+00>
// CHECK: memref.global "private" constant @__constant_1x1xf32 : memref<1x1xf32> = dense<2.000000e+00>

// CHECK-LABEL: func @main() {
func @main() {
  %c0 = constant 0: index
  %v0 = constant 0.0 : f32

  // Top of the function globals, memref.alloc, copy.
  // CHECK-DAG:   %[[A:.*]] = memref.get_global @__constant_1x1xf32 : memref<1x1xf32>
  // CHECK-DAG:   %[[B:.*]] = memref.get_global @__constant_1x1xf32_0 : memref<1x1xf32>
  // CHECK-DAG:   %[[C:.*]] = memref.get_global @__constant_1x1xf32_1 : memref<1x1xf32>
  //     CHECK:   %[[MUTABLE_C:.*]] = memref.alloc() : memref<1x1xf32>
  //     CHECK:   linalg.copy(%[[C]], %[[MUTABLE_C]]) : memref<1x1xf32>, memref<1x1xf32>
  %lhs = constant dense<[[2.]]> : tensor<1x1xf32>
  %rhs = constant dense<[[3.]]> : tensor<1x1xf32>
  %accum = constant dense<[[1.]]> : tensor<1x1xf32>

  //     CHECK:   vector.transfer_read %[[A]]{{.*}} {masked = [false, false]} : memref<1x1xf32>, vector<1x1xf32>
  %result_vector_0 = vector.transfer_read %lhs[%c0, %c0], %v0 : tensor<1x1xf32>, vector<1x1xf32>

  // EXEC: ( ( 2 ) )
  vector.print %result_vector_0: vector<1x1xf32>


  //     CHECK:   linalg.matmul ins(%[[A]], %[[B]] : memref<1x1xf32>, memref<1x1xf32>) outs(%[[MUTABLE_C]] : memref<1x1xf32>)
  %result = linalg.matmul ins(%lhs, %rhs : tensor<1x1xf32>, tensor<1x1xf32>)
    outs(%accum: tensor<1x1xf32>) -> tensor<1x1xf32>

  //     CHECK:   vector.transfer_read %[[MUTABLE_C]]{{.*}} {masked = [false, false]} : memref<1x1xf32>, vector<1x1xf32>
  %result_vector_1 = vector.transfer_read %result[%c0, %c0], %v0 : tensor<1x1xf32>, vector<1x1xf32>

  // EXEC: ( ( 7 ) )
  vector.print %result_vector_1: vector<1x1xf32>

  //     CHECK:   memref.dealloc %[[MUTABLE_C]] : memref<1x1xf32>
  return
}
