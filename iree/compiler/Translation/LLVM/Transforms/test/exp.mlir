// RUN: iree-opt -split-input-file -iree-llvm-transformation-pipeline %s | IreeFileCheck %s

// CHECK: func @exp
func @exp(%operand: memref<2x2xf32>) {
  %0 = iree.load_input(%operand : memref<2x2xf32>) : tensor<2x2xf32>
  %1 = "xla_hlo.exp"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  iree.store_output(%1 : tensor<2x2xf32>, %operand : memref<2x2xf32>)
  return
}
// CHECK: %{{.*}} = llvm.load %{{.*}}
// CHECK: llvm.mlir.constant
// CHECK: llvm.mlir.constant
