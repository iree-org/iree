// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @dynamic_tensor
func @dynamic_tensor() -> tensor<?x?xf32> attributes { iree.module.export } {
  %input = iree.dynamic_shape_constant dense<[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]> : tensor<2x3xf32> -> tensor<?x?xf32>
  %res = "mhlo.abs"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}

// CHECK: 2x3xf32=[1 2 3][4 5 6]
