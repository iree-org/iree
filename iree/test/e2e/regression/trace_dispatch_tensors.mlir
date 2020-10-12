// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmla -iree-flow-trace-dispatch-tensors %s 2>&1 | IreeFileCheck %s

func @two_dispatch() -> (tensor<5x5xf32>, tensor<3x5xf32>) attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<1.0> : tensor<5x3xf32>
  %1 = iree.unfoldable_constant dense<0.4> : tensor<3x5xf32>
  %2 = "mhlo.dot"(%0, %1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  %3 = "mhlo.dot"(%1, %2) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
  return %2, %3 : tensor<5x5xf32>, tensor<3x5xf32>
}
// CHECK: === Input for two_dispatch_ex_dispatch_0 ===
// CHECK: 5x3xf32=
// CHECK: 3x5xf32=
// CHECK: === Output for two_dispatch_ex_dispatch_0 ===
// CHECK: 5x5xf32=
// CHECK: === Input for two_dispatch_ex_dispatch_1 ===
// CHECK: 3x5xf32=
// CHECK: 5x5xf32=
// CHECK: === Output for two_dispatch_ex_dispatch_1 ===
// CHECK: 3x5xf32=
