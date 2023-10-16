// RUN: iree-run-mlir \
// RUN:   --Xcompiler,iree-input-type=stablehlo \
// RUN:   --Xcompiler,iree-hal-target-backends=vmvx \
// RUN:   --Xcompiler,iree-flow-trace-dispatch-tensors \
// RUN:   --Xcompiler,iree-opt-data-tiling=false \
// RUN:   %s 2>&1 | FileCheck %s

func.func @two_dispatch() -> (tensor<5x5xf32>, tensor<3x5xf32>) {
  %0 = util.unfoldable_constant dense<1.0> : tensor<5x3xf32>
  %1 = util.unfoldable_constant dense<0.4> : tensor<3x5xf32>
  %2 = "stablehlo.dot"(%0, %1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  %3 = "stablehlo.dot"(%1, %2) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
  return %2, %3 : tensor<5x5xf32>, tensor<3x5xf32>
}
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} inputs ===
// CHECK: 5x3xf32=
// CHECK: 3x5xf32=
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} outputs ===
// CHECK: 5x5xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} inputs ===
// CHECK: 3x5xf32=
// CHECK: 5x5xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} outputs ===
// CHECK: 3x5xf32=
