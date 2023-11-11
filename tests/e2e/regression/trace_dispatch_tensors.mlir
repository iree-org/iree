// RUN: iree-run-mlir \
// RUN:   --Xcompiler,iree-hal-target-backends=vmvx \
// RUN:   --Xcompiler,iree-flow-trace-dispatch-tensors \
// RUN:   --Xcompiler,iree-opt-data-tiling=false \
// RUN:   %s 2>&1 | FileCheck %s

func.func @two_dispatch() -> (tensor<15xf32>) {
  %0 = util.unfoldable_constant dense<1.0> : tensor<15xf32>
  %1 = arith.addf %0, %0 : tensor<15xf32>

  // Explicitly disable fusion because we want two dispatches.
  %barrier = util.optimization_barrier %1 : tensor<15xf32>
  %cst = arith.constant dense<1.0> : tensor<15xf32>
  %2 = arith.addf %barrier, %cst : tensor<15xf32>
  return %2 : tensor<15xf32>
}
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} inputs ===
// CHECK: 15xf32=
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} outputs ===
// CHECK: 15xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} inputs ===
// CHECK: 15xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} outputs ===
// CHECK: 15xf32=
