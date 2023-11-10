// RUN: iree-run-mlir \
// RUN:   --Xcompiler,iree-hal-target-backends=vmvx \
// RUN:   --Xcompiler,iree-flow-trace-dispatch-tensors \
// RUN:   %s 2>&1 | FileCheck %s

func.func private @double(%input : tensor<5x3xf32>) -> tensor<5x3xf32> {
  %init = tensor.empty() : tensor<5x3xf32>
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<5x3xf32>)
    outs(%init : tensor<5x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %in : f32
    linalg.yield %0 : f32
  } -> tensor<5x3xf32>
  return %res : tensor<5x3xf32>
}

func.func @two_dispatch() -> (tensor<5x3xf32>) {
  %0 = util.unfoldable_constant dense<1.0> : tensor<5x3xf32>
  %1 = call @double(%0) : (tensor<5x3xf32>) -> tensor<5x3xf32>

  // Explicitly disable fusion because we want two dispatches.
  %barrier = util.optimization_barrier %1 : tensor<5x3xf32>
  %2 = call @double(%barrier) : (tensor<5x3xf32>) -> tensor<5x3xf32>
  return %2 : tensor<5x3xf32>
}
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} inputs ===
// CHECK: 5x3xf32=
// CHECK: === two_dispatch_dispatch_0::two_dispatch_dispatch_0{{.*}} outputs ===
// CHECK: 5x3xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} inputs ===
// CHECK: 5x3xf32=
// CHECK: === two_dispatch_dispatch_1::two_dispatch_dispatch_1{{.*}} outputs ===
// CHECK: 5x3xf32=
