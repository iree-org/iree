// RUN: iree-run-mlir \
// RUN:   --Xcompiler,iree-hal-target-backends=vmvx \
// RUN:   --Xcompiler,iree-flow-trace-dispatch-tensors \
// RUN:   %s 2>&1 | FileCheck %s

func.func private @double(%input : tensor<15xf32>) -> tensor<15xf32> {
  %init = tensor.empty() : tensor<15xf32>
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<15xf32>)
    outs(%init : tensor<15xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %in : f32
    linalg.yield %0 : f32
  } -> tensor<15xf32>
  return %res : tensor<15xf32>
}

func.func @two_dispatch() -> (tensor<15xf32>) {
  %0 = util.unfoldable_constant dense<1.0> : tensor<15xf32>
  %1 = call @double(%0) : (tensor<15xf32>) -> tensor<15xf32>

  // Explicitly disable fusion because we want two dispatches.
  %barrier = util.optimization_barrier %1 : tensor<15xf32>
  %2 = call @double(%barrier) : (tensor<15xf32>) -> tensor<15xf32>
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
