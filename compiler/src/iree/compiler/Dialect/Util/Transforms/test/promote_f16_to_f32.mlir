// RUN: iree-opt --split-input-file -iree-util-promote-f16-to-f32 %s | FileCheck %s

// NOTE: for more comprehensive tests see demote_i64_to_i32.mlir.

//       CHECK: util.global {{.*}} : tensor<4xf32>
// CHECK-LABEL: func.func @simple_f16() -> tensor<4xf32>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xf32>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
//  CHECK-NEXT: return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf16>
func.func @simple_f16() -> (tensor<4xf16>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf16>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf16>> -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK: util.global
// CHECK-NOT: f16
// CHECK-LABEL: func.func @nested_region_f16()
// CHECK-NOT: f16
// CHECK: return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf16>
func.func @nested_region_f16() -> (tensor<4xf16>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf16>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf16>> -> tensor<4xf16>
  %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf16>) -> tensor<4x4xf16>
  %4 = mhlo.constant dense<0xFC00> : tensor<f16>
  %3 = "mhlo.reduce"(%2, %4) ( {
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    %5393 = mhlo.maximum %arg3, %arg4 : tensor<f16>
    "mhlo.return"(%5393) : (tensor<f16>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x4xf16>, tensor<f16>) -> tensor<4xf16>
  return %3 : tensor<4xf16>
}
