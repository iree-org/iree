// RUN: iree-opt --split-input-file --iree-util-promote-f16-to-f32 %s | FileCheck %s

// NOTE: for more comprehensive tests see demote_i64_to_i32.mlir.

//       CHECK: util.global {{.*}} : tensor<4xf32>
// CHECK-LABEL: util.func public @simple_f16() -> tensor<4xf32>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xf32>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
//  CHECK-NEXT: util.return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf16>
util.func public @simple_f16() -> (tensor<4xf16>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf16>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf16>> -> tensor<4xf16>
  util.return %1 : tensor<4xf16>
}

// -----

// CHECK: util.global
// CHECK-NOT: f16
// CHECK-LABEL: util.func public @nested_region_f16()
// CHECK-NOT: f16
// CHECK: util.return %{{.*}} : tensor<?xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf16>
util.func public @nested_region_f16() -> (tensor<?xf16>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf16>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf16>> -> tensor<4xf16>
  %c4 = arith.constant 4 : index
  %2 = tensor.generate %c4 {
  ^bb0(%arg0: index) :
    %element = tensor.extract %1[%arg0] : tensor<4xf16>
    tensor.yield %element : f16
  } : tensor<?xf16>
  util.return %2 : tensor<?xf16>
}
