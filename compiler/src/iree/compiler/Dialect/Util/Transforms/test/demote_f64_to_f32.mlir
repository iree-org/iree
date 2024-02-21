// RUN: iree-opt --split-input-file --iree-util-demote-f64-to-f32 %s | FileCheck %s

// NOTE: for more comprehensive tests see demote_i64_to_i32.mlir.

// CHECK-LABEL: util.func public @constantF64
// CHECK-SAME: () -> f32
util.func public @constantF64() -> f64 {
  // CHECK-NEXT: constant 123.{{.+}} : f32
  %c1234 = arith.constant 123.4 : f64
  util.return %c1234 : f64
}

// -----

// CHECK-LABEL: util.func public @tensorTypesF64
// CHECK-SAME: (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32>
util.func public @tensorTypesF64(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  util.return %arg0 : tensor<4x4xf64>
}

// -----

//       CHECK: util.global {{.*}} : tensor<4xf32>
// CHECK-LABEL: util.func public @simple_f64() -> tensor<4xf32>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xf32>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
//  CHECK-NEXT: util.return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf64>
util.func public @simple_f64() -> (tensor<4xf64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf64>> -> tensor<4xf64>
  util.return %1 : tensor<4xf64>
}

// -----

// CHECK: util.global
// CHECK-NOT: f64
// CHECK-LABEL: util.func public @nested_region_f64()
// CHECK-NOT: f64
// CHECK: util.return %{{.*}} : tensor<?xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf64>
util.func public @nested_region_f64() -> (tensor<?xf64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf64>> -> tensor<4xf64>
  %c4 = arith.constant 4 : index
  %2 = tensor.generate %c4 {
  ^bb0(%arg0: index) :
    %element = tensor.extract %1[%arg0] : tensor<4xf64>
    tensor.yield %element : f64
  } : tensor<?xf64>
  util.return %2 : tensor<?xf64>
}

// -----

// Check handling of width-sensitive arith casts.

// CHECK-LABEL: util.func public @arith.truncf(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> f32 {
// CHECK:         util.return %[[ARG0]] : f32
util.func public @arith.truncf(%arg0: f64) -> f32 {
  %0 = arith.truncf %arg0 : f64 to f32
  util.return %0 : f32
}

// CHECK-LABEL: util.func public @arith.extf(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> f32 {
// CHECK:         util.return %[[ARG0]] : f32
util.func public @arith.extf(%arg0: f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  util.return %0 : f64
}

// -----

// CHECK-LABEL: util.func public @complexTypesF64
// CHECK-SAME: (%arg0: complex<f32>) -> complex<f32>
util.func public @complexTypesF64(%arg0 : complex<f64>) -> complex<f64> {
  // CHECK-NEXT: util.return %arg0 : complex<f32>
  util.return %arg0 : complex<f64>
}

