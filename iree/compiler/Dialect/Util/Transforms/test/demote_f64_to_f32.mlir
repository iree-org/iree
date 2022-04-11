// RUN: iree-opt -split-input-file -iree-util-demote-f64-to-f32 %s | FileCheck %s

// NOTE: for more comprehensive tests see demote_i64_to_i32.mlir.

// CHECK-LABEL: func @constantF64
// CHECK-SAME: () -> f32
func.func @constantF64() -> f64 {
  // CHECK-NEXT: constant 123.{{.+}} : f32
  %c1234 = arith.constant 123.4 : f64
  return %c1234 : f64
}

// -----

// CHECK-LABEL: func @tensorTypesF64
// CHECK-SAME: (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @tensorTypesF64(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  return %arg0 : tensor<4x4xf64>
}

// -----

//       CHECK: util.global {{.*}} : tensor<4xf32>
// CHECK-LABEL: func @simple_f64() -> tensor<4xf32>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xf32>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
//  CHECK-NEXT: return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf64>
func.func @simple_f64() -> (tensor<4xf64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf64>> -> tensor<4xf64>
  return %1 : tensor<4xf64>
}

// -----

// CHECK: util.global
// CHECK-NOT: f64
// CHECK-LABEL: func @nested_region_f64()
// CHECK-NOT: f64
// CHECK: return %{{.*}} : tensor<4xf32>
util.global private @"__global" = dense<[1.000000e+01, 5.000000e+00, 1.000000e+01, 5.000000e+00]> : tensor<4xf64>
func.func @nested_region_f64() -> (tensor<4xf64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xf64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf64>> -> tensor<4xf64>
  %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf64>) -> tensor<4x4xf64>
  %4 = mhlo.constant dense<0xFF800000> : tensor<f64>
  %3 = "mhlo.reduce"(%2, %4) ( {
  ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
    %5393 = mhlo.maximum %arg3, %arg4 : tensor<f64>
    "mhlo.return"(%5393) : (tensor<f64>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x4xf64>, tensor<f64>) -> tensor<4xf64>
  return %3 : tensor<4xf64>
}

// -----

// Check handling of width-sensitive arith casts.

// CHECK-LABEL:   func @arith.truncf(
// CHECK-SAME:            %[[ARG0:.*]]: f32) -> f32 {
// CHECK:           return %[[ARG0]] : f32
func @arith.truncf(%arg0: f64) -> f32 {
  %0 = arith.truncf %arg0 : f64 to f32
  return %0 : f32
}

// CHECK-LABEL:   func @arith.extf(
// CHECK-SAME:            %[[ARG0:.*]]: f32) -> f32 {
// CHECK:           return %[[ARG0]] : f32
func @arith.extf(%arg0: f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}
