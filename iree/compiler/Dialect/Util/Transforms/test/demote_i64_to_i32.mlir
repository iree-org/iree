// RUN: iree-opt -split-input-file -iree-util-demote-i64-to-i32 %s | FileCheck %s

// CHECK-LABEL: func @constant_i64
// CHECK-SAME: () -> i32
func.func @constant_i64() -> i64 {
  // CHECK-NEXT: constant 123 : i32
  %c123 = arith.constant 123 : i64
  return %c123 : i64
}

// -----

// CHECK-LABEL: func @constant_splat_i64
// CHECK-SAME: () -> tensor<4xi32>
func.func @constant_splat_i64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<123> : tensor<4xi32>
  %c123 = arith.constant dense<123> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @constant_dense_i64
// CHECK-SAME: () -> tensor<4xi32>
func.func @constant_dense_i64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %c123 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @args_i64
// CHECK-SAME: (%arg0: i32) -> i32
func.func @args_i64(%arg0: i64) -> i64 {
  // CHECK-NEXT: return %arg0 : i32
  return %arg0 : i64
}

// -----

// CHECK-LABEL: func @args_ui64
// CHECK-SAME: (%arg0: ui32) -> ui32
func.func @args_ui64(%arg0: ui64) -> ui64 {
  // CHECK-NEXT: return %arg0 : ui32
  return %arg0 : ui64
}

// -----

// CHECK-LABEL: func @args_tensor_i64
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @args_tensor_i64(%arg0: tensor<4x4xi64>) -> tensor<4x4xi64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xi32>
  return %arg0 : tensor<4x4xi64>
}

// -----

// CHECK-LABEL: func @mhlo_constant_i64
// CHECK-SAME: () -> tensor<1xi32>
func.func @mhlo_constant_i64() -> tensor<1xi64> {
  // CHECK-NEXT: mhlo.constant dense<123> : tensor<1xi32>
  %c123 = mhlo.constant dense<123> : tensor<1xi64>
  return %c123 : tensor<1xi64>
}

// -----

// CHECK-LABEL: func @mhlo_constant_ui64
// CHECK-SAME: () -> tensor<1xui32>
func.func @mhlo_constant_ui64() -> tensor<1xui64> {
  // CHECK-NEXT: mhlo.constant dense<123> : tensor<1xui32>
  %c123 = mhlo.constant dense<123> : tensor<1xui64>
  return %c123 : tensor<1xui64>
}

// -----

// CHECK-LABEL: func @mhlo_compare_i64
// CHECK-SAME: (%arg0: tensor<i32>, %arg1: tensor<i32>) -> (i1, tensor<i32>)
func.func @mhlo_compare_i64(%arg0 : tensor<i64>, %arg1 : tensor<i64>) -> (i1, tensor<i64>) {
  // CHECK-NEXT: %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %1 = tensor.extract %0[] : tensor<i1>
  // CHECK-NEXT: cf.cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i32>), ^bb2(%1, %arg1 : i1, tensor<i32>)
  // CHECK-NEXT: ^bb1(%2: i1, %3: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %2, %3 : i1, tensor<i32>
  // CHECK-NEXT: ^bb2(%4: i1, %5: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %4, %5 : i1, tensor<i32>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %1 = tensor.extract %0[] : tensor<i1>
  cf.cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i64>), ^bb2(%1, %arg1 : i1, tensor<i64>)
^bb1(%2 : i1, %3 : tensor<i64>):
  return %2, %3 : i1, tensor<i64>
^bb2(%4 : i1, %5 : tensor<i64>):
  return %4, %5 : i1, tensor<i64>
}

// -----

// CHECK-LABEL: func @linalg_matmul_i64
func.func @linalg_matmul_i64(%arg0: tensor<2x3xi64>, %arg1: tensor<3x4xi64>, %arg2: tensor<2x4xi64>)  -> tensor<2x4xi64> {
  // CHECK: %[[T:.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xi32>, tensor<3x4xi32>)
  // CHECK-SAME:                     outs(%arg2 : tensor<2x4xi32>) -> tensor<2x4xi32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xi64>, tensor<3x4xi64>)
                    outs(%arg2 : tensor<2x4xi64>) -> tensor<2x4xi64>
  // CHECK-NEXT: return %[[T:.+]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi64>
}

// -----

// CHECK-LABEL: func @linalg_generic_i64
// CHECK-SAME: (%[[ARG:.+]]: tensor<2xi32>) -> tensor<2xi32>
func.func @linalg_generic_i64(%arg: tensor<2xi64>)  -> tensor<2xi64> {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [2] : tensor<2xi32>
  %init = linalg.init_tensor [2] : tensor<2xi64>
  // CHECK: %[[T:.+]] = linalg.generic {{.+}} ins(%[[ARG]] : tensor<2xi32>) outs(%[[INIT]] : tensor<2xi32>)
  %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg : tensor<2xi64>) outs(%init : tensor<2xi64>) {
  // CHECK-NEXT: ^bb0(%[[A:.+]]: i32, %[[B:.+]]: i32):
  ^bb0(%arg1: i64, %arg2: i64):
    // CHECK-NEXT: linalg.yield %[[A]] : i32
    linalg.yield %arg1 : i64
  } -> tensor<2xi64>
  // CHECK: %[[T]] : tensor<2xi32>
  return %generic : tensor<2xi64>
}

// -----

// CHECK-LABEL: func @linalg_non_structured_op
// CHECK-SAME:    (%arg0: tensor<9xi32>) -> tensor<1x9xi32>
func.func @linalg_non_structured_op(%arg0: tensor<9xi64>) -> tensor<1x9xi64> {
  // CHECK:       %[[RES:.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1]] : tensor<9xi32> into tensor<1x9xi32>
  // CHECK:       return %[[RES:.+]] : tensor<1x9xi32>
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<9xi64> into tensor<1x9xi64>
  return %0 : tensor<1x9xi64>
}

// -----

// CHECK: util.global public mutable @[[VAR:.+]] = dense<0> : tensor<i32>
// CHECK: util.global.load @[[VAR]]
// CHECK: util.global.store %{{.+}}, @[[VAR]]
util.global mutable @readwritevar = dense<0> : tensor<i64>
func.func @foo(%arg0 : tensor<i64>) {
  %0 = util.global.load @readwritevar : tensor<i64>
  %1 = chlo.broadcast_add %0, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i64>
  util.global.store %1, @readwritevar : tensor<i64>
  return
}

// -----

// CHECK: util.global private @{{.+}} : tensor<4xi32>
util.global private @v_initializer : tensor<4xi64>
util.initializer {
  // CHECK: %[[VALUE:.+]] = call @initializer() : () -> tensor<4xi32>
  %0 = call @initializer() : () -> tensor<4xi64>
  // CHECK: util.global.store %[[VALUE]], @v_initializer : tensor<4xi32>
  util.global.store %0, @v_initializer : tensor<4xi64>
  util.initializer.return
}
// CHECK: func private @initializer() -> tensor<4xi32>
func.func private @initializer() -> tensor<4xi64>

// -----

//       CHECK: util.global {{.*}} : tensor<4xi32>
// CHECK-LABEL: func @simple_i64() -> tensor<4xi32>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xi32>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xi32>> -> tensor<4xi32>
//  CHECK-NEXT: return %{{.*}} : tensor<4xi32>
util.global private @"__global" = dense<[1, 2, 3, 4]> : tensor<4xi64>
func.func @simple_i64() -> (tensor<4xi64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xi64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xi64>> -> tensor<4xi64>
  return %1 : tensor<4xi64>
}

// -----

// CHECK: util.global {{.+}} : tensor<4xi32>
util.global private @"__global" = dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK-LABEL: func @nested_region_i64()
func.func @nested_region_i64() -> (tensor<4xi64>) {
  // CHECK-NEXT: util.global.address {{.+}} : !util.ptr<tensor<4xi32>>
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xi64>>
  // CHECK-NEXT: util.global.load.indirect {{.+}} : !util.ptr<tensor<4xi32>> -> tensor<4xi32>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xi64>> -> tensor<4xi64>
  // CHECK-NEXT: "mhlo.broadcast_in_dim"({{.+}}) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<4x4xi32>
  %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<123> : tensor<i32>
  %4 = mhlo.constant dense<123> : tensor<i64>
  // CHECK-NEXT: mhlo.reduce
  %3 = mhlo.reduce(%2 init: %4) across dimensions = [1] : (tensor<4x4xi64>, tensor<i64>) -> tensor<4xi64>
  // CHECK-NEXT: reducer(%[[A:.+]]: tensor<i32>, %[[B:.+]]: tensor<i32>)
  reducer(%a: tensor<i64>, %b: tensor<i64>) {
    // CHECK-NEXT: mhlo.maximum %[[A]], %[[B]] : tensor<i32>
    %c = mhlo.maximum %a, %b : tensor<i64>
    // CHECK-NEXT: "mhlo.return"{{.+}} (tensor<i32>)
    "mhlo.return"(%c) : (tensor<i64>) -> ()
  }
  // CHECK: return %{{.*}} : tensor<4xi32>
  return %3 : tensor<4xi64>
}

// -----

// Check handling of width-sensitive arith casts.

// CHECK-LABEL:   func @arith.trunci(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func @arith.trunci(%arg0: i64) -> i32 {
  %0 = arith.trunci %arg0 : i64 to i32
  return %0 : i32
}

// CHECK-LABEL:   func @arith.extui(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func @arith.extui(%arg0: i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL:   func @arith.extsi(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func @arith.extsi(%arg0: i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}
