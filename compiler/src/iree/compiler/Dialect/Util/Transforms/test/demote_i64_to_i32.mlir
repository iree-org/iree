// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-util-demote-i64-to-i32 %s | FileCheck %s

// CHECK-LABEL: func.func @constant_i64
// CHECK-SAME: () -> i32
func.func @constant_i64() -> i64 {
  // CHECK-NEXT: constant 123 : i32
  %c123 = arith.constant 123 : i64
  return %c123 : i64
}

// -----

// CHECK-LABEL: func.func @constant_splat_i64
// CHECK-SAME: () -> tensor<4xi32>
func.func @constant_splat_i64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<123> : tensor<4xi32>
  %c123 = arith.constant dense<123> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func.func @constant_dense_i64
// CHECK-SAME: () -> tensor<4xi32>
func.func @constant_dense_i64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %c123 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func.func @args_i64
// CHECK-SAME: (%arg0: i32) -> i32
func.func @args_i64(%arg0: i64) -> i64 {
  // CHECK-NEXT: return %arg0 : i32
  return %arg0 : i64
}

// -----

// CHECK-LABEL: func.func @args_ui64
// CHECK-SAME: (%arg0: ui32) -> ui32
func.func @args_ui64(%arg0: ui64) -> ui64 {
  // CHECK-NEXT: return %arg0 : ui32
  return %arg0 : ui64
}

// -----

// CHECK-LABEL: func.func @args_tensor_i64
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @args_tensor_i64(%arg0: tensor<4x4xi64>) -> tensor<4x4xi64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xi32>
  return %arg0 : tensor<4x4xi64>
}

// -----

// Return types should be converted for all operations, even those that the
// core compiler is not directly aware of.

// CHECK-LABEL: func.func @custom_constant_i64
// CHECK-SAME: () -> tensor<1xi32>
func.func @custom_constant_i64() -> tensor<1xi64> {
  // CHECK-NEXT: "custom.constant"() : () -> tensor<1xi32>
  %c0 = "custom.constant"() : () -> tensor<1xi64>
  return %c0 : tensor<1xi64>
}

// -----

// CHECK-LABEL: func.func @custom_constant_ui64
// CHECK-SAME: () -> tensor<1xui32>
func.func @custom_constant_ui64() -> tensor<1xui64> {
  // CHECK-NEXT: "custom.constant"() : () -> tensor<1xui32>
  %c0 = "custom.constant"() : () -> tensor<1xui64>
  return %c0 : tensor<1xui64>
}

// -----

// CHECK-LABEL: func.func @arith_cmpi_i64
// CHECK-SAME: (%arg0: tensor<i32>, %arg1: tensor<i32>) -> (i1, tensor<i32>)
func.func @arith_cmpi_i64(%arg0 : tensor<i64>, %arg1 : tensor<i64>) -> (i1, tensor<i64>) {
  // CHECK-NEXT: %0 = arith.cmpi slt, %arg0, %arg1 : tensor<i32>
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %0[] : tensor<i1>
  // CHECK-NEXT: cf.cond_br %[[EXT]], ^bb1(%[[EXT]], %arg0 : i1, tensor<i32>), ^bb2(%[[EXT]], %arg1 : i1, tensor<i32>)
  // CHECK-NEXT: ^bb1(%[[ARG1:.+]]: i1, %[[ARG2:.+]]: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %[[ARG1]], %[[ARG2]] : i1, tensor<i32>
  // CHECK-NEXT: ^bb2(%[[ARG3:.+]]: i1, %[[ARG4:.+]]: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %[[ARG3]], %[[ARG4]] : i1, tensor<i32>
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<i64>
  %1 = tensor.extract %0[] : tensor<i1>
  cf.cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i64>), ^bb2(%1, %arg1 : i1, tensor<i64>)
^bb1(%2 : i1, %3 : tensor<i64>):
  return %2, %3 : i1, tensor<i64>
^bb2(%4 : i1, %5 : tensor<i64>):
  return %4, %5 : i1, tensor<i64>
}

// -----

// CHECK-LABEL: func.func @linalg_matmul_i64
func.func @linalg_matmul_i64(%arg0: tensor<2x3xi64>, %arg1: tensor<3x4xi64>, %arg2: tensor<2x4xi64>)  -> tensor<2x4xi64> {
  // CHECK: %[[T:.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xi32>, tensor<3x4xi32>)
  // CHECK-SAME:                     outs(%arg2 : tensor<2x4xi32>) -> tensor<2x4xi32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xi64>, tensor<3x4xi64>)
                    outs(%arg2 : tensor<2x4xi64>) -> tensor<2x4xi64>
  // CHECK-NEXT: return %[[T:.+]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi64>
}

// -----

// CHECK-LABEL: func.func @linalg_generic_i64
// CHECK-SAME: (%[[ARG:.+]]: tensor<2xi32>) -> tensor<2xi32>
func.func @linalg_generic_i64(%arg: tensor<2xi64>)  -> tensor<2xi64> {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2xi32>
  %init = tensor.empty() : tensor<2xi64>
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

// CHECK-LABEL: func.func @linalg_non_structured_op
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
  %1 = arith.addi %0, %arg0 : tensor<i64>
  util.global.store %1, @readwritevar : tensor<i64>
  return
}

// -----

// CHECK: util.global private @{{.+}} : tensor<4xi32>
util.global private @v_initializer : tensor<4xi64>
util.initializer {
  // CHECK: %[[VALUE:.+]] = func.call @initializer() : () -> tensor<4xi32>
  %0 = func.call @initializer() : () -> tensor<4xi64>
  // CHECK: util.global.store %[[VALUE]], @v_initializer : tensor<4xi32>
  util.global.store %0, @v_initializer : tensor<4xi64>
  util.initializer.return
}
// CHECK: func.func private @initializer() -> tensor<4xi32>
func.func private @initializer() -> tensor<4xi64>

// -----

//       CHECK: util.global {{.*}} : tensor<4xi32>
// CHECK-LABEL: func.func @simple_i64() -> tensor<4xi32>
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

// CHECK: util.global
// CHECK-NOT: i64
// CHECK-LABEL: func.func @nested_region_i64()
// CHECK-NOT: i64
// CHECK: return %{{.*}} : tensor<?xi32>
util.global private @"__global" = dense<[1, 2, 3, 4]> : tensor<4xi64>
func.func @nested_region_i64() -> (tensor<?xi64>) {
  %0 = util.global.address @"__global" : !util.ptr<tensor<4xi64>>
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xi64>> -> tensor<4xi64>
  %c4 = arith.constant 4 : index
  %2 = tensor.generate %c4 {
  ^bb0(%arg0: index) :
    %element = tensor.extract %1[%arg0] : tensor<4xi64>
    tensor.yield %element : i64
  } : tensor<?xi64>
  return %2 : tensor<?xi64>
}

// -----

// Check handling of width-sensitive arith casts.

// CHECK-LABEL:   func.func @arith.trunci(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func.func @arith.trunci(%arg0: i64) -> i32 {
  %0 = arith.trunci %arg0 : i64 to i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @arith.extui(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func.func @arith.extui(%arg0: i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL:   func.func @arith.extsi(
// CHECK-SAME:            %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
func.func @arith.extsi(%arg0: i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// -----

// Check: ml_program is also handled.

// CHECK: ml_program.global
// CHECK-SAME: i32
"ml_program.global"() {sym_name = "_v", sym_visibility = "private", type = tensor<2x2xi64>, value = dense<1> : tensor<2x2xi64>} : () -> ()
func.func @run() -> tensor<2x2xi64> {
  %0 = "ml_program.global_load"() {global = @_v} : () -> tensor<2x2xi64>
  %1 = call @f(%0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
  return %1 : tensor<2x2xi64>
}

func.func private @f(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  return %arg0 : tensor<2x2xi64>
}

