// RUN: iree-opt --split-input-file --iree-stablehlo-legalize-dynamic-reduce-window %s | FileCheck %s

// A window spanning the dynamic axis with size - 1 low padding is a prefix scan.
// CHECK-LABEL: func.func @prefix(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<7x?xi32>
//       CHECK:   %[[INIT:.+]] = stablehlo.constant dense<0> : tensor<i32>
//       CHECK:   %[[OUT:.+]] = tensor.empty(%{{.+}}) : tensor<7x?xi32>
//       CHECK:   %[[ACC:.+]] = linalg.generic {{.*}} ins(%[[INIT]] : tensor<i32>) outs(%{{.+}} : tensor<7xi32>)
//       CHECK:   %[[SCAN:.+]]:2 = iree_linalg_ext.scan dimension(1) inclusive(true) ins(%[[INPUT]] : tensor<7x?xi32>) outs(%[[OUT]], %[[ACC]] : tensor<7x?xi32>, tensor<7xi32>)
//       CHECK:   ^bb0(%[[A:.+]]: i32, %[[B:.+]]: i32):
//       CHECK:     %[[SUM:.+]] = arith.addi %[[A]], %[[B]] : i32
//       CHECK:     iree_linalg_ext.yield %[[SUM]] : i32
//       CHECK:   return %[[SCAN]]#0 : tensor<7x?xi32>
func.func @prefix(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// Padding size - 1 on the high side is a suffix scan: reverse, scan, reverse.
// CHECK-LABEL: func.func @suffix(
//       CHECK:   %[[REVERSED:.+]] = stablehlo.reverse %{{.+}}, dims = [1] : tensor<7x?xi32>
//       CHECK:   %[[SCAN:.+]]:2 = iree_linalg_ext.scan dimension(1) inclusive(true) ins(%[[REVERSED]] : tensor<7x?xi32>)
//       CHECK:   %[[RESULT:.+]] = stablehlo.reverse %[[SCAN]]#0, dims = [1] : tensor<7x?xi32>
//       CHECK:   return %[[RESULT]]
func.func @suffix(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %c0_i32, %n_m1 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// The reducer body is cloned whatever it computes.
// CHECK-LABEL: func.func @prefix_max(
//       CHECK:   iree_linalg_ext.scan dimension(1)
//       CHECK:     arith.maximumf
func.func @prefix_max(%input: tensor<7x?xf32>) -> tensor<7x?xf32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xf32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xf32>
  return %result : tensor<7x?xf32>
}
func.func private @reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32> {
  %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
  return %max : tensor<f32>
}

// -----

// A window one shorter than the axis is no scan.
// CHECK-LABEL: func.func @partial_window(
//   CHECK-NOT:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
func.func @partial_window(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n_m1 : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A window from another tensor's length proves nothing about this one.
// CHECK-LABEL: func.func @window_not_input_dim(
//   CHECK-NOT:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
func.func @window_not_input_dim(%input: tensor<7x?xi32>, %other: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %other_dim = tensor.dim %other, %c1 : tensor<7x?xi32>
  %other_n = arith.index_cast %other_dim : index to i32
  %window = tensor.from_elements %c1_i32, %other_n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A size truncated below 32 bits may have wrapped.
// CHECK-LABEL: func.func @narrow_window(
//   CHECK-NOT:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
func.func @narrow_window(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %narrow = arith.trunci %n : i32 to i16
  %back = arith.extsi %narrow : i16 to i32
  %window = tensor.from_elements %c1_i32, %back : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A scan drops the init, so an init the reducer does not ignore keeps the call.
// CHECK-LABEL: func.func @non_identity_init(
//   CHECK-NOT:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
func.func @non_identity_init(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<1> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A window with fewer entries than the input has dimensions is malformed.
// CHECK-LABEL: func.func @short_window(
//   CHECK-NOT:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
func.func @short_window(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32 : tensor<1xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func private @reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// JAX emits the reducer as a public function; no use remains once the window
// is a scan.
// CHECK-LABEL: func.func @public_reducer_lowered(
//       CHECK:   iree_linalg_ext.scan
//   CHECK-NOT: func.func @public_reducer(
func.func @public_reducer_lowered(%input: tensor<7x?xi32>) -> tensor<7x?xi32> {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %result = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@public_reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %result : tensor<7x?xi32>
}
func.func @public_reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A custom call left in place still names the reducer.
// CHECK-LABEL: func.func @public_reducer_still_called(
//       CHECK:   iree_linalg_ext.scan
//       CHECK:   stablehlo.custom_call @stablehlo.dynamic_reduce_window
//       CHECK: func.func @shared_reducer(
func.func @public_reducer_still_called(%input: tensor<7x?xi32>) -> (tensor<7x?xi32>, tensor<7x?xi32>) {
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %short = tensor.from_elements %c1_i32, %n_m1 : tensor<2xi32>
  %scan = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@shared_reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  %partial = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %short, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@shared_reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %scan, %partial : tensor<7x?xi32>, tensor<7x?xi32>
}
func.func @shared_reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}

// -----

// A LinalgExt body that predates the pass keeps its StableHLO ops for the
// general lowering, which handles ops the scan's body patterns do not.
// CHECK-LABEL: func.func @earlier_linalg_ext_body(
//       CHECK:   iree_linalg_ext.sort
//       CHECK:     stablehlo.round_nearest_even
//       CHECK:   iree_linalg_ext.scan
func.func @earlier_linalg_ext_body(%values: tensor<3x5xf32>, %input: tensor<7x?xi32>) -> (tensor<3x5xf32>, tensor<7x?xi32>) {
  %sorted = iree_linalg_ext.sort dimension(1) outs(%values : tensor<3x5xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %lhs_tensor = tensor.from_elements %lhs : tensor<f32>
    %rounded = stablehlo.round_nearest_even %lhs_tensor : tensor<f32>
    %lhs_rounded = tensor.extract %rounded[] : tensor<f32>
    %less = arith.cmpf olt, %lhs_rounded, %rhs : f32
    iree_linalg_ext.yield %less : i1
  } -> tensor<3x5xf32>
  %c1 = arith.constant 1 : index
  %init = stablehlo.constant dense<0> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %ones = stablehlo.constant dense<1> : tensor<2xi32>
  %zeros = stablehlo.constant dense<0> : tensor<1x2xi32>
  %dim = tensor.dim %input, %c1 : tensor<7x?xi32>
  %n = arith.index_cast %dim : index to i32
  %wide = arith.extsi %n : i32 to i64
  %wide_m1 = arith.addi %wide, %c-1_i64 : i64
  %n_m1 = arith.trunci %wide_m1 : i64 to i32
  %window = tensor.from_elements %c1_i32, %n : tensor<2xi32>
  %row = tensor.from_elements %n_m1, %c0_i32 : tensor<1x2xi32>
  %padding = stablehlo.concatenate %zeros, %row, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  %scan = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%input, %init, %window, %ones, %ones, %ones, %padding) {api_version = 2 : i32, called_computations = [@earlier_body_reducer]} : (tensor<7x?xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<7x?xi32>
  return %sorted, %scan : tensor<3x5xf32>, tensor<7x?xi32>
}
func.func private @earlier_body_reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %sum = stablehlo.add %lhs, %rhs : tensor<i32>
  return %sum : tensor<i32>
}
