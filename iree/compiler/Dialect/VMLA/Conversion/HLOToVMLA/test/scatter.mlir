// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s -verify-diagnostics | IreeFileCheck %s

// CHECK-LABEL: @scatter_update_1D(
// CHECK-SAME: [[SRC:arg0]]: !vmla.buffer,
// CHECK-SAME: [[INDICES:arg1]]: !vmla.buffer,
// CHECK-SAME: [[UPDATES:arg2]]: !vmla.buffer
func @scatter_update_1D(%arg0: tensor<8xi32>, %arg1: tensor<3x1xi32>, %arg2: tensor<3xi32>) -> tensor<8xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[BUFFER:%.+]] = vmla.buffer.alloc
  // CHECK: vmla.copy
  // CHECK-SAME: [[BUFFER]]
  // CHECK: vmla.scatter
  // CHECK-SAME: [[BUFFER]]
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<[]> : tensor<0xi64>
    },
    unique_indices = false
  } : (tensor<8xi32>, tensor<3x1xi32>, tensor<3xi32>) -> tensor<8xi32>
  // CHECK: return [[BUFFER]]
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @scatter_update_2D
func @scatter_update_2D(%arg0: tensor<4x3xi32>, %arg1: tensor<3x2xi32>, %arg2: tensor<3xi32>) -> tensor<4x3xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[BUFFER:%.+]] = vmla.buffer.alloc
  // CHECK: vmla.copy
  // CHECK-SAME: [[BUFFER]]
  // CHECK: vmla.scatter
  // CHECK-SAME: [[BUFFER]]
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
      scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
      update_window_dims = dense<[]> : tensor<0xi64>},
    unique_indices = false
  } : (tensor<4x3xi32>, tensor<3x2xi32>, tensor<3xi32>) -> tensor<4x3xi32>
  // CHECK: return [[BUFFER]]
  return %0 : tensor<4x3xi32>
}

// -----

// CHECK-LABEL: @scatter_update_2D_slice
func @scatter_update_2D_slice(%arg0: tensor<4x3xi32>, %arg1: tensor<3x1xi32>, %arg2: tensor<3x3xi32>) -> tensor<4x3xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[BUFFER:%.+]] = vmla.buffer.alloc
  // CHECK: vmla.copy
  // CHECK-SAME: [[BUFFER]]
  // CHECK: vmla.scatter
  // CHECK-SAME: [[BUFFER]]
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<1> : tensor<1xi64>},
    unique_indices = false
  } : (tensor<4x3xi32>, tensor<3x1xi32>, tensor<3x3xi32>) -> tensor<4x3xi32>
  // CHECK: return [[BUFFER]]
  return %0 : tensor<4x3xi32>
}
