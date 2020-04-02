// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

module {
  // CHECK: func @foo
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<5x1x10xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1)[s0] -> (s0, 0, d0)>]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: memref<i64>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (0)>]
  // CHECK-SAME: iree.symbol_number_info
  // CHECK-SAME: [affine_map<(d0, d1) -> (0)>, 0 : i32]
  func @foo(%arg0: memref<5x1x10xf32>, %arg1: memref<i64>, %arg2: memref<1x10xf32>)
  attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<5x1x10xf32>) : tensor<5x1x10xf32>
    %1 = iree.load_input(%arg1 : memref<i64>) : tensor<i64>
    %2 = "xla_hlo.gather"(%0, %1) {
      dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
      },
      slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>
    } : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
    iree.store_output(%2 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
    return
  }
}
