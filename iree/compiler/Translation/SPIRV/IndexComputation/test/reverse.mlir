// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: func @reverse_2d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: memref<12x12xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (-d1 + 11, -d0 + 11)>]
  func @reverse_2d(%arg0: memref<12x12xf32>, %arg1 : memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    // CHECK: xla_hlo.reverse
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (-d1 + 11, -d0 + 11)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%1 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @reverse_3d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<3x3x3xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d2, -d1 + 2, d0)>]
  func @reverse_3d(%arg0: memref<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<3x3x3xf32>) : tensor<3x3x3xf32>
    // CHECK: xla_hlo.reverse
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d2, -d1 + 2, d0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d2, d1, d0)>]
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    iree.store_output(%1 : tensor<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
    iree.return
  }
}
