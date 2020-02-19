// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: func @pad_zero_interior
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1 - 4, d0 - 5)>]
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<18x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1 - 4, d0 - 5)>]
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    iree.store_output(%2 : tensor<18x12xf32>, %arg1 : memref<18x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @pad_no_op
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices = []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
  func @pad_no_op(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>
    iree.store_output(%2 : tensor<12x4xf32>, %arg1 : memref<12x4xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @pad_with_stride
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1 floordiv 2 - 2, (d0 - 5) floordiv 3)>]
  func @pad_with_stride(%arg0 : memref<12x4xf32>, %arg1 : memref<29x18xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1 floordiv 2 - 2, (d0 - 5) floordiv 3)>]
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d1, d0)>]
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<29x18xf32>
    iree.store_output(%2 : tensor<29x18xf32>, %arg1 : memref<29x18xf32>)
    iree.return
  }
}
