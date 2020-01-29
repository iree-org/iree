// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

module {
  // CHECK: func @reduction_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<5xi32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (d0)>]
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: memref<i32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1, d2) -> (0)>]
  func @reduction_entry(%arg0: memref<5xi32>, %arg1: memref<i32>, %arg2: memref<i32>) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 0 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[5, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5xi32>)  : tensor<5xi32>
    iree.store_reduce(%0 : tensor<5xi32>, %arg2 : memref<i32>, @reduction_apply)
    iree.return
  }
}

// -----

module {
  // CHECK: func @reduction_2D_dim0_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<5x4xi32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: memref<4xi32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (d0)>]
  func @reduction_2D_dim0_entry(%arg0: memref<5x4xi32>, %arg1: memref<i32>, %arg2: memref<4xi32>) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 0 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5x4xi32>)  : tensor<5x4xi32>
    iree.store_reduce(%0 : tensor<5x4xi32>, %arg2 : memref<4xi32>, @reduction_apply)
    iree.return
  }
}

// -----

module {
  // CHECK: func @reduction_2D_dim1_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<5x4xi32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: memref<5xi32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (d1)>]
  func @reduction_2D_dim1_entry(%arg0: memref<5x4xi32>, %arg1: memref<i32>, %arg2: memref<5xi32>) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5x4xi32>)  : tensor<5x4xi32>
    iree.store_reduce(%0 : tensor<5x4xi32>, %arg2 : memref<5xi32>, @reduction_apply)
    iree.return
  }
}
