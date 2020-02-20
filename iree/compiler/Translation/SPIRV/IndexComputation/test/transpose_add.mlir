// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

module {
 // CHECK: func @transpose_add
 // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x12xf32>
 // CHECK-SAME: iree.index_computation_info
 // CHECK-SAME: operand_indices
 // CHECK-SAME: []
 // CHECK-SAME: result_index
 // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
 // CHECK-SAME: operand_indices
 // CHECK-SAME: []
 // CHECK-SAME: result_index
 // CHECK-SAME: [affine_map<(d0, d1) -> (d0, d1)>]
 func @transpose_add(%arg0: memref<12x12xf32>, %arg1: memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: iree.load_input
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d0, d1)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d0, d1)>]
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    // CHECK: xla_hlo.transpose
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d0, d1)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    %1 = "xla_hlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: xla_hlo.add
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    %2 = xla_hlo.add %0, %1 : tensor<12x12xf32>
    iree.store_output(%2 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}
