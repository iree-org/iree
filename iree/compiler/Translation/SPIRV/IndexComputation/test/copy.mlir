// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

module {
   // CHECK: func @simple_load_store
   // CHECK: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x42xi32>
   // CHECK-SAME: iree.index_computation_info
   // CHECK-SAME: operand_indices
   // CHECK-SAME: []
   // CHECK-SAME: result_index
   // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
  func @simple_load_store(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: iree.load_input
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: xla_hlo.copy
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (d1, d0)>]
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}
