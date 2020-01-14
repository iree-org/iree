// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -o - %s | IreeFileCheck %s

// CHECK-DAG: [[MAP0:#.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: [[MAP1:#.*]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

module {
  // CHECK: func @broadcast_2D_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x42xi32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
  func @broadcast_2D_3D(%arg0: memref<12x42xi32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: xla_hlo.broadcast
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast"(%0) {broadcast_sizes = dense<[3]> : tensor<1xi64>} : (tensor<12x42xi32>) -> tensor<3x12x42xi32>
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = affine_map<(d0, d1, d2) -> (0)>
// CHECK-DAG: [[MAP1:#.*]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

module {
  // CHECK: func @broadcast_scalar_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<i32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
  func @broadcast_scalar_3D(%arg0: memref<i32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<i32>) : tensor<i32>
    // CHECK: xla_hlo.broadcast
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast"(%0) {broadcast_sizes = dense<[3, 12, 42]>: tensor<3xi64>} : (tensor<i32>) -> tensor<3x12x42xi32>
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}
