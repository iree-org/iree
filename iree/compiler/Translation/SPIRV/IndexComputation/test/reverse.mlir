// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (-d1 + 11, -d0 + 11)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1, d0)

module {
  // CHECK: func @reverse_2d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: memref<12x12xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @reverse_2d(%arg0: memref<12x12xf32>, %arg1 : memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    // CHECK: xla_hlo.reverse
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%1 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1, d2) -> (d2, -d1 + 2, d0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1, d2) -> (d2, d1, d0)

module {
  // CHECK: func @reverse_3d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<3x3x3xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @reverse_3d(%arg0: memref<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[3, 3, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<3x3x3xf32>) : tensor<3x3x3xf32>
    // CHECK: xla_hlo.reverse
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    iree.store_output(%1 : tensor<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
    iree.return
  }
}
