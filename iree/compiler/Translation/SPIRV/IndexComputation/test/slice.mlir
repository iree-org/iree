// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

// CHECK-DAG: [[MAP0:#.*]] = affine_map<(d0, d1) -> (d0 floordiv 3 + 2, d0 mod 3 + 1)>
// CHECK-DAG: [[MAP1:#.*]] = affine_map<(d0, d1) -> (d0 floordiv 3, d0 mod 3)>

module {
  // CHECK: func @slice_unit_stride
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<6x6xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @slice_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    // CHECK: xla_hlo.slice
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 4]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = affine_map<(d0, d1) -> (d0 floordiv 3 + 2, (d0 mod 3) * 2 + 1)>
// CHECK-DAG: [[MAP1:#.*]] = affine_map<(d0, d1) -> (d0 floordiv 3, d0 mod 3)>

module {
  // CHECK: func @slice_non_unit_stride
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<6x6xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @slice_non_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    // CHECK: xla_hlo.slice
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 6]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}
