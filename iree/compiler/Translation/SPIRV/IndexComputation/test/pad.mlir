// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1 - 4, d0 - 5)
// CHECK-DAG: [[MAP2:#.*]] = (d0, d1) -> (d1, d0)

module {
  // CHECK: func @pad_zero_interior
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32> {iree.index_computation_info = {{\[\[}}[[MAP1]]{{\]\]}}}
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<18x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 18, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP2]], [[MAP1]], [[MAP0]]{{\]\]}}
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    iree.store_output(%2 : tensor<18x12xf32>, %arg1 : memref<18x12xf32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1, d0)

module {
  // CHECK: func @pad_no_op
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32> {iree.index_computation_info = {{\[\[}}[[MAP1]]{{\]\]}}}
  func @pad_no_op(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[4, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP1]], [[MAP0]]{{\]\]}}
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>
    iree.store_output(%2 : tensor<12x4xf32>, %arg1 : memref<12x4xf32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1 floordiv 2 - 2, (d0 - 5) floordiv 3)
// CHECK-DAG: [[MAP2:#.*]] = (d0, d1) -> (d1, d0)

module {
  // CHECK: func @pad_zero_interior
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x4xf32> {iree.index_computation_info = {{\[\[}}[[MAP1]]{{\]\]}}}
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<29x18xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[18, 29, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: xla_hlo.pad
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP2]], [[MAP1]], [[MAP0]]{{\]\]}}
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<29x18xf32>
    iree.store_output(%2 : tensor<29x18xf32>, %arg1 : memref<29x18xf32>)
    iree.return
  }
}
