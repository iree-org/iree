// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -o - %s | IreeFileCheck %s

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1, d2) -> (d1, d0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1, d2) -> (d2, d1, d0)

module {
  // CHECK: func @broadcast_in_dim_2D_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<12x42xi32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
  func @broadcast_in_dim_2D_3D(%arg0: memref<12x42xi32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x42xi32>) -> tensor<3x12x42xi32>
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1, d2) -> (0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1, d2) -> (d2, d1, d0)

module {
  // CHECK: func @broadcast_in_dim_scalar_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<i32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
  func @broadcast_in_dim_scalar_3D(%arg0: memref<i32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<i32>) : tensor<i32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) : (tensor<i32>) -> tensor<3x12x42xi32>
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (d1)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1, d0)

module {
  func @const_float_splat(%arg0: memref<12x42xf32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: constant
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
    %0 = constant dense<1.0> : tensor<12xf32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xf32>) -> tensor<12x42xf32>
    iree.store_output(%1 : tensor<12x42xf32>, %arg0 : memref<12x42xf32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (d1)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d1, d0)

module {
  func @const_int_splat(%arg0: memref<12x42xi32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: constant
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
    %0 = constant dense<42> : tensor<12xi32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg0 : memref<12x42xi32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1, d2) -> (d2)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1, d2) -> (d2, d1, d0)

module {
  func @const_int_nonsplat(%arg0: memref<2x12x42xi32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 2]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: constant
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
    %0 = constant dense<[42, 21]> : tensor<2xi32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<2x12x42xi32>
    iree.store_output(%1 : tensor<2x12x42xi32>, %arg0 : memref<2x12x42xi32>)
    iree.return
  }
}

// -----

// CHECK-DAG: [[MAP0:#.*]] = (d0, d1, d2) -> (0)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1, d2) -> (d0)

module {
  func @zero_element_1dtensor(%arg0 : memref<f32>, %arg1 : memref<4xf32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<f32>) : tensor<f32>
    // CHECK: xla_hlo.broadcast_in_dim
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP1]], [[MAP0]]{{\]\]}}
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<4xf32>
    iree.store_output(%1 : tensor<4xf32>, %arg1 : memref<4xf32>)
    iree.return
  }
}
