// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK:spv.module Logical GLSL450
  // CHECK-DAG: spv.globalVariable @[[GLOBALIDVAR:.+]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.func @[[FN:broadcast_in_dim_2D_3D]]
  // CHECK-SAME: %[[ARG0:.+]]: !spv.ptr<!spv.struct<!spv.array<504 x i32, stride=4> [0]>, StorageBuffer>
  // CHECK-SAME: %[[ARG1:.+]]: !spv.ptr<!spv.struct<!spv.array<1512 x i32, stride=4> [0]>, StorageBuffer>
  func @broadcast_in_dim_2D_3D(%arg0: memref<12x42xi32>, %arg1: memref<3x12x42xi32>)
  attributes {iree.dispatch_fn_name = "broadcast_in_dim_2D_3D"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x42xi32>) -> tensor<3x12x42xi32>
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    return
  }
}

// -----

module {
  // CHECK:spv.module Logical GLSL450
  // CHECK-DAG: spv.globalVariable @[[GLOBALIDVAR:.+]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.func @[[FN:broadcast_in_dim_scalar_3D]]
  // CHECK-SAME: %[[ARG0:.+]]: !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-SAME: %[[ARG1:.+]]: !spv.ptr<!spv.struct<!spv.array<1512 x i32, stride=4> [0]>, StorageBuffer>
  func @broadcast_in_dim_scalar_3D(%arg0: memref<i32>, %arg1: memref<3x12x42xi32>)
  attributes {iree.dispatch_fn_name = "broadcast_in_dim_scalar_3D"} {
    // CHECK: %[[ARG0LOADPTR:.+]] = spv.AccessChain %[[ARG0]]
    // CHECK: %[[VAL:.+]] = spv.Load "StorageBuffer" %[[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<i32>) : tensor<i32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<3x12x42xi32>
    // CHECK: %[[ARG1STOREPTR:.+]] = spv.AccessChain %[[ARG1]]
    // CHECK: spv.Store "StorageBuffer" %[[ARG1STOREPTR]], %[[VAL]]
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    return
  }
}

// -----

module {
  func @const_float_splat(%arg0: memref<12x42xf32>)
    attributes {iree.dispatch_fn_name = "const_float_splat"} {
    // CHECK: spv.constant 1.000000e+00 : f32
    %0 = constant dense<1.0> : tensor<12xf32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xf32>) -> tensor<12x42xf32>
    iree.store_output(%1 : tensor<12x42xf32>, %arg0 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @const_int_splat(%arg0: memref<12x42xi32>)
    attributes {iree.dispatch_fn_name = "const_int_splat"} {
    // CHECK: spv.constant 42 : i32
    %0 = constant dense<42> : tensor<12xi32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg0 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  // CHECK: spv.func @const_int_nonsplat
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: !spv.ptr<!spv.struct<!spv.array<1008 x i32, stride=4> [0]>, StorageBuffer>
  func @const_int_nonsplat(%arg0: memref<2x12x42xi32>)
    attributes {iree.dispatch_fn_name = "const_int_nonsplat"} {
    // CHECK: %[[CST:.+]] = spv.constant dense<[42, 21]>
    // CHECK: %[[VAR:.+]] = spv.Variable init(%[[CST]]) : !spv.ptr<!spv.array<2 x i32, stride=4>, Function>
    // CHECK: %[[LOADPTR:.+]] = spv.AccessChain %[[VAR]]
    // CHECK: %[[LOADVAL:.+]] = spv.Load
    %0 = constant dense<[42, 21]> : tensor<2xi32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<2x12x42xi32>
    // CHECK: %[[STOREPTR:.+]] = spv.AccessChain %[[ARG0]]
    // CHECK: spv.Store "StorageBuffer" %[[STOREPTR]], %[[LOADVAL]]
    iree.store_output(%1 : tensor<2x12x42xi32>, %arg0 : memref<2x12x42xi32>)
    return
  }
}
