// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1) -> (0, d1 - 64)>

module {
  // CHECK: func @concatenate
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<1x64xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: memref<1x10xf32> {iree.index_computation_info = {{\[\[}}[[MAP1]]{{\]\]}}
  func @concatenate(%arg0: memref<1x64xf32>, %arg1 : memref<1x10xf32>, %arg2 : memref<1x74xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[1, 74]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<1x64xf32>) : tensor<1x64xf32>
    %1 = iree.load_input(%arg1 : memref<1x10xf32>) : tensor<1x10xf32>
    // CHECK: xla_hlo.concatenate
    // CHECK-SAME: iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]], [[MAP1]]{{\]\]}}
    %2 = "xla_hlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
    iree.store_output(%2 : tensor<1x74xf32>, %arg2 : memref<1x74xf32>)
    iree.return
  }
}
