// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | FileCheck %s

// CHECK: [[MAP0:\#.*]] = ([[DIM0:d.*]], [[DIM1:d.*]]) -> ([[DIM1]], [[DIM0]])

module {
   // CHECK: func {{@.*}}({{%.*}}: memref<12x42xi32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @simple_load_store(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: {{%.*}} = iree.load_input({{%.*}} : memref<12x42xi32>) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: {{%.*}} = "xla_hlo.copy"({{%.*}}) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}
