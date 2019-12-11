// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

// CHECK: [[MAP0:\#.*]] = ({{d.*}}, {{d.*}}, {{d.*}}) -> (0)

module {
  // CHECK: func @extract_element
  // CHECK-SAME: {{%.*}}: memref<i1> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @extract_element(%arg0: memref<i1>, %arg1: memref<i1>)
    attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<1> : tensor<3xi32>, iree.num_dims = 3 : i32, iree.ordinal = 0 : i32} {
    // CHECK: iree.load_input
    // CHECK-SAME: {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %0 = "iree.load_input"(%arg0) : (memref<i1>) -> tensor<i1>
    // CHECK: extract_element
    // CHECK-SAME: {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %1 = "std.extract_element"(%0) : (tensor<i1>) -> i1
    "iree.store_output"(%1, %arg1) : (i1, memref<i1>) -> ()
    "iree.return"() : () -> ()
  }
}
