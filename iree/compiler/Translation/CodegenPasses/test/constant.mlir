// RUN: iree-opt -iree-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: func @const
func @const() -> tensor<5xi32> attributes {iree.dispatch_fn_name = ""} {
  // CHECK: linalg.generic
  // CHECK-SAME: args_in = 0
  // CHECK-SAME: args_out = 1
  // CHECK-SAME: indexing_maps = [#[[MAP]]],
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK: %[[CONST:.+]] = constant 1 : i32
  // CHECK-NEXT: linalg.yield %[[CONST]]
  %0 = constant dense<1> : tensor<5xi32>
  return %0 : tensor<5xi32>
}
