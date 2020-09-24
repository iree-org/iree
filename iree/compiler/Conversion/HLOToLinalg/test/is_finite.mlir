// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @is_finite
func @is_finite(%operand: tensor<1x8x32xf32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.is_finite"(%operand) : (tensor<1x8x32xf32>) -> tensor<1x8x32xi1>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]]
// CHECK-SAME: #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:   ins(%{{[a-z0-9]*}} : tensor<1x8x32xf32>) {
// CHECK: ^bb0(%[[IN:.+]]: f32): 
// CHECK:    %[[INF_VAL:.+]] = constant 0x7F800000 : f32
// CHECK:    %[[OUT:.+]] = cmpf "one", %arg1, %[[INF_VAL]] : f32
// CHECK:    linalg.yield %[[OUT]] : i1
// CHECK: } -> tensor<1x8x32xi1>
