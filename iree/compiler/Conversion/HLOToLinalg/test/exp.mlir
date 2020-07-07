// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @exp
func @exp(%operand: tensor<2x2xf32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.exponential"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN:.+]]: f32):
// CHECK-NEXT:   %[[RESULT:.+]] = exp %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }: tensor<2x2xf32> -> tensor<2x2xf32>
