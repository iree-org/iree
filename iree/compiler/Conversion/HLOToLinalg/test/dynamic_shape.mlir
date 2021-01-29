// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

func @dynamic_shape(%operand: tensor<?x?xf32>) -> (tensor<?x?xf32>)
attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.exponential"(%operand) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @dynamic_shape
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>
//      CHECK:   %[[C0:.+]] = constant 0 : index
//      CHECK:   %[[T0:.+]] = dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[C1:.+]] = constant 1 : index
//      CHECK:   %[[T1:.+]] = dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[T2:.+]] = linalg.init_tensor [%[[T0]], %[[T1]]]
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:     ins(%[[ARG0]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x?xf32>)
// CHECK-NEXT:     ^{{.+}}(%[[OPERAND_IN:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:       %[[RESULT:.+]] = exp %[[OPERAND_IN]] : f32
// CHECK-NEXT:       linalg.yield %[[RESULT]] : f32
//      CHECK:   return %[[T3]]

