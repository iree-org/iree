// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/torch.pdl.mlir}, cse)" %s | FileCheck %s

// CHECK-LABEL:   stream.executable private @mlp_external_executable
//       CHECK:   stream.executable.export public @mlp_external_entry_point
//       CHECK:   builtin.module
//       CHECK:     func.func private @mlp_external
//  CHECK-SAME:         (memref<f32>, index, memref<f32>, index, memref<f32>, index, i32, i32, i32)
//  CHECK-SAME:         attributes {llvm.bareptr = [true]}
//       CHECK:     func.func @mlp_external_entry_point
//  CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG8:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG9:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG10:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG11:[a-zA-Z0-9]+]]: index
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[STREAM0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG6]], %[[ARG7]]}
//  CHECK-NEXT:       %[[STREAM0_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM0]]
//       CHECK:       %[[STREAM1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG8]], %[[ARG9]]}
//  CHECK-NEXT:       %[[STREAM1_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM1]]
//       CHECK:       %[[STREAM2:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG10]], %[[ARG11]]}
//  CHECK-NEXT:       %[[STREAM2_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM2]]
//       CHECK:       call @mlp_external
//  CHECK-SAME:           %[[STREAM0_BASE]], %[[C0]], %[[STREAM1_BASE]], %[[C0]], %[[STREAM2_BASE]], %[[C0]], %[[ARG3]], %[[ARG4]], %[[ARG5]]

//       CHECK:     func.func @mlp_invocation
//  CHECK-SAME:         (%[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>, %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//   CHECK-DAG:       %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:       %[[C1:.+]] = arith.constant 1
//       CHECK:       %[[M:.+]] = tensor.dim %[[LHS]], %[[C0]]
//       CHECK:       %[[N:.+]] = tensor.dim %[[RHS]], %[[C1]]
//       CHECK:       %[[K:.+]] = tensor.dim %[[LHS]], %[[C1]]
//       CHECK:       %[[M_I32:.+]] = arith.index_cast %[[M]] : index to i32
//       CHECK:       %[[N_I32:.+]] = arith.index_cast %[[N]] : index to i32
//       CHECK:       %[[K_I32:.+]] = arith.index_cast %[[K]] : index to i32
//       CHECK:       %[[K_0:.+]] = tensor.dim %[[RHS]], %[[C0]]
//       CHECK:       %[[RESULT:.+]] = flow.dispatch
//  CHECK-SAME:           @mlp_external_executable::@mlp_external_entry_point
//  CHECK-SAME:           (%[[LHS]], %[[RHS]], %[[M_I32]], %[[N_I32]], %[[K_I32]], %[[M]], %[[K]], %[[K_0]], %[[N]], %[[M]], %[[N]])
//       CHECK:       linalg.generic
//  CHECK-SAME:           ins(%[[RESULT]] :

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mlp_invocation(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %torch_lhs = torch_c.from_builtin_tensor %lhs : tensor<?x?xf32> -> !torch.vtensor<[?, ?], f32>
  %torch_rhs = torch_c.from_builtin_tensor %rhs : tensor<?x?xf32> -> !torch.vtensor<[?, ?], f32>
  %mm = torch.aten.mm %torch_lhs, %torch_rhs
      : !torch.vtensor<[?, ?], f32>, !torch.vtensor<[?, ?], f32> -> !torch.vtensor<[?, ?], f32>
  %relu = torch.aten.relu %mm : !torch.vtensor<[?, ?], f32> -> !torch.vtensor<[?, ?], f32>
  %cast= torch_c.to_builtin_tensor %relu : !torch.vtensor<[?, ?], f32> ->  tensor<?x?xf32>
  %negf = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types  = ["parallel", "parallel"]}
      ins(%cast : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %0 = arith.negf %b0 : f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %negf : tensor<?x?xf32>
}
