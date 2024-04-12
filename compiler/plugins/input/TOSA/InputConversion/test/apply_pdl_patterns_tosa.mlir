// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/tosa.pdl.mlir})" %s | FileCheck %s

// CHECK-LABEL:   stream.executable private @mlp_external_f32_f32_f32_i32_i32_i32_executable
//       CHECK:   stream.executable.export public @mlp_external_entry_point
//       CHECK:   builtin.module
//       CHECK:     func.func private @mlp_external_f32_f32_f32_i32_i32_i32
//  CHECK-SAME:         (memref<f32>, index, memref<f32>, index, memref<f32>, index, i32, i32, i32)
//  CHECK-SAME:         attributes {llvm.bareptr = [true]}
//       CHECK:     func.func @mlp_external_entry_point
//  CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]+]]: i32
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[STREAM0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<1x2x4xf32, strided<[8, 4, 1], offset: ?>>
//  CHECK-NEXT:       %[[STREAM0_BASE:[a-zA-Z0-9_]+]], %[[OFFSET0:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = iree_codegen.extract_strided_metadata %[[STREAM0]]
//       CHECK:       %[[STREAM1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<1x4x8xf32, strided<[32, 8, 1], offset: ?>>
//  CHECK-NEXT:       %[[STREAM1_BASE:[a-zA-Z0-9_]+]], %[[OFFSET1:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = iree_codegen.extract_strided_metadata %[[STREAM1]]
//       CHECK:       %[[STREAM2:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<1x2x8xf32, strided<[16, 8, 1], offset: ?>>
//  CHECK-NEXT:       %[[STREAM2_BASE:[a-zA-Z0-9_]+]], %[[OFFSET2:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = iree_codegen.extract_strided_metadata %[[STREAM2]]
//       CHECK:       call @mlp_external_f32_f32_f32_i32_i32_i32
//  CHECK-SAME:           %[[STREAM0_BASE]], %[[OFFSET0]], %[[STREAM1_BASE]], %[[OFFSET1]], %[[STREAM2_BASE]], %[[OFFSET2]], %[[ARG3]], %[[ARG4]], %[[ARG5]]

//       CHECK:     func.func @mlp_invocation
//  CHECK-SAME:         (%[[ARG0:.+]]: tensor<2x4xf32>, %[[ARG1:.+]]: tensor<4x8xf32>)
//   CHECK-DAG:       %[[C4:.+]] = arith.constant 4 : i32
//   CHECK-DAG:       %[[C8:.+]] = arith.constant 8 : i32
//   CHECK-DAG:       %[[C2:.+]] = arith.constant 2 : i32
//   CHECK-DAG:       %[[LHS:.+]] = tosa.reshape %[[ARG0]]
//   CHECK-DAG:       %[[RHS:.+]] = tosa.reshape %[[ARG1]]
//       CHECK:       %[[RESULT:.+]] = flow.dispatch
//  CHECK-SAME:           @mlp_external_f32_f32_f32_i32_i32_i32_executable::@mlp_external_entry_point
//  CHECK-SAME:           (%[[LHS]], %[[RHS]], %[[C2]], %[[C8]], %[[C4]])
//       CHECK:       tosa.negate %[[RESULT]]

func.func @mlp_invocation(%lhs: tensor<2x4xf32>, %rhs : tensor<4x8xf32>) -> tensor<2x8xf32> {
  %lhs_3D = tosa.reshape %lhs {new_shape = array<i64 : 1, 2, 4>} : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
  %rhs_3D = tosa.reshape %rhs {new_shape = array<i64 : 1, 4, 8>} : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
  %0 = tosa.matmul %lhs_3D, %rhs_3D : (tensor<1x2x4xf32>, tensor<1x4x8xf32>) -> tensor<1x2x8xf32>
  %1 = tosa.clamp %0 {
      min_int = 0 : i64, max_int = 9223372036854775807 : i64,
      min_fp = 0.0 : f32, max_fp = 3.4028235e+38 : f32}
      : (tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
  %2 = tosa.negate %1 : (tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64 : 2, 8>}  : (tensor<1x2x8xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>
}
