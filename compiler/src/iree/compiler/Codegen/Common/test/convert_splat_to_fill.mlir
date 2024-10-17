// RUN: iree-opt --split-input-file --mlir-print-local-scope %s \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-convert-splat-constant-to-fill))" | FileCheck %s

func.func @tensor_splat() -> tensor<1x2x3xi32> {
  %cst = arith.constant dense<5> : tensor<1x2x3xi32>
  return %cst : tensor<1x2x3xi32>
}

// CHECK-LABEL: func.func @tensor_splat
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : i32
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x2x3xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C5]]
//  CHECK-SAME:     outs(%[[EMPTY]]
//       CHECK:   return %[[FILL]]

// -----

func.func @vector_splat() -> vector<1x2x3xi32> {
  %cst = arith.constant dense<5> : vector<1x2x3xi32>
  return %cst : vector<1x2x3xi32>
}

// Verify that non-tensor splats are not converted.
// CHECK-LABEL: func.func @vector_splat
//       CHECK:   %[[CST:.+]] = arith.constant dense<5> : vector<1x2x3xi32>
//       CHECK:   return %[[CST]]
