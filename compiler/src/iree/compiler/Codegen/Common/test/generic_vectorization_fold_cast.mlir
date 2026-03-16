// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{fold-cast-into-contract=true}))" --split-input-file %s | FileCheck %s

// Tests for the iree-codegen-generic-vectorization pass with
// fold-cast-into-contract enabled. Verifies that arith.extf operations are
// folded into vector.contract operands rather than kept separate.

func.func @matmul(%lhs: tensor<3x4xf16>, %rhs: tensor<4x5xf16>, %acc: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<3x4xf16>, tensor<4x5xf16>) outs(%acc: tensor<3x5xf32>) -> tensor<3x5xf32>
  return %result: tensor<3x5xf32>
}
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[RES:.+]] = vector.contract {{.+}} %[[LHS_VEC]], %[[RHS_VEC]], %[[OUT_VEC]]
