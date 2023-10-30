// RUN: iree-opt --iree-codegen-generic-vectorization --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-codegen-generic-vectorization="fold-cast-into-contract=true" --split-input-file %s | FileCheck %s -check-prefix=CHECK-FOLD

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
// CHECK:         %[[EXT_LHS:.+]] = arith.extf %[[LHS_VEC]]
// CHECK:         %[[EXT_RHS:.+]] = arith.extf %[[RHS_VEC]]
// CHECK:         %[[RES:.+]] = vector.contract {{.+}} %[[EXT_LHS]], %[[EXT_RHS]], %[[OUT_VEC]]

// CHECK-FOLD-LABEL: func.func @matmul
// CHECK-FOLD-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-FOLD-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-FOLD-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK-FOLD:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]
// CHECK-FOLD:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]
// CHECK-FOLD:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]
// CHECK-FOLD:         %[[RES:.+]] = vector.contract {{.+}} %[[LHS_VEC]], %[[RHS_VEC]], %[[OUT_VEC]]
