// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-quantized-matmul-to-matmul))" --split-input-file %s | FileCheck %s

// Tests -iree-linalg-quantized-matmul-to-matmul, converting linalg.quantized_matmul
// ops to linalg.matmul ops plus additional arithmetic to account for any
// nonzero zero-point.

func.func @quantized_matmul_both_zp_0_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>, %acc : tensor<?x?xi32>) -> tensor<?x?xi32> {
    %lhs_zp = arith.constant 0 : i32
    %rhs_zp = arith.constant 0 : i32
    %1 = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) outs(%acc : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @quantized_matmul_both_zp_0_dynamic
// CHECK-SAME:    %[[LHS:.+]]: tensor<?x?xi8>, %[[RHS:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ACC:.+]]: tensor<?x?xi32>
// CHECK:       %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xi8>, tensor<?x?xi8>) outs(%[[ACC]] : tensor<?x?xi32>)
// CHECK:       return %[[MATMUL]]
// -----

func.func @quantized_matmul_lhs_zp_0_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>, %rhs_zp : i32, %acc : tensor<?x?xi32>) -> tensor<?x?xi32> {
    %lhs_zp = arith.constant 0 : i32
    %1 = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) outs(%acc : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @quantized_matmul_lhs_zp_0_dynamic
// CHECK-SAME:    %[[LHS:.+]]: tensor<?x?xi8>, %[[RHS:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[RHS_ZP:.+]]: i32
// CHECK-SAME:    %[[ACC:.+]]: tensor<?x?xi32>
// CHECK:       %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:       %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xi8>, tensor<?x?xi8>) outs(%[[ACC]] : tensor<?x?xi32>)
// CHECK-DAG:   %[[INIT_RESULT:.+]] = tensor.empty
// CHECK-DAG:   %[[INIT_LHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_LHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_LHS_SUMS_ACC]] :
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[LHS]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%[[ZERO_LHS_SUMS_ACC]] : tensor<?xi32>)
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel"
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[RHS_ZP]] : tensor<?x?xi32>, tensor<?xi32>, i32)
// CHECK-SAME:    outs(%[[INIT_RESULT]] : tensor<?x?xi32>)
// CHECK:       return %[[RESULT]]
// -----

func.func @quantized_matmul_rhs_zp_0_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>, %lhs_zp : i32, %acc : tensor<?x?xi32>) -> tensor<?x?xi32> {
    %rhs_zp = arith.constant 0 : i32
    %1 = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) outs(%acc : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @quantized_matmul_rhs_zp_0_dynamic
// CHECK-SAME:    %[[LHS:.+]]: tensor<?x?xi8>, %[[RHS:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[LHS_ZP:.+]]: i32
// CHECK-SAME:    %[[ACC:.+]]: tensor<?x?xi32>
// CHECK:       %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:       %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xi8>, tensor<?x?xi8>) outs(%[[ACC]] : tensor<?x?xi32>)
// CHECK-DAG:   %[[INIT_RESULT:.+]] = tensor.empty
// CHECK-DAG:   %[[INIT_RHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_RHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_RHS_SUMS_ACC]] :
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[RHS]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%[[ZERO_RHS_SUMS_ACC]] : tensor<?xi32>)
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel"
// CHECK-SAME:    ins(%[[MATMUL]], %[[RHS_SUMS]], %[[LHS_ZP]] : tensor<?x?xi32>, tensor<?xi32>, i32)
// CHECK-SAME:    outs(%[[INIT_RESULT]] : tensor<?x?xi32>)
// CHECK:       return %[[RESULT]]
// -----

func.func @quantized_matmul_neither_zp_0_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>, %lhs_zp : i32, %rhs_zp : i32, %acc : tensor<?x?xi32>) -> tensor<?x?xi32> {
    %1 = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) outs(%acc : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @quantized_matmul_neither_zp_0_dynamic
// CHECK-SAME:    %[[LHS:.+]]: tensor<?x?xi8>, %[[RHS:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[LHS_ZP:.+]]: i32, %[[RHS_ZP:.+]]: i32
// CHECK-SAME:    %[[ACC:.+]]: tensor<?x?xi32>
// CHECK-DAG:   %[[C1_INDEX:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:       %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xi8>, tensor<?x?xi8>) outs(%[[ACC]] : tensor<?x?xi32>)
// CHECK-DAG:   %[[INIT_RESULT:.+]] = tensor.empty
// CHECK-DAG:   %[[INIT_LHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_LHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_LHS_SUMS_ACC]] :
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[LHS]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%[[ZERO_LHS_SUMS_ACC]] : tensor<?xi32>)
// CHECK:       %[[INIT_RHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_RHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_RHS_SUMS_ACC]] :
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[RHS]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%[[ZERO_RHS_SUMS_ACC]] : tensor<?xi32>)
// CHECK:       %[[LHS_ZP_TIMES_RHS_ZP:.+]] = arith.muli %[[LHS_ZP]], %[[RHS_ZP]]
// CHECK:       %[[K_SIZE:.+]] = tensor.dim %[[LHS]], %[[C1_INDEX]]
// CHECK:       %[[K_SIZE_I32:.+]] = arith.index_cast %[[K_SIZE]] : index to i32
// CHECK:       %[[PRODUCT_TERM:.+]] = arith.muli %[[LHS_ZP_TIMES_RHS_ZP]], %[[K_SIZE_I32]]
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel"
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[RHS_ZP]], %[[RHS_SUMS]], %[[LHS_ZP]], %[[PRODUCT_TERM]] : tensor<?x?xi32>, tensor<?xi32>, i32, tensor<?xi32>, i32, i32)
// CHECK-SAME:    outs(%[[INIT_RESULT]] : tensor<?x?xi32>)
// CHECK:       return %[[RESULT]]
// -----

func.func @quantized_matmul_neither_zp_0_3x4x5(%lhs : tensor<3x4xi8>, %rhs : tensor<4x5xi8>, %lhs_zp : i32, %rhs_zp : i32, %acc : tensor<3x5xi32>) -> tensor<3x5xi32> {
    %1 = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) outs(%acc : tensor<3x5xi32>) -> tensor<3x5xi32>
    return %1 : tensor<3x5xi32>
}
// CHECK-LABEL: func.func @quantized_matmul_neither_zp_0_3x4x5
// CHECK-SAME:    %[[LHS:.+]]: tensor<3x4xi8>, %[[RHS:.+]]: tensor<4x5xi8>
// CHECK-SAME:    %[[LHS_ZP:.+]]: i32, %[[RHS_ZP:.+]]: i32
// CHECK-SAME:    %[[ACC:.+]]: tensor<3x5xi32>
// CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-DAG:   %[[C4_I32:.+]] = arith.constant 4 : i32
// CHECK:       %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<3x4xi8>, tensor<4x5xi8>) outs(%[[ACC]] : tensor<3x5xi32>)
// CHECK-DAG:   %[[INIT_RESULT:.+]] = tensor.empty
// CHECK-DAG:   %[[INIT_LHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_LHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_LHS_SUMS_ACC]] :
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[LHS]] : tensor<3x4xi8>)
// CHECK-SAME:    outs(%[[ZERO_LHS_SUMS_ACC]] : tensor<3xi32>)
// CHECK:       %[[INIT_RHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_RHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_RHS_SUMS_ACC]] :
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "reduction"
// CHECK-SAME:    ins(%[[RHS]] : tensor<4x5xi8>)
// CHECK-SAME:    outs(%[[ZERO_RHS_SUMS_ACC]] : tensor<5xi32>)
// CHECK:       %[[LHS_ZP_TIMES_RHS_ZP:.+]] = arith.muli %[[LHS_ZP]], %[[RHS_ZP]]
// CHECK:       %[[PRODUCT_TERM:.+]] = arith.muli %[[LHS_ZP_TIMES_RHS_ZP]], %[[C4_I32]]
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel"
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[RHS_ZP]], %[[RHS_SUMS]], %[[LHS_ZP]], %[[PRODUCT_TERM]] : tensor<3x5xi32>, tensor<3xi32>, i32, tensor<5xi32>, i32, i32)
// CHECK-SAME:    outs(%[[INIT_RESULT]] : tensor<3x5xi32>)
// CHECK:       return %[[RESULT]]
// -----

func.func @quantized_batch_matmul_both_zp_0_dynamic(%lhs : tensor<?x?x?xi8>, %rhs : tensor<?x?x?xi8>, %acc : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
    %lhs_zp = arith.constant 0 : i32
    %rhs_zp = arith.constant 0 : i32
    %1 = linalg.quantized_batch_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?x?xi8>, tensor<?x?x?xi8>, i32, i32) outs(%acc : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    return %1 : tensor<?x?x?xi32>
}
// CHECK-LABEL: func.func @quantized_batch_matmul_both_zp_0_dynamic
// CHECK-SAME:    %[[LHS:.+]]: tensor<?x?x?xi8>, %[[RHS:.+]]: tensor<?x?x?xi8>
// CHECK-SAME:    %[[ACC:.+]]: tensor<?x?x?xi32>
// CHECK:       %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?x?xi8>, tensor<?x?x?xi8>) outs(%[[ACC]] : tensor<?x?x?xi32>)
// CHECK:       return %[[MATMUL]]
// -----

func.func @quantized_batch_matmul_neither_zp_0_2x3x4x5(%lhs : tensor<2x3x4xi8>, %rhs : tensor<2x4x5xi8>, %lhs_zp : i32, %rhs_zp : i32, %acc : tensor<2x3x5xi32>) -> tensor<2x3x5xi32> {
    %1 = linalg.quantized_batch_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<2x3x4xi8>, tensor<2x4x5xi8>, i32, i32) outs(%acc : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
    return %1 : tensor<2x3x5xi32>
}
// CHECK-LABEL: func.func @quantized_batch_matmul_neither_zp_0_2x3x4x5
// CHECK-SAME:    %[[LHS:.+]]: tensor<2x3x4xi8>, %[[RHS:.+]]: tensor<2x4x5xi8>
// CHECK-SAME:    %[[LHS_ZP:.+]]: i32, %[[RHS_ZP:.+]]: i32
// CHECK-SAME:    %[[ACC:.+]]: tensor<2x3x5xi32>
// CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-DAG:   %[[C4_I32:.+]] = arith.constant 4 : i32
// CHECK:       %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[LHS]], %[[RHS]] : tensor<2x3x4xi8>, tensor<2x4x5xi8>) outs(%[[ACC]] : tensor<2x3x5xi32>)
// CHECK-DAG:   %[[INIT_RESULT:.+]] = tensor.empty
// CHECK-DAG:   %[[INIT_LHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_LHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_LHS_SUMS_ACC]] :
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel", "reduction"
// CHECK-SAME:    ins(%[[LHS]] : tensor<2x3x4xi8>)
// CHECK-SAME:    outs(%[[ZERO_LHS_SUMS_ACC]] : tensor<2x3xi32>)
// CHECK:       %[[INIT_RHS_SUMS_ACC:.+]] = tensor.empty
// CHECK:       %[[ZERO_RHS_SUMS_ACC:.+]] = linalg.fill
// CHECK-SAME:    ins(%[[C0_I32]] :
// CHECK-SAME:    outs(%[[INIT_RHS_SUMS_ACC]] :
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel", "reduction"
// CHECK-SAME:    ins(%[[RHS]] : tensor<2x4x5xi8>)
// CHECK-SAME:    outs(%[[ZERO_RHS_SUMS_ACC]] : tensor<2x5xi32>)
// CHECK:       %[[LHS_ZP_TIMES_RHS_ZP:.+]] = arith.muli %[[LHS_ZP]], %[[RHS_ZP]]
// CHECK:       %[[PRODUCT_TERM:.+]] = arith.muli %[[LHS_ZP_TIMES_RHS_ZP]], %[[C4_I32]]
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    "parallel", "parallel", "parallel"
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[RHS_ZP]], %[[RHS_SUMS]], %[[LHS_ZP]], %[[PRODUCT_TERM]] : tensor<2x3x5xi32>, tensor<2x3xi32>, i32, tensor<2x5xi32>, i32, i32)
// CHECK-SAME:    outs(%[[INIT_RESULT]] : tensor<2x3x5xi32>)
// CHECK:       return %[[RESULT]]
// -----
