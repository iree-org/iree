// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/pdl_patterns.mlir})" %s | FileCheck %s
// This is a contrived example to show pdl pass converting two divisions and a multiplication to two multiplications and a division

// CHECK-LABEL: @two_div_one_mul
//  CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: f32, %[[ARG1:[a-zA-Z0-9]+]]: f32, %[[ARG2:[a-zA-Z0-9]+]]: f32, %[[ARG3:[a-zA-Z0-9]+]]: f32)
//       CHECK: %[[A:.+]] = arith.mulf %[[ARG0]], %[[ARG2]] : f32
//       CHECK: %[[B:.+]] = arith.mulf %[[ARG1]], %[[ARG3]] : f32
//       CHECK: %[[DIV:.+]] = arith.divf %[[A]], %[[B]] : f32
//       CHECK: return %[[DIV]] : f32

func.func @two_div_one_mul(%0 : f32, %1 : f32, %2 : f32, %3 : f32) -> f32 {
    %a = arith.divf %0, %1 : f32
    %b = arith.divf %2, %3 : f32
    %mul = arith.mulf %a, %b : f32
    return %mul : f32
}