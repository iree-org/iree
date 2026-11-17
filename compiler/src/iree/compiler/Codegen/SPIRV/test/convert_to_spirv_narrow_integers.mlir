// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-convert-to-spirv)' %s | FileCheck %s

// Without Int8, i8 values are held in the low bits of i32 words. Operations
// whose result depends on the high bits first extend their operands by
// signedness: signed ones sign-extend from bit 7.

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL: spirv.func @narrow_divsi
  //  CHECK-SAME:   (%[[A:.+]]: i32, %[[B:.+]]: i32)
  //       CHECK:   %[[A_SHL:.+]] = spirv.ShiftLeftLogical %[[A]], %{{.+}}
  //       CHECK:   %[[A_EXT:.+]] = spirv.ShiftRightArithmetic %[[A_SHL]], %{{.+}}
  //       CHECK:   %[[B_SHL:.+]] = spirv.ShiftLeftLogical %[[B]], %{{.+}}
  //       CHECK:   %[[B_EXT:.+]] = spirv.ShiftRightArithmetic %[[B_SHL]], %{{.+}}
  //       CHECK:   %[[DIV:.+]] = spirv.SDiv %[[A_EXT]], %[[B_EXT]] : i32
  //       CHECK:   %[[LOW:.+]] = spirv.BitwiseAnd %[[DIV]], %{{.+}} : i32
  //       CHECK:   spirv.ReturnValue %[[LOW]]
  func.func @narrow_divsi(%a: i8, %b: i8) -> i8 {
    %0 = arith.divsi %a, %b : i8
    return %0 : i8
  }
}

// -----

// A signed comparison compares sign-extended operands.
module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL: spirv.func @narrow_cmpi_slt
  //  CHECK-SAME:   (%[[A:.+]]: i32, %[[B:.+]]: i32)
  //       CHECK:   %[[A_SHL:.+]] = spirv.ShiftLeftLogical %[[A]], %{{.+}}
  //       CHECK:   %[[A_EXT:.+]] = spirv.ShiftRightArithmetic %[[A_SHL]], %{{.+}}
  //       CHECK:   %[[B_SHL:.+]] = spirv.ShiftLeftLogical %[[B]], %{{.+}}
  //       CHECK:   %[[B_EXT:.+]] = spirv.ShiftRightArithmetic %[[B_SHL]], %{{.+}}
  //       CHECK:   %[[LT:.+]] = spirv.SLessThan %[[A_EXT]], %[[B_EXT]] : i32
  //       CHECK:   spirv.ReturnValue %[[LT]]
  func.func @narrow_cmpi_slt(%a: i8, %b: i8) -> i1 {
    %0 = arith.cmpi slt, %a, %b : i8
    return %0 : i1
  }
}

// -----

// An unsigned division masks its operands to their logical bits.
module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL: spirv.func @narrow_divui
  //  CHECK-SAME:   (%[[A:.+]]: i32, %[[B:.+]]: i32)
  //       CHECK:   %[[A_EXT:.+]] = spirv.BitwiseAnd %[[A]], %{{.+}} : i32
  //       CHECK:   %[[B_EXT:.+]] = spirv.BitwiseAnd %[[B]], %{{.+}} : i32
  //       CHECK:   %[[DIV:.+]] = spirv.UDiv %[[A_EXT]], %[[B_EXT]] : i32
  //       CHECK:   %[[LOW:.+]] = spirv.BitwiseAnd %[[DIV]], %{{.+}} : i32
  //       CHECK:   spirv.ReturnValue %[[LOW]]
  func.func @narrow_divui(%a: i8, %b: i8) -> i8 {
    %0 = arith.divui %a, %b : i8
    return %0 : i8
  }
}
