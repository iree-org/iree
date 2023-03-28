// RUN: iree-opt \
// RUN:   --split-input-file \
// RUN:   --iree-plugin=split_mlir \
// RUN:   --pass-pipeline="builtin.module(iree-outline-functions)" %s \
// RUN: | FileCheck --dump-input-context=100 %s

// Outline op that does not take any arguments and is not used anywhere.
// CHECK-LABEL: func.func @no_args_and_result
func.func @no_args_and_result() {
//       CHECK: call @no_args_and_result_outline_0_0() : () -> ()
    %cts1 = mhlo.constant {outline_range_first, outline_range_last} dense<1.000000e+00> : tensor<2xf32>
//  CHECK-NEXT: {{return$}}
  return
}
// CHECK-LABEL: func.func @no_args_and_result_outline_0_0()
//       CHECK: mhlo.constant dense<{{.+}}> : tensor<2xf32>
//   CHECK-NOT: outline_range_first
//   CHECK-NOT: outline_range_last
//  CHECK-NEXT: {{return$}}

// -----

// Outline an op that takes one argument and has one result that is used.
// CHECK-LABEL: func.func @one_arg_and_one_result
//  CHECK-SAME: ([[ARG0:%.+]]: tensor<2xf32>) -> tensor<2xf32>
func.func @one_arg_and_one_result(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//  CHECK-NEXT: [[RES0:%.+]] = call @one_arg_and_one_result_outline_0_0([[ARG0]])
  %res = mhlo.cosine %arg0 {outline_range_first, outline_range_last} : tensor<2xf32>
//  CHECK-NEXT: return [[RES0]] : tensor<2xf32>
  return %res : tensor<2xf32>
}
// CHECK-LABEL: func.func @one_arg_and_one_result_outline_0_0
//  CHECK-SAME: ([[ARG1:%.+]]: tensor<2xf32>) -> tensor<2xf32>
//  CHECK-NEXT: [[RES1:%.+]] = mhlo.cosine [[ARG1]] : tensor<2xf32>
//  CHECK-NEXT: return [[RES1]] : tensor<2xf32>

// -----

// Multiple ops in a range with multiple arguments and results.
// CHECK-LABEL: func.func @multiple_ops
//  CHECK-SAME: ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> (i32, i32)
func.func @multiple_ops(%arg0: i32, %arg1: i32) -> (i32, i32) {
//  CHECK-NEXT: [[RES0:%.+]]:2 = call @multiple_ops_outline_0_0([[ARG0]], [[ARG1]]) : (i32, i32) -> (i32, i32) 
  %add = arith.addi %arg0, %arg0 {outline_range_first} : i32
  %mul = arith.muli %add, %arg1 {outline_range_last} : i32
//  CHECK-NEXT: return [[RES0]]#0, [[RES0]]#1 : i32, i32
  return %add, %mul : i32, i32
}
// CHECK-LABEL: func.func @multiple_ops_outline_0_0
//  CHECK-SAME: ([[ARG10:%.+]]: i32, [[ARG11:%.+]]: i32) -> (i32, i32)
//  CHECK-NEXT: [[ADD:%.+]] = arith.addi [[ARG10]], [[ARG10]] : i32
//  CHECK-NEXT: [[MUL:%.+]] = arith.muli [[ADD]], [[ARG11]] : i32
//  CHECK-NEXT: return [[ADD]], [[MUL]] : i32, i32

// -----

// Outline multiple ranges in the same function.
// CHECK-LABEL: func.func @multiple_ranges_in_same_func
//  CHECK-SAME: ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> (i32, i32)
func.func @multiple_ranges_in_same_func(%arg0: i32, %arg1: i32) -> (i32, i32) {
//  CHECK-NEXT: [[ADD:%.+]] = call @multiple_ranges_in_same_func_outline_0_0([[ARG0]]) : (i32) -> i32
  %add = arith.addi %arg0, %arg0 {outline_range_first, outline_range_last} : i32
//  CHECK-NEXT: [[MUL:%.+]] = call @multiple_ranges_in_same_func_outline_0_1([[ADD]], [[ARG1]]) : (i32, i32) -> i32
  %mul = arith.muli %add, %arg1 {outline_range_first, outline_range_last} : i32
//  CHECK-NEXT: return [[ADD]], [[MUL]] : i32, i32
  return %add, %mul : i32, i32
}
// CHECK-LABEL: func.func @multiple_ranges_in_same_func_outline_0_0
//  CHECK-SAME: ([[ARG10:%.+]]: i32) -> i32
//  CHECK-NEXT: [[ADD1:%.+]] = arith.addi [[ARG10]], [[ARG10]] : i32
//  CHECK-NEXT: return [[ADD1]] : i32
// CHECK-LABEL: func.func @multiple_ranges_in_same_func_outline_0_1
//  CHECK-SAME: ([[ARG20:%.+]]: i32, [[ARG21:%.+]]: i32) -> i32
//  CHECK-NEXT: [[MUL2:%.+]] = arith.muli [[ARG20]], [[ARG21]] : i32
//  CHECK-NEXT: return [[MUL2]] : i32

// -----

// Outline multiple ranges in different blocks.
// CHECK-LABEL: func.func @multiple_ranges_in_different_blocks
//  CHECK-SAME: ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32
func.func @multiple_ranges_in_different_blocks(%arg0: i32, %arg1: i32) -> i32 {
//  CHECK-NEXT: [[ADD:%.+]] = call @multiple_ranges_in_different_blocks_outline_0_0([[ARG0]]) : (i32) -> i32
  %add = arith.addi %arg0, %arg0 {outline_range_first, outline_range_last} : i32
//  CHECK-NEXT: cf.br ^bb1([[ARG1]] : i32)
  cf.br ^bb1(%arg1 : i32)
//  CHECK-NEXT: ^bb1
//  CHECK-SAME: ([[ARG2:%.+]]: i32)
^bb1 (%arg2: i32):
//  CHECK-NEXT: [[MUL:%.+]] = call @multiple_ranges_in_different_blocks_outline_1_0([[ADD]], [[ARG2]]) : (i32, i32) -> i32
  %mul = arith.muli %add, %arg2 {outline_range_first, outline_range_last} : i32
//  CHECK-NEXT: return [[MUL]] : i32
  return %mul : i32
}
// CHECK-LABEL: func.func @multiple_ranges_in_different_blocks_outline_0_0
//  CHECK-SAME: ([[ARG10:%.+]]: i32) -> i32
//  CHECK-NEXT: [[ADD1:%.+]] = arith.addi [[ARG10]], [[ARG10]] : i32
//  CHECK-NEXT: return [[ADD1]] : i32
// CHECK-LABEL: func.func @multiple_ranges_in_different_blocks_outline_1_0
//  CHECK-SAME: ([[ARG20:%.+]]: i32, [[ARG21:%.+]]: i32) -> i32
//  CHECK-NEXT: [[MUL2:%.+]] = arith.muli [[ARG20]], [[ARG21]] : i32
//  CHECK-NEXT: return [[MUL2]] : i32
