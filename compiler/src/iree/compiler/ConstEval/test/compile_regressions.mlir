// RUN: iree-opt --split-input-file --verify-diagnostics --iree-consteval-jit-debug --iree-consteval-jit-globals  %s | FileCheck %s

// Test case reduced by running the pass --iree-util-hoist-into-globals on the
// following (and then chang the check to a return):
// func.func @i1_inline_constant() {
//   %control = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
//   %a = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
//   %b = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
//   %init = tensor.empty() : tensor<4xi32>
//   %c = linalg.generic {
//       indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
//                        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
//       iterator_types = ["parallel"]}
//       ins(%control, %a, %b : tensor<4xi1>, tensor<4xi32>, tensor<4xi32>)
//       outs(%init : tensor<4xi32>) {
//     ^bb0(%b1 : i1, %b2 : i32, %b3 : i32, %b4 : i32):
//       %0 = arith.select %b1, %b2, %b3 : i32
//       linalg.yield %0 : i32
//     } -> tensor<4xi32>
//   check.expect_eq_const(%c, dense<[1, 6, 3, 8]> : tensor<4xi32>) : tensor<4xi32>
//   return
// }

// CHECK-LABEL: module @hoisted_tensor_i1_input
// Verify the original check based on constant folding.
// CHECK: = dense<[1, 6, 3, 8]>
#map = affine_map<(d0) -> (d0)>
module @hoisted_tensor_i1_input {
  util.global private @hoisted : tensor<4xi32>
  func.func @i1_inline_constant() -> tensor<4xi32> {
    %hoisted = util.global.load @hoisted : tensor<4xi32>
    return %hoisted : tensor<4xi32>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %cst_0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %cst_1 = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
    %0 = tensor.empty() : tensor<4xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%cst, %cst_0, %cst_1 : tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) outs(%0 : tensor<4xi32>) {
    ^bb0(%in: i1, %in_2: i32, %in_3: i32, %out: i32):
      %2 = arith.select %in, %in_2, %in_3 : i32
      linalg.yield %2 : i32
    } -> tensor<4xi32>
    util.global.store %1, @hoisted : tensor<4xi32>
    util.return
  }
}

// -----
