// RUN: iree-opt -split-input-file -iree-consteval-jit-globals %s | IreeFileCheck %s

// TODO: Full type matrix for tests.

// CHECK-LABEL: @linalg_tensor_jit
// CHECK: util.global private @{{.*}} = dense<4.000000e+04> : tensor<5x6xf32>
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @linalg_tensor_jit {
  util.global private @hoisted : tensor<5x6xf32>
  func @main() -> tensor<5x6xf32> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf32>
    return %hoisted : tensor<5x6xf32>
  }
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<2.0e+02> : tensor<f32>
    %0 = linalg.init_tensor [5, 6] : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
      linalg.yield %arg0 : f32
    } -> tensor<5x6xf32>
    %2 = linalg.init_tensor [5, 6] : tensor<5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%2 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
      %4 = arith.mulf %arg0, %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x6xf32>
    util.global.store %3, @hoisted : tensor<5x6xf32>
    util.initializer.return
  }
}

// TODO: Crashes compiler.
// COM-CHECK-LABEL: @eval_f16_tensor
// module @eval_f16_tensor {
//   util.global private @hoisted : tensor<5x6xf16>
//   func @main() -> tensor<5x6xf16> {
//     %hoisted = util.global.load @hoisted : tensor<5x6xf16>
//     return %hoisted : tensor<5x6xf16>
//   }
//   util.initializer {
//     %cst = arith.constant dense<2.0e+2> : tensor<5x6xf16>
//     util.global.store %cst, @hoisted : tensor<5x6xf16>
//     util.initializer.return
//   }
// }

// TODO: Error on 'hal.command_buffer.fill_buffer'
// COM-CHECK-LABEL: @eval_f16_tensor
// module @eval_bf16_tensor {
//   util.global private @hoisted : tensor<5x6xbf16>
//   func @main() -> tensor<5x6xbf16> {
//     %hoisted = util.global.load @hoisted : tensor<5x6xbf16>
//     return %hoisted : tensor<5x6xbf16>
//   }
//   util.initializer {
//     %cst = arith.constant dense<2.0e+2> : tensor<5x6xbf16>
//     util.global.store %cst, @hoisted : tensor<5x6xbf16>
//     util.initializer.return
//   }
// }

// TODO: Error on 'hal.command_buffer.fill_buffer'
// COM-CHECK-LABEL: @eval_i4_tensor
// module @eval_i4_tensor {
//   util.global private @hoisted : tensor<5x6xi4>
//   func @main() -> tensor<5x6xi4> {
//     %hoisted = util.global.load @hoisted : tensor<5x6xi4>
//     return %hoisted : tensor<5x6xi4>
//   }
//   util.initializer {
//     %cst = arith.constant dense<3> : tensor<5x6xi4>
//     util.global.store %cst, @hoisted : tensor<5x6xi4>
//     util.initializer.return
//   }
// }

// TODO: Error: mapped memory region was not valid for constructing tensor of type 'tensor<5x6xi1>' (length=30)
// COM-CHECK-LABEL: @eval_i1_tensor
// module @eval_i1_tensor {
//   util.global private @hoisted : tensor<5x6xi1>
//   func @main() -> tensor<5x6xi1> {
//     %hoisted = util.global.load @hoisted : tensor<5x6xi1>
//     return %hoisted : tensor<5x6xi1>
//   }
//   util.initializer {
//     %cst = arith.constant dense<1> : tensor<5x6xi1>
//     util.global.store %cst, @hoisted : tensor<5x6xi1>
//     util.initializer.return
//   }
// }
