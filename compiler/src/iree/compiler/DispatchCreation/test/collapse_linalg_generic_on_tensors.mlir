// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}, iree-dispatch-creation-clone-producers-into-dispatch-regions, iree-dispatch-creation-collapse-dimensions, cse))" %s | FileCheck %s
!type = tensor<2x4x8x16x32x64xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type

util.func public @collapse1() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse All (d0, d1, d2, d3, d4, d5)
  %6 = linalg.generic { indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
         }
    ins(%input : !type) outs(%output : !type) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> !type
  util.return %6: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: util.func public @collapse1
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2x4x8x16x32x64xf32> into tensor<2097152xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<2097152xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<2097152xf32>) outs(%[[OUT]] : tensor<2097152xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2, 3, 4, 5]] output_shape [2, 4, 8, 16, 32, 64] : tensor<2097152xf32> into tensor<2x4x8x16x32x64xf32>

// -----

!type = tensor<2x4x8x32x32x64x128xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type

util.func public @collapse2() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse (d0, d1) and (d5, d6)
  %6 = linalg.generic { indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d3, d5, d6)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>],
            iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]
         }
    ins(%input : !type) outs(%output : !type) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> !type
  util.return %6: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: util.func public @collapse2
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<2x4x8x32x32x64x128xf32> into tensor<8x8x32x32x8192xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x32x32x8192xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x32x32x8192xf32>) outs(%[[OUT]] : tensor<8x8x32x32x8192xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] output_shape [2, 4, 8, 32, 32, 64, 128] : tensor<8x8x32x32x8192xf32> into tensor<2x4x8x32x32x64x128xf32>

// -----
!type = tensor<2x4x8x16x32x64x128x256xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type

util.func public @collapse3() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse (d0, d1) and (d3, d4, d5, d6, d7)
  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
          iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "parallel", "parallel"]
        }
  ins(%input : !type) outs(%output : !type) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> !type
  util.return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: util.func public @collapse3
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<2x4x8x16x32x64x128x256xf32> into tensor<8x8x1073741824xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x1073741824xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "reduction", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x1073741824xf32>) outs(%[[OUT]] : tensor<8x8x1073741824xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] output_shape [2, 4, 8, 16, 32, 64, 128, 256] : tensor<8x8x1073741824xf32> into tensor<2x4x8x16x32x64x128x256xf32>

// -----

!type = tensor<2x4x8x16x64x64x128x256xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type
util.func public @collapse4() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse (d0, d1) and (d6, d7)
  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d5, d4, d6, d7)>],
          iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "parallel", "parallel"]
        }
  ins(%input : !type) outs(%output : !type) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> !type
  util.return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
// CHECK-LABEL: util.func public @collapse4
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x8x16x64x64x128x256xf32> into tensor<8x8x16x64x64x32768xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x16x64x64x32768xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x16x64x64x32768xf32>) outs(%[[OUT]] : tensor<8x8x16x64x64x32768xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] output_shape [2, 4, 8, 16, 64, 64, 128, 256] : tensor<8x8x16x64x64x32768xf32> into tensor<2x4x8x16x64x64x128x256xf32>

// -----

!type = tensor<2x4x32x32x32x64x128x256xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type
util.func public @collapse5() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %input2 = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %input3 = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse (d0, d1) and (d6, d7)
  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d4, d3, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d4, d3, d2, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "parallel", "parallel"]
        }
  ins(%input, %input2, %input3  : !type, !type, !type)
  outs(%output : !type) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    linalg.yield %arg1 : f32
  } -> !type
  util.return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP1:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d2, d4, d5)>
//       CHECK: #[[$MAP2:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2, d1, d4, d5)>
// CHECK-LABEL: util.func public @collapse5
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[IN1:.+]] = tensor.collapse_shape %[[INPUT1:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[IN2:.+]] = tensor.collapse_shape %[[INPUT2:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x32x32x32x64x32768xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]], %[[IN1]], %[[IN2]] : tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>) outs(%[[OUT]] : tensor<8x32x32x32x64x32768xf32>)
//       CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] output_shape [2, 4, 32, 32, 32, 64, 128, 256] : tensor<8x32x32x32x64x32768xf32> into tensor<2x4x32x32x32x64x128x256xf32>

// -----

!type = tensor<32x2x4x8x16x16x64x128xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type
util.func public @collapse6() -> !type {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type

  // Can collapse (d2, d3) and (d6, d7)
  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d5, d4, d6, d7)>],
          iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
        }
  ins(%input : !type) outs(%output : !type) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> !type
  util.return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
// CHECK-LABEL: util.func public @collapse6
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x4x8x16x16x64x128xf32> into tensor<32x2x32x16x16x8192xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<32x2x32x16x16x8192xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<32x2x32x16x16x8192xf32>) outs(%[[OUT]] : tensor<32x2x32x16x16x8192xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] output_shape [32, 2, 4, 8, 16, 16, 64, 128] : tensor<32x2x32x16x16x8192xf32> into tensor<32x2x4x8x16x16x64x128xf32>

// -----

!type_out = tensor<2x4x8x16xf32>
!type_in = tensor<2x4x8xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type_in
util.func public @collapse7() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type_in>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type_in> -> !type_in
  %output = tensor.empty() : !type_out

  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
  ins(%input : !type_in) outs(%output : !type_out) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> !type_out
  util.return %result: !type_out
}

//       CHECK: #[[$MAP:.+]] =  affine_map<(d0, d1) -> (d1)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: util.func public @collapse7
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<64x16xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<64xf32>) outs(%[[OUT]] : tensor<64x16xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2], [3]] output_shape [2, 4, 8, 16] : tensor<64x16xf32> into tensor<2x4x8x16xf32>

// -----

!type_in = tensor<16x4x32x2xf32>
!type_out = tensor<8x16x4x32x8x2xf32>
util.func public @collapse8(%input : !type_in) -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %output = tensor.empty() : !type_out
  // Can collapse (d3, d0, d1)
  %6 = linalg.generic { indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, d1, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d0, d1, d4, d5)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
         }
    ins(%input : !type_in) outs(%output : !type_out) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> !type_out
  util.return %6: !type_out
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @collapse8
//  CHECK-SAME:     (%[[IN:.+]]: tensor<16x4x32x2xf32>)
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1, 2], [3]{{\]}}
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x2048x8x2xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[COLLAPSE]] : tensor<2048x2xf32>) outs(%[[OUT]] : tensor<8x2048x8x2xf32
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0], [1, 2, 3], [4], [5]] output_shape [8, 16, 4, 32, 8, 2] : tensor<8x2048x8x2xf32> into tensor<8x16x4x32x8x2xf32>

// -----

!type_in = tensor<16x4xf32>
!type_out = tensor<16x32x4xf32>
util.func public @dont_collapse() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input = tensor.empty() : !type_in
  %output = tensor.empty() : !type_out
  %6 = linalg.generic { indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]
         }
    ins(%input : !type_in) outs(%output : !type_out) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> !type_out
  util.return %6: !type_out
}
// CHECK-LABEL: util.func public @dont_collapse
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP:.+]], #[[$MAP2:.+]]], iterator_types = ["parallel", "parallel", "parallel"]}

// -----

!type_in = tensor<2x4x8x16x32x64x128x256xf32>
!type_out = tensor<2x4x16x64x32x128x256xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type_in

util.func public @collapse9() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type_in>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type_in> -> !type_in
  %output = tensor.empty() : !type_out

  // Can collapse (d0, d1) and (d6, d7)
  %result = linalg.generic { indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
          affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d3, d5, d4, d6, d7)>],
          iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "parallel", "parallel"]
        }
  ins(%input : !type_in) outs(%output : !type_out) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> !type_out
  util.return %result: !type_out
}


//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d3, d5)>
// CHECK-LABEL: util.func public @collapse9
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}


// -----

!type_in = tensor<10x10x30xf32>
!type_out = tensor<20x10x10x30x20xf32>

util.func public @collapse10(%input : !type_in) -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %output = tensor.empty() : !type_out

  // Can collapse (d1, d3, d0)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d0)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d3, d0, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
  ins(%input : !type_in) outs(%output : !type_out) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  }  -> !type_out

  util.return %result: !type_out
}

// CHECK-LABEL: util.func public @collapse10
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel"]}

// -----

!type_in =  tensor<10x20xf32>
!type_out =  tensor<10x20xf32>

util.func public @collapse11(%input : !type_in) -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %output = tensor.empty() : !type_out

  // Can collapse (d1, d0)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"] }
  ins(%input : !type_in) outs(%output : !type_out) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  }  -> !type_out

  util.return %result: !type_out
}


//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: util.func public @collapse11
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

// -----

!type = tensor<16x32xi32>
util.func public @dont_collapse_dueto_index(%height : index, %width : index) -> !type {
  %init_source = tensor.empty() : !type
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init_source : !type) {
    ^bb0(%b0 : i32):
      %outer = linalg.index 0 : index
      %inner = linalg.index 1 : index
      %strided = arith.muli %outer, %width : index
      %linearized = arith.addi %inner, %strided : index
      %linearized_i32 = arith.index_cast %linearized : index to i32
      linalg.yield %linearized_i32 : i32
  } -> !type
  util.return %source : !type
}

// CHECK-LABEL: util.func public @dont_collapse
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP:.+]]], iterator_types = ["parallel"]}

// -----

!type = tensor<2x4x8x16x32x64xf32>
util.global private @"__transpose_10_input" {inlining_policy = #util.inline.never} = dense<1.0> : !type

util.func public @collapse12() -> (!type,!type,!type,!type) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input_ptr = util.global.address @"__transpose_10_input" : !util.ptr<!type>
  %input = util.global.load.indirect %input_ptr : !util.ptr<!type> -> !type
  %output = tensor.empty() : !type
  %output1 = tensor.empty() : !type
  %output2 = tensor.empty() : !type
  %output3 = tensor.empty() : !type

  %6, %7, %8, %9 = linalg.generic { indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>,
            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
         }
    ins(%input : !type) outs(%output, %output1, %output2, %output3 : !type, !type, !type, !type) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
      %0 = arith.addf %arg1, %arg2 : f32
      %1 = arith.addf %0, %arg3 : f32
      %2 = arith.addf %1, %arg4 : f32
      %3 = arith.addf %2, %arg5 : f32
      linalg.yield %0,%1,%2,%3 : f32, f32, f32, f32
    } -> (!type,!type,!type,!type)
  util.return %6, %7, %8, %9  : !type,!type,!type,!type
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: util.func public @collapse12
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

// -----

util.func public @multi_reduce_dim(%arg0: tensor<2x32x10x4096xf32>) -> tensor<2x32x1x1xf32> {
  %cst = arith.constant -0.000000e+00 : f32
  %1 = tensor.empty() : tensor<2x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0 : tensor<2x32x10x4096xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<2x32xf32>
  %4 = tensor.expand_shape %3 [[0], [1, 2, 3]] output_shape [2, 32, 1, 1] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
  util.return %4 : tensor<2x32x1x1xf32>
}

// Check that we collapse dimensions.
// CHECK-LABEL: @multi_reduce_dim
//  CHECK-SAME: (%[[ARG0:.+]]: tensor<2x32x10x4096xf32>)
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}}
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<64xf32>
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[EMPTY]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[COLLAPSE]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:      flow.return %[[GENERIC]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[DISPATCH]] {{\[}}[0, 1]{{\]}}

// -----

// Collapsing is not supported when an input is broadcasted; we can't collapse
// the input from tensor<4xf32> to tensor<32xf32> for example.

util.func public @input_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %empty = tensor.empty() : tensor<f32>
  %reduce = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4xf32>) outs(%empty : tensor<f32>) {
  ^bb0(%arg2: f32, %arg3: f32, %out: f32):
    %div = arith.divf %arg2, %arg3 : f32
    %add = arith.addf %out, %div : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  util.return %reduce : tensor<f32>
}

//     CHECK: @input_broadcast
// CHECK-NOT: tensor.collapse_shape

// -----

util.func public @batchnorm_failure_repro(%arg0 : tensor<2x4xf32>, %arg1 : tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<4xf32>) outs(%0 : tensor<2x4xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
  } -> tensor<2x4xf32>
  util.return %1 : tensor<2x4xf32>
}
// CHECK-LABEL: util.func public @batchnorm_failure_repro
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func public @catch_invalid_collapse(%arg0 : tensor<10x20x30xf32>) -> tensor<10x30x40xf32> {
  %0 = tensor.empty() : tensor<10x30x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<10x20x30xf32>) outs(%0 : tensor<10x30x40xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
  } -> tensor<10x30x40xf32>
  util.return %1 : tensor<10x30x40xf32>
}
// CHECK-LABEL: util.func public @catch_invalid_collapse
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
