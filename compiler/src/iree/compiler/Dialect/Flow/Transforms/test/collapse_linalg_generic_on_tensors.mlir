// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-multi-use=true}, iree-flow-clone-producers-into-dispatch-regions, iree-flow-collapse-dimensions, cse))" %s | FileCheck %s
!type = tensor<2x4x8x16x32x64xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse1() -> !type {
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
  return %6: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @collapse1
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2x4x8x16x32x64xf32> into tensor<2097152xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<2097152xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<2097152xf32>) outs(%[[OUT]] : tensor<2097152xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2097152xf32> into tensor<2x4x8x16x32x64xf32>

// -----

!type = tensor<2x4x8x32x32x64x128xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse2() -> !type {
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
  return %6: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @collapse2
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<2x4x8x32x32x64x128xf32> into tensor<8x8x32x32x8192xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x32x32x8192xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x32x32x8192xf32>) outs(%[[OUT]] : tensor<8x8x32x32x8192xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<8x8x32x32x8192xf32> into tensor<2x4x8x32x32x64x128xf32>

// -----
!type = tensor<2x4x8x16x32x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse3() -> !type {
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
  return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @collapse3
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<2x4x8x16x32x64x128x256xf32> into tensor<8x8x1073741824xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x1073741824xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "reduction", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x1073741824xf32>) outs(%[[OUT]] : tensor<8x8x1073741824xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<8x8x1073741824xf32> into tensor<2x4x8x16x32x64x128x256xf32>

// -----

!type = tensor<2x4x8x16x64x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse4() -> !type {
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
  return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
// CHECK-LABEL: func.func @collapse4
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x8x16x64x64x128x256xf32> into tensor<8x8x16x64x64x32768xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x8x16x64x64x32768xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<8x8x16x64x64x32768xf32>) outs(%[[OUT]] : tensor<8x8x16x64x64x32768xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x8x16x64x64x32768xf32> into tensor<2x4x8x16x64x64x128x256xf32>

// -----

!type = tensor<2x4x32x32x32x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse5() -> !type {
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
  return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP1:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d2, d4, d5)>
//       CHECK: #[[$MAP2:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2, d1, d4, d5)>
// CHECK-LABEL: func.func @collapse5
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[IN1:.+]] = tensor.collapse_shape %[[INPUT1:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[IN2:.+]] = tensor.collapse_shape %[[INPUT2:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x32x32x32x64x32768xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]], %[[IN1]], %[[IN2]] : tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>) outs(%[[OUT]] : tensor<8x32x32x32x64x32768xf32>)
//       CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x32x32x32x64x32768xf32> into tensor<2x4x32x32x32x64x128x256xf32>

// -----

!type = tensor<32x2x4x8x16x16x64x128xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse6() -> !type {
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
  return %result: !type

}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
// CHECK-LABEL: func.func @collapse6
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x4x8x16x16x64x128xf32> into tensor<32x2x32x16x16x8192xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<32x2x32x16x16x8192xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<32x2x32x16x16x8192xf32>) outs(%[[OUT]] : tensor<32x2x32x16x16x8192xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x32x16x16x8192xf32> into tensor<32x2x4x8x16x16x64x128xf32>

// -----

!type_out = tensor<2x4x8x16xf32>
!type_in = tensor<2x4x8xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type_in
func.func @collapse7() -> !type_out {
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
  return %result: !type_out
}

//       CHECK: #[[$MAP:.+]] =  affine_map<(d0, d1) -> (d1)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @collapse7
//       CHECK:   %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<64x16xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[IN]] : tensor<64xf32>) outs(%[[OUT]] : tensor<64x16xf32>)
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2], [3]] : tensor<64x16xf32> into tensor<2x4x8x16xf32>

// -----

!type_in = tensor<16x4x32x2xf32>
!type_out = tensor<8x16x4x32x8x2xf32>
func.func @collapse8(%input : !type_in) -> !type_out {
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
  return %6: !type_out
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @collapse8
//  CHECK-SAME:     (%[[IN:.+]]: tensor<16x4x32x2xf32>)
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1, 2], [3]{{\]}}
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<8x2048x8x2xf32>
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[COLLAPSE]] : tensor<2048x2xf32>) outs(%[[OUT]] : tensor<8x2048x8x2xf32
//       CHECK:   tensor.expand_shape %[[RES]] {{\[}}[0], [1, 2, 3], [4], [5]] : tensor<8x2048x8x2xf32> into tensor<8x16x4x32x8x2xf32>

// -----

!type_in = tensor<16x4xf32>
!type_out = tensor<16x32x4xf32>
func.func @dont_collapse() -> !type_out {
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
  return %6: !type_out
}
// CHECK-LABEL: func.func @dont_collapse
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP:.+]], #[[$MAP2:.+]]], iterator_types = ["parallel", "parallel", "parallel"]}

// -----

!type_in = tensor<2x4x8x16x32x64x128x256xf32>
!type_out = tensor<2x4x16x64x32x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type_in

func.func @collapse9() -> !type_out {
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
  return %result: !type_out
}


//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d3, d5)>
// CHECK-LABEL: func.func @collapse9
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}


// -----

!type_in = tensor<10x10x30xf32>
!type_out = tensor<20x10x10x30x20xf32>

func.func @collapse10(%input : !type_in) -> !type_out {
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

  return %result: !type_out
}

// CHECK-LABEL: func.func @collapse10
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel"]}

// -----

!type_in =  tensor<10x20xf32>
!type_out =  tensor<10x20xf32>

func.func @collapse11(%input : !type_in) -> !type_out {
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

  return %result: !type_out
}


//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @collapse11
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

// -----

!type = tensor<16x32xi32>
func.func @dont_collapse_dueto_index(%height : index, %width : index) -> !type {
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
  return %source : !type
}

// CHECK-LABEL: func.func @dont_collapse
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP:.+]]], iterator_types = ["parallel", "parallel"]}

// -----

!type = tensor<2x4x8x16x32x64xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse12() -> (!type,!type,!type,!type) {
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
  return %6, %7, %8, %9  : !type,!type,!type,!type
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @collapse12
//       CHECK:   %[[RES:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

// -----

func.func @multi_reduce_dim(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %cst = arith.constant -0.000000e+00 : f32
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x32x10x4096xf32>
  %1 = tensor.empty() : tensor<2x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%0 : tensor<2x32x10x4096xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<2x32xf32>
  %4 = tensor.expand_shape %3 [[0], [1, 2, 3]] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
  %5 = hal.tensor.export %4 : tensor<2x32x1x1xf32> -> !hal.buffer_view
  return %5 : !hal.buffer_view
}

// Check that we collapse dimensions.
// CHECK-LABEL: @multi_reduce_dim(
//   CHECK-DAG:   %[[ARG0:.+]] = hal.tensor.import
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

func.func @input_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %empty = tensor.empty() : tensor<f32>
  %reduce = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4xf32>) outs(%empty : tensor<f32>) {
  ^bb0(%arg2: f32, %arg3: f32, %out: f32):
    %div = arith.divf %arg2, %arg3 : f32
    %add = arith.addf %out, %div : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  return %reduce : tensor<f32>
}

//     CHECK: @input_broadcast
// CHECK-NOT: tensor.collapse_shape

// -----

// Do nothing if the dispatch is not a single elementwise op (with tensor.empty/linalg.fill producers)

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
    %0 = flow.dispatch.region -> (tensor<1x1x4096xf32>) {
      %cst_1 = arith.constant 0.000000e+00 : f32
      %1 = tensor.empty() : tensor<1x1x4096xf32>
      %2 = tensor.empty() : tensor<4096x32x128xf32>
      %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst, %cst_0 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%2 : tensor<4096x32x128xf32>) {
      ^bb0(%in: i8, %in_2: f32, %in_3: f32, %out: f32):
        %6 = arith.extui %in : i8 to i32
        %7 = arith.uitofp %6 : i32 to f32
        %8 = arith.subf %7, %in_3 : f32
        %9 = arith.mulf %8, %in_2 : f32
        linalg.yield %9 : f32
      } -> tensor<4096x32x128xf32>
      %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %4 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%3 : tensor<1x1x4096xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %6 = arith.mulf %in, %in_2 : f32
        %7 = arith.addf %6, %out : f32
        linalg.yield %7 : f32
      } -> tensor<1x1x4096xf32>
      flow.return %5 : tensor<1x1x4096xf32>
    }
    return %0 : tensor<1x1x4096xf32>
  }
}

// CHECK-LABEL:  func.func @quantized_matmul
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      linalg.generic
//  CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:      linalg.generic
//  CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//       CHECK:      flow.return
//       CHECK:    return %[[DISPATCH]]

// -----

module {
  func.func @batchnorm_failure_repro(%arg0 : tensor<2x4xf32>, %arg1 : tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<4xf32>) outs(%0 : tensor<2x4xf32>) {
      ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
        %2 = arith.addf %b0, %b1 : f32
        linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
// CHECK-LABEL: func @batchnorm_failure_repro
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel"]
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   return %[[DISPATCH]]

// -----

module {
  func.func @catch_invalid_collapse(%arg0 : tensor<10x20x30xf32>) -> tensor<10x30x40xf32> {
    %0 = tensor.empty() : tensor<10x30x40xf32>
    %1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<10x20x30xf32>) outs(%0 : tensor<10x30x40xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        linalg.yield %b0 : f32
    } -> tensor<10x30x40xf32>
    return %1 : tensor<10x30x40xf32>
  }
}
// CHECK-LABEL: func @catch_invalid_collapse
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
