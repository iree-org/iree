// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{aggressive-fusion=true}))" %s | FileCheck %s
!type = tensor<2x4x8x16x32x64xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse() -> !type {
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
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2x4x8x16x32x64xf32> into tensor<2097152xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2x4x8x16x32x64xf32> into tensor<2097152xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP:.+]]], iterator_types = ["parallel"]}
//      CHECK: ins(%[[IN]] : tensor<2097152xf32>) outs(%[[OUT]] : tensor<2097152xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2097152xf32> into tensor<2x4x8x16x32x64xf32>

// -----

!type = tensor<2x4x8x32x32x64x128xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse() -> !type {
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

//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<2x4x8x32x32x64x128xf32> into tensor<8x8x32x32x8192xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<2x4x8x32x32x64x128xf32> into tensor<8x8x32x32x8192xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP:.+]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x32x32x8192xf32>) outs(%[[OUT]] : tensor<8x8x32x32x8192xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<8x8x32x32x8192xf32> into tensor<2x4x8x32x32x64x128xf32>

// -----
!type = tensor<2x4x8x16x32x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type

func.func @collapse() -> !type {
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

//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<2x4x8x16x32x64x128x256xf32> into tensor<8x8x1073741824xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<2x4x8x16x32x64x128x256xf32> into tensor<8x8x1073741824xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP:.+]]], iterator_types = ["parallel", "reduction", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x1073741824xf32>) outs(%[[OUT]] : tensor<8x8x1073741824xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<8x8x1073741824xf32> into tensor<2x4x8x16x32x64x128x256xf32>

// -----

!type = tensor<2x4x8x16x64x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse() -> !type {
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

//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x8x16x64x64x128x256xf32> into tensor<8x8x16x64x64x32768xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x8x16x64x64x128x256xf32> into tensor<8x8x16x64x64x32768xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP:.+]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x16x64x64x32768xf32>) outs(%[[OUT]] : tensor<8x8x16x64x64x32768xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x8x16x64x64x32768xf32> into tensor<2x4x8x16x64x64x128x256xf32>

// -----

!type = tensor<2x4x32x32x32x64x128x256xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse() -> !type {
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

//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[IN1:.+]] = tensor.collapse_shape %[[INPUT1:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[IN2:.+]] = tensor.collapse_shape %[[INPUT2:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP1:.+]], #[[MAP2:.+]], #[[MAP:.+]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]}
//      CHECK: ins(%[[IN]], %[[IN1]], %[[IN2]] : tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>) outs(%[[OUT]] : tensor<8x32x32x32x64x32768xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x32x32x32x64x32768xf32> into tensor<2x4x32x32x32x64x128x256xf32>

// -----

!type = tensor<32x2x4x8x16x16x64x128xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type
func.func @collapse() -> !type {
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

//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x4x8x16x16x64x128xf32> into tensor<32x2x32x16x16x8192xf32>
//      CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x4x8x16x16x64x128xf32> into tensor<32x2x32x16x16x8192xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP2:.+]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<32x2x32x16x16x8192xf32>) outs(%[[OUT]] : tensor<32x2x32x16x16x8192xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x32x16x16x8192xf32> into tensor<32x2x4x8x16x16x64x128xf32>

// -----

!type_out = tensor<2x4x8x16xf32>
!type_in = tensor<2x4x8xf32>
util.global private @"__transpose_10_input" {noinline} = dense<1.0> : !type_in
func.func @collapse() -> !type_out {
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

// CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
// CHECK: %[[OUT:.+]] = tensor.collapse_shape %[[OUTPUT:.+]] {{\[}}[0, 1, 2], [3]] : tensor<2x4x8x16xf32> into tensor<64x16xf32>
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP2:.+]]], iterator_types = ["parallel", "parallel"]}
// CHECK: ins(%[[IN]] : tensor<64xf32>) outs(%[[OUT]] : tensor<64x16xf32>)
// CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2], [3]] : tensor<64x16xf32> into tensor<2x4x8x16xf32>
