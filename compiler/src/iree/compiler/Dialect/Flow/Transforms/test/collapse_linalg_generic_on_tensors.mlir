// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-multi-use=true}, iree-flow-collapse-dimensions))" %s | FileCheck %s
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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
//      CHECK-LABEL: func.func @collapse1
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2x4x8x16x32x64xf32> into tensor<2097152xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<2097152xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}
//      CHECK: ins(%[[IN]] : tensor<2097152xf32>) outs(%[[OUT]] : tensor<2097152xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2, 3, 4, 5]] : tensor<2097152xf32> into tensor<2x4x8x16x32x64xf32>

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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
//      CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK-LABEL: func.func @collapse2
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<2x4x8x32x32x64x128xf32> into tensor<8x8x32x32x8192xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<8x8x32x32x8192xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x32x32x8192xf32>) outs(%[[OUT]] : tensor<8x8x32x32x8192xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5, 6]] : tensor<8x8x32x32x8192xf32> into tensor<2x4x8x32x32x64x128xf32>

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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK-LABEL: func.func @collapse3
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<2x4x8x16x32x64x128x256xf32> into tensor<8x8x1073741824xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<8x8x1073741824xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "reduction", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x1073741824xf32>) outs(%[[OUT]] : tensor<8x8x1073741824xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3, 4, 5, 6, 7]] : tensor<8x8x1073741824xf32> into tensor<2x4x8x16x32x64x128x256xf32>

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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//      CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
//      CHECK-LABEL: func.func @collapse4
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x8x16x64x64x128x256xf32> into tensor<8x8x16x64x64x32768xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<8x8x16x64x64x32768xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<8x8x16x64x64x32768xf32>) outs(%[[OUT]] : tensor<8x8x16x64x64x32768xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x8x16x64x64x32768xf32> into tensor<2x4x8x16x64x64x128x256xf32>

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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//      CHECK: #[[$MAP1:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d2, d4, d5)>
//      CHECK: #[[$MAP2:.+]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2, d1, d4, d5)>
//      CHECK-LABEL: func.func @collapse5
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[IN1:.+]] = tensor.collapse_shape %[[INPUT1:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[IN2:.+]] = tensor.collapse_shape %[[INPUT2:.+]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<2x4x32x32x32x64x128x256xf32> into tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<8x32x32x32x64x32768xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]}
//      CHECK: ins(%[[IN]], %[[IN1]], %[[IN2]] : tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>, tensor<8x32x32x32x64x32768xf32>) outs(%[[OUT]] : tensor<8x32x32x32x64x32768xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1], [2], [3], [4], [5], [6, 7]] : tensor<8x32x32x32x64x32768xf32> into tensor<2x4x32x32x32x64x128x256xf32>

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

//      CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//      CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
//      CHECK-LABEL: func.func @collapse6
//      CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x4x8x16x16x64x128xf32> into tensor<32x2x32x16x16x8192xf32>
//      CHECK: %[[OUT:.+]] = tensor.empty() : tensor<32x2x32x16x16x8192xf32>
//      CHECK: %[[RES:.+]] = flow.dispatch.region
//      CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}
//      CHECK: ins(%[[IN]] : tensor<32x2x32x16x16x8192xf32>) outs(%[[OUT]] : tensor<32x2x32x16x16x8192xf32>)
//      CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0], [1], [2, 3], [4], [5], [6, 7]] : tensor<32x2x32x16x16x8192xf32> into tensor<32x2x4x8x16x16x64x128xf32>

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

// CHECK: #[[$MAP:.+]] =  affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @collapse7
// CHECK: %[[IN:.+]] = tensor.collapse_shape %[[INPUT:.+]] {{\[}}[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<64x16xf32>
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]}
// CHECK: ins(%[[IN]] : tensor<64xf32>) outs(%[[OUT]] : tensor<64x16xf32>)
// CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2], [3]] : tensor<64x16xf32> into tensor<2x4x8x16xf32>

// -----

!type_in = tensor<16x4x32x2xf32>
!type_out = tensor<8x16x4x32x8x2xf32>
func.func @collapse8() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input = tensor.empty() : !type_in
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

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @collapse8
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<2048x2xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<8x2048x8x2xf32>
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK: ins(%[[IN]] : tensor<2048x2xf32>) outs(%[[OUT]] : tensor<8x2048x8x2xf32
// CHECK:  tensor.expand_shape %[[RES]] {{\[}}[0], [1, 2, 3], [4], [5]] : tensor<8x2048x8x2xf32> into tensor<8x16x4x32x8x2xf32>

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
// CHECK: linalg.generic {indexing_maps = [#[[$MAP:.+]], #[[$MAP2:.+]]], iterator_types = ["parallel", "parallel", "parallel"]}

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


// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d3, d5)>
// CHECK-LABEL: func.func @collapse9
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel"]}


// -----

!type_in = tensor<10x10x30xf32>
!type_out = tensor<20x10x10x30x20xf32>

func.func @collapse10() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input = tensor.empty() : !type_in
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

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: func.func @collapse10
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel"]}

// -----

!type_in =  tensor<10x20xf32>
!type_out =  tensor<10x20xf32>

func.func @collapse11() -> !type_out {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input = tensor.empty() : !type_in
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


// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @collapse11
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

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
// CHECK: linalg.generic {indexing_maps = [#[[$MAP:.+]]], iterator_types = ["parallel", "parallel"]}

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

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @collapse12
// CHECK: %[[RES:.+]] = flow.dispatch.region
// CHECK: linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]}

