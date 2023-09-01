// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=cuda %s

//===----------------------------------------------------------------------===//
// Transpose ops.
// Naming convention: '_'.join(
//   [transpose,
//    {output-shape])
//
//===----------------------------------------------------------------------===//

util.global private @"__transpose_4096_4096_input" {noinline} = dense<1.0> : tensor<4096x4096xf32>

func.func @transpsoe_4096_4096() -> tensor<4096x4096xf32> {
  %input_ptr = util.global.address @"__transpose_4096_4096_input" : !util.ptr<tensor<4096x4096xf32>>
  %input = util.global.load.indirect %input_ptr : !util.ptr<tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
  %output = tensor.empty() : tensor<4096x4096xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4096x4096xf32>) outs(%output : tensor<4096x4096xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
        linalg.yield %arg1 : f32
    } -> tensor<4096x4096xf32>
 return %result : tensor<4096x4096xf32>
}

util.global private @"__transpose_10_2048_1024_input" {noinline} = dense<1.0> : tensor<10x2048x1024xf32>

func.func @transpsoe_10_1024_2048() -> tensor<10x1024x2048xf32> {
  %input_ptr = util.global.address @"__transpose_10_2048_1024_input" : !util.ptr<tensor<10x2048x1024xf32>>
  %input = util.global.load.indirect %input_ptr : !util.ptr<tensor<10x2048x1024xf32>> -> tensor<10x2048x1024xf32>
  %output = tensor.empty() : tensor<10x1024x2048xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input : tensor<10x2048x1024xf32>) outs(%output : tensor<10x1024x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
        linalg.yield %arg1 : f32
    } -> tensor<10x1024x2048xf32>
 return %result : tensor<10x1024x2048xf32>
}

util.global private @"__transpose_10_2048_1024_lhs" {noinline} = dense<1.0> : tensor<10x2048x1024xf32>
util.global private @"__transpose_10_2048_1024_rhs" {noinline} = dense<1.0> : tensor<10x2048x1024xf32>

func.func @transpsoe_10_1024_2048_fusion() -> tensor<10x1024x2048xf32> {
  %lhs_ptr = util.global.address @"__transpose_10_2048_1024_lhs" : !util.ptr<tensor<10x2048x1024xf32>>
  %lhs = util.global.load.indirect %lhs_ptr : !util.ptr<tensor<10x2048x1024xf32>> -> tensor<10x2048x1024xf32>
  %rhs_ptr = util.global.address @"__transpose_10_2048_1024_rhs" : !util.ptr<tensor<10x2048x1024xf32>>
  %rhs = util.global.load.indirect %rhs_ptr : !util.ptr<tensor<10x2048x1024xf32>> -> tensor<10x2048x1024xf32>
  %output = tensor.empty() : tensor<10x1024x2048xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%lhs, %rhs : tensor<10x2048x1024xf32>, tensor<10x2048x1024xf32>) outs(%output : tensor<10x1024x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
        %0 = arith.addf %arg1, %arg2 : f32
        linalg.yield %0 : f32
    } -> tensor<10x1024x2048xf32>
 return %result : tensor<10x1024x2048xf32>
}

util.global private @"__transpose_10_2064_1024_input" {noinline} = dense<1.0> : tensor<10x2064x1024xf32>

func.func @transpsoe_10_1024_2064_unaligned() -> tensor<10x1024x2064xf32> {
  %input_ptr = util.global.address @"__transpose_10_2064_1024_input" : !util.ptr<tensor<10x2064x1024xf32>>
  %input = util.global.load.indirect %input_ptr : !util.ptr<tensor<10x2064x1024xf32>> -> tensor<10x2064x1024xf32>
  %output = tensor.empty() : tensor<10x1024x2064xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input : tensor<10x2064x1024xf32>) outs(%output : tensor<10x1024x2064xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
        linalg.yield %arg1 : f32
    } -> tensor<10x1024x2064xf32>
 return %result : tensor<10x1024x2064xf32>
}
