// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/chained_reduction_match_spec.mlir' --split-input-file --verify-diagnostics

func.func @forward_dispatch_43_generic_16x4096x4096(%arg0: tensor<16x4096x4096xf32>) -> tensor<16x4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -6.550400e+04 : f32
  %7 = tensor.empty() : tensor<16x4096xf32>
  %8 = tensor.empty() : tensor<16x4096x4096xf32>
  // expected-remark @below {{fill_1}}
  %fill_1 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
  // expected-remark @below {{fill_2}}
  %fill_2 = linalg.fill ins(%cst : f32) outs(%7 : tensor<16x4096xf32>) -> tensor<16x4096xf32>

  // expected-remark @below {{reduction_1}}
  %reduction_1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0 : tensor<16x4096x4096xf32>)
    outs(%fill_1 : tensor<16x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %14 = arith.maxf %in, %out : f32
    linalg.yield %14 : f32
  } -> tensor<16x4096xf32>

  // expected-remark @below {{middle}}
  %trailing_1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0, %reduction_1 : tensor<16x4096x4096xf32>, tensor<16x4096xf32>)
    outs(%8 : tensor<16x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %14 = arith.subf %in, %in_1 : f32
    %15 = math.exp %14 : f32
    linalg.yield %15 : f32
  } -> tensor<16x4096x4096xf32>

  // expected-remark @below {{reduction_2}}
  %reduction_2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%trailing_1 : tensor<16x4096x4096xf32>)
    outs(%fill_2 : tensor<16x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %16 = arith.addf %in, %out : f32
    linalg.yield %16 : f32
  } -> tensor<16x4096xf32>

  // expected-remark @below {{trailing_2}}
  %trailing_2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%trailing_1, %reduction_2 : tensor<16x4096x4096xf32>, tensor<16x4096xf32>)
    outs(%8 : tensor<16x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %14 = arith.divf %in, %in_1 : f32
    linalg.yield %14 : f32
  } -> tensor<16x4096x4096xf32>
  return %trailing_2 : tensor<16x4096x4096xf32>
}
