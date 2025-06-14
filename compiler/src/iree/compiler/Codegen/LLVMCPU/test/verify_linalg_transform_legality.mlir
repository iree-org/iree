// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-verify-linalg-transform-legality))" %s --verify-diagnostics

func.func @generic_with_marker(%arg0: tensor<123x4x114xf32>, %arg1: tensor<4x114x789xf32>) -> tensor<4x123x789xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<4x123x789xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x123x789xf32>) -> tensor<4x123x789xf32>
  // expected-error @+1 {{expected no Linalg transform markers}}
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<123x4x114xf32>, tensor<4x114x789xf32>)
    outs(%1 : tensor<4x123x789xf32>)
    attrs = {__internal_linalg_transform__ = "DEADBEEF"} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<4x123x789xf32>
  return %2 : tensor<4x123x789xf32>
}
