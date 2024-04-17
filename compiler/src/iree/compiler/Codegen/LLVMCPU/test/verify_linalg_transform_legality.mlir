// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-verify-linalg-transform-legality))" %s --verify-diagnostics -split-input-file

func.func @matmul_123x456xf32_times_456x789xf32_into_123x789xf32_dispatch_0() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<123x4x114xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x114x789xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x123x789xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [123, 4, 114], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<123x4x114xf32>> -> tensor<123x4x114xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 114, 789], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x114x789xf32>> -> tensor<4x114x789xf32>
  %5 = tensor.empty() : tensor<4x123x789xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x123x789xf32>) -> tensor<4x123x789xf32>
  // expected-error @+1 {{expected no Linalg transform markers}}
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%3, %4 : tensor<123x4x114xf32>, tensor<4x114x789xf32>)
    outs(%6 : tensor<4x123x789xf32>)
    attrs =  {__internal_linalg_transform__ = "DEADBEEF", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %8 = arith.mulf %arg0, %arg1 : f32
    %9 = arith.addf %arg2, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<4x123x789xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 123, 789], strides = [1, 1, 1] : tensor<4x123x789xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x123x789xf32>>
  return
}
