// iree-transform-compile   /usr/local/google/home/ntv/github/iree/tests/transform_dialect/cuda/warp_reduction_dispatch.mlir -b cuda        -- --iree-hal-benchmark-dispatch-repeat-count=5 |   /usr/local/cuda-11.4/bin/nvprof  --print-gpu-trace  iree-run-module --entry_function=warp_reduction_dispatch --device=cuda --function_input="512x10240xf32=1"
// Tested to run in 40us (i.e. 524GB/s on an RTX2080Ti on which https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest says 520GB/s)

!in_tensor_t = tensor<512x10240xf32>
!out_tensor_t = tensor<512xf32>

func.func @warp_reduction_dispatch(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -0.000000e+00 : f32
  
  %d0 = tensor.dim %arg, %c0 : !in_tensor_t
  %0 = tensor.empty() : !out_tensor_t
  // %0 = tensor.empty(%d0) : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->   !out_tensor_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_t
  return %2 : !out_tensor_t
}
