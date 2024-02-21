!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  return %6 : !out_tensor_t
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="8x64xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xf32=256 256 256 256 256 256 256 256
