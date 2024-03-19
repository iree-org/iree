!in_tensor_t = tensor<33x1024xf32>
!out_tensor_t = tensor<33xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
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

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-transform-dialect-library=%p/reduction_v2_codegen_spec.mlir@codegen | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="33x1024xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

// RUN: iree-compile %s --iree-hal-target-backends=cuda | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="33x1024xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

// only checking the first 6 of 33
//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 33xf32=1024 1024 1024 1024 1024 1024
