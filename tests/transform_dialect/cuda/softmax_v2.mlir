// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-flow-fuse-multi-use \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-transform-dialect-library=%p/softmax_v2_codegen_spec.mlir \
// RUN:     --iree-codegen-use-transform-dialect-strategy=codegen | \
// RUN: FileCheck %s --check-prefix=CHECK-SHUFFLE

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false \
/// Constant JIT'ing must be disabled because the transform-dialect debug
/// flags leak to the JIT session, which doesn't know what to do with them.
// RUN:     --iree-flow-fuse-multi-use \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-transform-dialect-library=%p/softmax_v2_codegen_spec.mlir \
// RUN:     --iree-codegen-use-transform-dialect-strategy=codegen | \
// RUN: iree-run-module --module=- --function=softmax --device=cuda | \
// RUN: FileCheck %s

!tmp_tensor_t = tensor<16x128xf32>
!in_tensor_t = tensor<16x128x128xf32>
!out_tensor_t = tensor<16x128x128xf32>

// Compilation checks that shuffles are produced.
// CHECK-SHUFFLE: gpu.shuffle  xor

// Execution only checks that @softmax runs.
//      CHECK: EXEC @softmax
//      CHECK: 16x128x128xf32=[
// CHECK-SAME:                [0.0078125 0.0078125 0.0078125 0.0078125

func.func @softmax() -> !out_tensor_t {
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32
  %cst_min = arith.constant -3.40282347E+38 : f32
  %input = arith.constant dense<5.000000e+00> : !out_tensor_t
  util.optimization_barrier %input : !in_tensor_t

  %softmax = flow.dispatch.region[] -> (!out_tensor_t) {
    %input_max_empty = tensor.empty() : !tmp_tensor_t
    %input_max_filled = linalg.fill ins(%cst_min : f32)
      outs(%input_max_empty : !tmp_tensor_t) -> !tmp_tensor_t
    %input_max = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%input : !in_tensor_t)
    outs(%input_max_filled : !tmp_tensor_t) {
      ^bb0(%arg0: f32, %arg1: f32):
        %max = arith.maximumf %arg0, %arg1 : f32
        linalg.yield %max : f32
      } -> !tmp_tensor_t

    // This has been fused manually to avoid the fusion on tensors pass and reduce noise atm.
    %exps_empty = tensor.empty() : !out_tensor_t
    %exps_sum_empty = tensor.empty() : !tmp_tensor_t
    %exps_sum_filled = linalg.fill ins(%cst_0 : f32)
      outs(%exps_sum_empty : !tmp_tensor_t) -> !tmp_tensor_t
    %exps, %exps_sum = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%input, %input_max : !in_tensor_t, !tmp_tensor_t)
      outs(%exps_empty, %exps_sum_filled : !out_tensor_t, !tmp_tensor_t) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
          %sub = arith.subf %arg0, %arg1 : f32
          %exp = math.exp %sub : f32
          %add = arith.addf %exp, %arg3 : f32
          linalg.yield %exp, %add : f32, f32
        } -> (!out_tensor_t, !tmp_tensor_t)

    %res_empty = tensor.empty() : !out_tensor_t
    %res = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                        iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%exps, %exps_sum : !out_tensor_t, !tmp_tensor_t)
      outs(%res_empty : !out_tensor_t) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          // %10 = arith.divf %cst_1, %arg1 : f32
          // %11 = arith.mulf %arg0, %10 : f32
          %div = arith.divf %arg0, %arg1 : f32
          linalg.yield %div : f32
        } -> !out_tensor_t
      flow.return %res : !out_tensor_t
  }
  return %softmax: !out_tensor_t
}
