
// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/softmax_partial_codegen_spec.mlir \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
// RUN: FileCheck %s --check-prefix=CHECK-SHUFFLE

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false \
/// Constant JIT'ing must be disabled because the transform-dialect debug
/// flags leak to the JIT session, which doesn't know what to do with them.
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/softmax_partial_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=softmax_partial --device=cuda | \
// RUN: FileCheck %s

!tmp_tensor_t = tensor<16x128xf32>
!out_tensor_t = tensor<16x128x128xf32>

// Compilation checks that shuffles are produced.
// CHECK-SHUFFLE: gpu.shuffle  xor

// Execution only checks that @softmax_partial runs.
//      CHECK: EXEC @softmax_partial
//      CHECK: 16x128x128xf32=[
// CHECK-SAME:                [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

func.func @softmax_partial() -> !out_tensor_t {
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant dense<1121212.000000e+00> : !out_tensor_t
  %cst_1 = arith.constant dense<5.000000e+00> : !out_tensor_t
  %0 = util.optimization_barrier %cst_1 : !out_tensor_t

  %1 = tensor.empty() : !tmp_tensor_t
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !tmp_tensor_t) -> !tmp_tensor_t
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%0 : !out_tensor_t) outs(%2 : !tmp_tensor_t) {
  ^bb0(%arg0: f32, %arg1: f32):
    %8 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %8 : f32
  } -> !tmp_tensor_t

  // This has been fused manually to avoid the fusion on tensors pass and reduce noise atm.
  %4 = tensor.empty() : !out_tensor_t
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%0, %3 : !out_tensor_t, !tmp_tensor_t) outs(%4 : !out_tensor_t) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %6 = arith.subf %arg0, %arg1 : f32
    %7 = math.exp %6 : f32
    linalg.yield %7 : f32
  } -> !out_tensor_t

  return %5: !out_tensor_t
}
