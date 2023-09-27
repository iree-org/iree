// RUN: not iree-compile \
// RUN:   --iree-codegen-transform-library-file-name=%p/transform_dialect_dummy_spec.mlir \
// RUN:   --iree-llvmcpu-transform-dialect-select-strategy=%p/transform_dialect_dummy_select.mlir \
// RUN:   --iree-hal-target-backends=llvm-cpu \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MATCHER

// Check that we can select the transform dialect strategy used for the lowering
// by passing a "select file" to the compiler.
// We know the compilation will fail here because we use a dummy strategy, i.e.,
// the code is not actually lowered. Hence the `not` in the command line.

// When using the matcher, check that we can run the right transform dialect
// strategy, even if the matched instruction didn't have the
// "TransformDialectCodegen codegen_spec" attribute before.
// I.e., make sure the attribute gets added properly by observing that the
// expected transform gets called. In this case we want a print of use
// print_matvec6.
// MATCHER: IR printer: from_selected6

!tlhs = tensor<6x768xf32>
!trhs = tensor<768x2304xf32>
!tres = tensor<6x2304xf32>

func.func @matmul_6x2304x768_f32(
  %a: !tlhs,
  %b: !trhs,
  %c: !tres) -> !tres attributes { llvm.emit_c_interface } {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b: !tlhs, !trhs) outs(%c: !tres) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 {fastmath = #arith.fastmath<fast>} : f32
    %1 = arith.addf %arg2, %0  {fastmath = #arith.fastmath<fast>} : f32
    linalg.yield %1 : f32
  } -> !tres
  return %result : !tres
}
