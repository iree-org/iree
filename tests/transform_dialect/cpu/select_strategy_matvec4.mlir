// RUN: not iree-compile \
// RUN:   --iree-codegen-transform-library-file-name=%p/transform_dialect_dummy_spec.mlir \
// RUN:   --iree-hal-target-backends=llvm-cpu \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT

// RUN: not iree-compile \
// RUN:   --iree-codegen-transform-library-file-name=%p/transform_dialect_dummy_spec.mlir \
// RUN:   --iree-llvmcpu-transform-dialect-select-strategy=%p/transform_dialect_dummy_select.mlir \
// RUN:   --iree-hal-target-backends=llvm-cpu \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MATCHER

// Check that we can select the transform dialect strategy used for the lowering
// by passing a "select file" to the compiler.
// We know the compilation will fail here because we use a dummy strategy, i.e.,
// the code is not actually lowered. Hence the `not` in the command line.

// Default we don't select a strategy and just use what is set in the attribute:
// print_config.
// DEFAULT: IR printer: from_config

// When using the matcher, check that we override what is already set in the
// attribute. I.e., use print_matvec4 instead of print_config.
// MATCHER: IR printer: from_selected4

!tlhs = tensor<4x768xf32>
!trhs = tensor<768x2304xf32>
!tres = tensor<4x2304xf32>

#blank_config = #iree_codegen.lowering_config<tile_sizes = []>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec=@print_config>
#config = #iree_codegen.compilation_info<lowering_config = #blank_config, translation_info = #translation, workgroup_size = []>

func.func @matmul_4x2304x768_f32(
  %a: !tlhs,
  %b: !trhs,
  %c: !tres) -> !tres attributes { llvm.emit_c_interface } {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b: !tlhs, !trhs) outs(%c: !tres)
    attrs = {compilation_info = #config} {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 {fastmath = #arith.fastmath<fast>} : f32
    %1 = arith.addf %arg2, %0  {fastmath = #arith.fastmath<fast>} : f32
    linalg.yield %1 : f32
  } -> !tres
  return %result : !tres
}
