// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmgpu-lower-executable-target))' %s | FileCheck %s

// Verify that a custom pass pipeline specified via #iree_codegen.pass_pipeline
// attribute is executed by the LLVMGPU lower executable target pass.

#executable_target = #hal.executable.target<"cuda", "cuda-nvptx-fb">

// The arith.addi with zero should be folded away by canonicalize.
func.func @test_custom_pipeline(%arg0: index) -> index attributes {
  hal.executable.target = #executable_target,
  translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.pass_pipeline<"canonicalize">>
} {
  %c0 = arith.constant 0 : index
  %0 = arith.addi %arg0, %c0 : index
  return %0 : index
}
// CHECK-LABEL: func.func @test_custom_pipeline
// CHECK-SAME:    (%[[ARG0:.+]]: index)
// CHECK-NEXT:    return %[[ARG0]]
