// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-lower-executable-target))' %s | FileCheck %s

// Verify that a custom pass pipeline specified via #iree_codegen.pass_pipeline
// attribute is executed by the LLVMCPU lower executable target pass.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"
}>

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
