// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

// Test scenario: While doing vectorization with masking strategy, and when the vector size is not a multiple of the element size,
// the vectorization could result in holes in the vectorized load/store, which might result in undefined behavior such as divide by zero.
// To avoid this, do not use masking strategy if the vectorized operation may result in undefined behavior. In this case, `arith.remsi`
// could result in divide by zero exception with masking strategy when the loop size is not a multiple of the vector size.

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 16}>
module {
  func.func @test_mod_vectorizing_strategy_peeling() attributes {hal.executable.target = #executable_target_system_elf_x86_64_}{
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>], flags = Indirect>]>) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<6xi32>>
    %1 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>], flags = Indirect>]>) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<6xi32>>
    %2 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>], flags = Indirect>]>) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<6xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<6xi32>> -> tensor<6xi32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<6xi32>> -> tensor<6xi32>
    %5 = tensor.empty() : tensor<6xi32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<6xi32>, tensor<6xi32>) outs(%5 : tensor<6xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %7 = arith.remsi %in, %in_0 : i32
      linalg.yield %7 : i32
    } -> tensor<6xi32>
    flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [6], strides = [1] : tensor<6xi32> -> !flow.dispatch.tensor<writeonly:tensor<6xi32>>
    return
  }
}

// CHECK: #translation = #iree_codegen.translation_info<CPUDoubleTilingExpert, {enable_loop_peeling}>
// CHECK-LABEL: @test_mod_vectorizing_strategy_peeling
// CHECK-SAME: attributes {hal.executable.target = #executable_target_system_elf_x86_64_, translation_info = #translation}
