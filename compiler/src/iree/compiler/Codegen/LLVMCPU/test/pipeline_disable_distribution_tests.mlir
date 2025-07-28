// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target, iree-llvmcpu-check-ir-before-llvm-conversion))' --iree-llvmcpu-disable-distribution --split-input-file %s | FileCheck %s

// Test that iree_linalg_ext.map_scatter op is not generated when distribution
// is disabled. The op is used in the fallback solution when the pack op is not
// fusible in consumer fusion. We do not expect the op if the distribution is
// disabled. For more details, see
// https://github.com/iree-org/iree/issues/20723#issuecomment-3006445505

#executable_target_embedded_elf_arm_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+reserve-x18", max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
func.func @pack_without_distribution() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1980x192xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<248x192x8x1xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1980, 192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1980x192xf32>> -> tensor<1980x192xf32>
  %3 = tensor.empty() : tensor<248x192x8x1xf32>
  %pack = linalg.pack %2 padding_value(%cst : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<1980x192xf32> -> tensor<248x192x8x1xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [248, 192, 8, 1], strides = [1, 1, 1, 1] : tensor<248x192x8x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<248x192x8x1xf32>>
  return
}
// CHECK-LABEL: func.func @pack_without_distribution
// CHECK-NOT:     scf.forall
// CHECK-NOT:     iree_linalg_ext.map_scatter
