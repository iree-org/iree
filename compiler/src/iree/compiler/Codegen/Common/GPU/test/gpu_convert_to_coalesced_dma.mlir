// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma))" %s --split-input-file | FileCheck %s

// TODO: Re-enable gather test once lowering config tiling is properly supported

#gpu_target_copy = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_copy = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_copy}>
#translation_copy = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @copy_with_max_load_bits
func.func @copy_with_max_load_bits(%source: tensor<1x128xf32>, %init: tensor<1x128xf32>) -> tensor<1x128xf32>
  attributes {hal.executable.target = #exec_target_copy, translation_info = #translation_copy} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<1x128xf32>)
    outs(%init : tensor<1x128xf32>) -> tensor<1x128xf32>

  // CHECK: scf.forall (%{{.*}}, %{{.*}}) in (1, 1)
  // CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (1, 32)
  // CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [1, 4] [1, 128]
  // CHECK:     iree_gpu.coalesced_gather_dma
  // CHECK-SAME: tensor<1x4xf32>, tensor<1x128xf32> -> tensor<1x128xf32>
  // CHECK:   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK-NOT: linalg.copy
  return %result : tensor<1x128xf32>
}

// -----

#gpu_target_2x256 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_2x256 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_2x256}>
#translation_2x256 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @copy_2x256_strided
func.func @copy_2x256_strided(%source: tensor<2x256xf32>, %init: tensor<2x256xf32>) -> tensor<2x256xf32>
  attributes {hal.executable.target = #exec_target_2x256, translation_info = #translation_2x256} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<2x256xf32>)
    outs(%init : tensor<2x256xf32>) -> tensor<2x256xf32>

  // CHECK: scf.forall (%{{.*}}, %{{.*}}) in (2, 2)
  // CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (1, 32)
  // CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [1, 4] [1, 128]
  // CHECK:     iree_gpu.coalesced_gather_dma
  // CHECK-SAME: tensor<1x4xf32>, tensor<1x128xf32> -> tensor<1x128xf32>
  // CHECK:   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK-NOT: linalg.copy
  return %result : tensor<2x256xf32>
}

// -----

#gpu_target_2x512 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_2x512 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_2x512}>
#translation_2x512 = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

// CHECK-LABEL: func.func @copy_2x512_strided
func.func @copy_2x512_strided(%source: tensor<2x512xf32>, %init: tensor<2x512xf32>) -> tensor<2x512xf32>
  attributes {hal.executable.target = #exec_target_2x512, translation_info = #translation_2x512} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<2x512xf32>)
    outs(%init : tensor<2x512xf32>) -> tensor<2x512xf32>

  // CHECK: scf.forall (%{{.*}}, %{{.*}}) in (2, 4)
  // CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (1, 32)
  // CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [1, 4] [1, 128]
  // CHECK:     iree_gpu.coalesced_gather_dma
  // CHECK-SAME: tensor<1x4xf32>, tensor<1x128xf32> -> tensor<1x128xf32>
  // CHECK:   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK-NOT: linalg.copy
  return %result : tensor<2x512xf32>
}
