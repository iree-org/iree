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
#translation_copy = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 64, 64] subgroup_size = 32>

// CHECK-LABEL: func.func @copy_with_lowering_config
func.func @copy_with_lowering_config(%source: tensor<64x1024xf32>, %init: tensor<64x1024xf32>) -> tensor<64x1024xf32>
  attributes {hal.executable.target = #exec_target_copy, translation_info = #translation_copy} {
  %result = linalg.copy {lowering_config = #iree_gpu.lowering_config<{subgroup = [4, 128]}>}
    ins(%source : tensor<64x1024xf32>)
    outs(%init : tensor<64x1024xf32>) -> tensor<64x1024xf32>

  // Warp-level forall: tiles the 64x1024 tensor into 4x128 tiles
  // CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (64, 1024) step (4, 128)
  // CHECK-SAME: shared_outs(%{{.*}} = %{{.*}}) -> (tensor<64x1024xf32>)
  // CHECK:   tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [4, 128] [1, 1] : tensor<64x1024xf32> to tensor<4x128xf32>
  // CHECK:   tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [4, 128] [1, 1] : tensor<64x1024xf32> to tensor<4x128xf32>

  // Thread-level forall: distributes the 4x128 tile across 1x32 threads
  // CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (1, 32) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<4x128xf32>)
  // CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [4, 4] [1, 1] : tensor<4x128xf32> to tensor<4x4xf32>
  // CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [4, 4] [1, 1] : tensor<4x128xf32> to tensor<4x4xf32>

  // DMA operation in the in_parallel region
  // CHECK:     scf.forall.in_parallel
  // CHECK:       iree_gpu.coalesced_gather_dma %{{.*}} into %{{.*}} lane(%{{.*}}) : tensor<4x4xf32>, tensor<4x128xf32>, index -> tensor<4x128xf32>

  // CHECK:   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK:   scf.forall.in_parallel
  // CHECK:     tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] [4, 128] [1, 1]
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK-NOT: linalg.copy

  return %result : tensor<64x1024xf32>
}
