// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma,canonicalize))" %s --split-input-file | FileCheck %s

#gpu_target_copy = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_copy = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_copy}>
#translation_copy = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 64, 64] subgroup_size = 32>

// CHECK-LABEL: func.func @copy
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<64x1024xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x1024xf32>
func.func @copy(%source: tensor<64x1024xf32>, %init: tensor<64x1024xf32>) -> tensor<64x1024xf32>
  attributes {hal.executable.target = #exec_target_copy, translation_info = #translation_copy} {
  %result = linalg.copy {lowering_config = #iree_gpu.lowering_config<{subgroup = [4, 128]}>}
    ins(%source : tensor<64x1024xf32>)
    outs(%init : tensor<64x1024xf32>) -> tensor<64x1024xf32>

  // Warp-level forall:
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (64, 1024) step (4, 128)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<64x1024xf32>) {
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], %[[IV1]]] [4, 128] [1, 1]
  // CHECK-SAME:   : tensor<64x1024xf32> to tensor<4x128xf32>
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], %[[IV1]]] [4, 128] [1, 1]
  // CHECK-SAME:   : tensor<64x1024xf32> to tensor<4x128xf32>

  // Thread-level forall:
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[TIV0:.+]], %[[TIV1:.+]]) in (1, 32)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<4x128xf32>) {
  // CHECK:     %[[AFFINE:.+]] = affine.apply #{{.+}}(%[[TIV1]])
  // CHECK:     %[[ELEM_SLICE:.+]] = tensor.extract_slice %[[SLICE_SRC]][0, %[[AFFINE]]] [4, 4] [1, 1]
  // CHECK-SAME:     : tensor<4x128xf32> to tensor<4x4xf32>

  // CHECK:     scf.forall.in_parallel {
  // CHECK:       %[[DMA:.+]] = iree_gpu.coalesced_gather_dma %[[ELEM_SLICE]] into %[[THREAD_INIT]] lane(%[[TIV1]])
  // CHECK-SAME:       : tensor<4x4xf32>, tensor<4x128xf32>, index -> tensor<4x128xf32>
  // CHECK:     }

  // CHECK:   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV0]], %[[IV1]]] [4, 128] [1, 1]
  // CHECK-SAME:     : tensor<4x128xf32> into tensor<64x1024xf32>
  // CHECK:   }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x1024xf32>
}

// -----

#gpu_target_gather = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_gather = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_gather}>
#translation_gather = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 64, 64] subgroup_size = 64>

// CHECK-LABEL: func.func @gather
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<4x256xf32>
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]: tensor<4x256xindex>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x256xf32>
func.func @gather(%source: tensor<4x256xf32>, %indices: tensor<4x256xindex>, %init: tensor<4x256xf32>) -> tensor<4x256xf32>
  attributes {hal.executable.target = #exec_target_gather, translation_info = #translation_gather} {
  %result = iree_linalg_ext.gather {lowering_config = #iree_gpu.lowering_config<{subgroup = [4, 256]}>}
    dimension_map = [0]
    ins(%source, %indices : tensor<4x256xf32>, tensor<4x256xindex>)
    outs(%init : tensor<4x256xf32>) -> tensor<4x256xf32>

  // Warp-level forall:
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (4, 256) step (4, 256)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<4x256xf32>)
  // CHECK:   %[[SLICE_INDICES:.+]] = tensor.extract_slice %[[INDICES]][0, 0] [4, 1] [1, 1]
  // CHECK-SAME:   : tensor<4x256xindex> to tensor<4x1xindex>

  // Thread-level forall (loop count = innermost dimension size = 256):
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[TIV0:.+]], %[[TIV1:.+]]) in (1, 256)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[INIT_TILE]]) -> (tensor<4x256xf32>)
  // CHECK:     %[[ELEM_SLICE:.+]] = tensor.extract_slice %[[SRC]][0, %[[TIV1]]] [4, 1] [1, 1]
  // CHECK-SAME:     : tensor<4x256xf32> to tensor<4x1xf32>
  // CHECK:     %[[INDICES_VEC:.+]] = vector.transfer_read %[[SLICE_INDICES]]
  // CHECK-SAME:     : tensor<4x1xindex>, vector<4x1xindex>

  // CHECK:     scf.forall.in_parallel
  // CHECK:       %[[DMA:.+]] = iree_gpu.coalesced_gather_dma %[[ELEM_SLICE]][%[[INDICES_VEC]]] into %[[THREAD_INIT]] lane(%[[TIV1]])
  // CHECK-SAME:       : tensor<4x1xf32>, vector<4x1xindex>, tensor<4x256xf32>, index -> tensor<4x256xf32>

  // CHECK:   {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

  // CHECK:   scf.forall.in_parallel
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][0, 0] [4, 256] [1, 1]
  // CHECK-SAME:     : tensor<4x256xf32> into tensor<4x256xf32>

  // CHECK: {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: iree_linalg_ext.gather

  return %result : tensor<4x256xf32>
}
