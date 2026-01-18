// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma,canonicalize))" %s --split-input-file | FileCheck %s

#gpu_target_copy = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_copy = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_copy}>
#translation_copy = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 512, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<64x512xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x512xf32>
func.func @copy(%source: tensor<64x512xf32>, %init: tensor<64x512xf32>) -> tensor<64x512xf32>
  attributes {hal.executable.target = #exec_target_copy, translation_info = #translation_copy} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x512xf32>)
    outs(%init : tensor<64x512xf32>) -> tensor<64x512xf32>

  // Warp-level forall with contiguous subviews (columns kept whole):
  // With 16 warps (128*512/64/64) and 64 rows: step = ceil(64/16) = 4 rows, 512 cols (whole)
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (64, 512) step (4, 512)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<64x512xf32>) {
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [4, 512] [1, 1]
  // CHECK-SAME:   : tensor<64x512xf32> to tensor<4x512xf32>
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [4, 512] [1, 1]
  // CHECK-SAME:   : tensor<64x512xf32> to tensor<4x512xf32>

  // Thread-level forall:
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<4x512xf32>) {
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[SLICE_SRC]] into %[[THREAD_INIT]] lane(%[[LANE]])
  // CHECK-SAME:       : tensor<4x512xf32>, tensor<4x512xf32>, index
  // CHECK:     }

  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV0]], 0] [4, 512] [1, 1]
  // CHECK-SAME:     : tensor<4x512xf32> into tensor<64x512xf32>
  // CHECK:   }
  // CHECK: }

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x512xf32>
}

// -----

#gpu_target_gather = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_gather = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_gather}>
#translation_gather = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1024, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @gather
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<64x512xf32>
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]: tensor<64xi32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x512xf32>
func.func @gather(%source: tensor<64x512xf32>, %indices: tensor<64xi32>, %init: tensor<64x512xf32>) -> tensor<64x512xf32>
  attributes {hal.executable.target = #exec_target_gather, translation_info = #translation_gather} {
  %result = iree_linalg_ext.gather {lowering_config = #iree_gpu.use_global_load_dma}
    dimension_map = [0]
    ins(%source, %indices : tensor<64x512xf32>, tensor<64xi32>)
    outs(%init : tensor<64x512xf32>) -> tensor<64x512xf32>

  // Warp-level forall with contiguous subviews (columns kept whole):
  // With 64 warps and 64 rows: step = ceil(64/64) = 1 row, 512 cols (whole)
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (64, 512) step (1, 512)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<64x512xf32>) {
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [1, 512] [1, 1]
  // CHECK-SAME:   : tensor<64x512xf32> to tensor<1x512xf32>
  // CHECK:   %[[SLICE_INDICES:.+]] = tensor.extract_slice %[[INDICES]][%[[IV0]]] [1] [1]
  // CHECK-SAME:   : tensor<64xi32> to tensor<1xi32>

  // Thread-level forall:
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<1x512xf32>) {
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[SRC]][%[[SLICE_INDICES]]] into %[[THREAD_INIT]] lane(%[[LANE]])
  // CHECK-SAME:       : tensor<64x512xf32>, tensor<1xi32>, tensor<1x512xf32>, index
  // CHECK:     }
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV0]], 0] [1, 512] [1, 1]
  // CHECK-SAME:     : tensor<1x512xf32> into tensor<64x512xf32>
  // CHECK:   }
  // CHECK: }

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: iree_linalg_ext.gather

  return %result : tensor<64x512xf32>
}

// -----

// Test: Skip coalesced DMA when innermost dimension < subgroup size. This is to ensure we do not go down
// the slow path (which is not implemented yet).

#gpu_target_small_inner = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_small_inner = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_small_inner}>
#translation_small_inner = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_small_innermost_dim
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<64x32xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x32xf32>
func.func @copy_small_innermost_dim(%source: tensor<64x32xf32>, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
  attributes {hal.executable.target = #exec_target_small_inner, translation_info = #translation_small_inner} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x32xf32>)
    outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>

  // Innermost dimension (32) < subgroup size (64), so coalesced DMA should NOT be applied.
  // The linalg.copy should remain unchanged.
  // CHECK: linalg.copy
  // CHECK-NOT: iree_gpu.coalesced_gather_dma

  return %result : tensor<64x32xf32>
}

// -----

// Test: Prefer contiguous subviews when tiling would split the innermost dimension.
// With 64x128 tensor, workgroup_size=[256,1,1], subgroup_size=64:
// - 4 warps available (256/64)
// - Default tiling would split 128 columns into 2x64, creating non-contiguous subviews
// - Instead, we should tile rows to 16 (64/4) and keep columns whole (128)
// This ensures subviews are contiguous in memory.

#gpu_target_contiguous = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_contiguous = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_contiguous}>
#translation_contiguous = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_prefer_contiguous_subview
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<64x128xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x128xf32>
func.func @copy_prefer_contiguous_subview(%source: tensor<64x128xf32>, %init: tensor<64x128xf32>) -> tensor<64x128xf32>
  attributes {hal.executable.target = #exec_target_contiguous, translation_info = #translation_contiguous} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x128xf32>)
    outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>

  // With 4 warps and 64x128 tensor:
  // - Rows are tiled: step = 64/4 = 16
  // - Columns are NOT tiled (step = 128, full dimension) to ensure contiguous subviews
  // Warp-level forall iterates over rows with step 16, columns with step 128 (whole):
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (64, 128) step (16, 128)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<64x128xf32>) {

  // Key check: subviews are 16x128 (contiguous) not 64x64 (non-contiguous)
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [16, 128] [1, 1]
  // CHECK-SAME:   : tensor<64x128xf32> to tensor<16x128xf32>
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [16, 128] [1, 1]
  // CHECK-SAME:   : tensor<64x128xf32> to tensor<16x128xf32>

  // Thread-level forall distributes across lanes:
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<16x128xf32>) {
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[SLICE_SRC]] into %[[THREAD_INIT]] lane(%[[LANE]])
  // CHECK-SAME:       : tensor<16x128xf32>, tensor<16x128xf32>, index
  // CHECK:     }
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV0]], 0] [16, 128] [1, 1]
  // CHECK-SAME:     : tensor<16x128xf32> into tensor<64x128xf32>
  // CHECK:   }
  // CHECK: }

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x128xf32>
}
