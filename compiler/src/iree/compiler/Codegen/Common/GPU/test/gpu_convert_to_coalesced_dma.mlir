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

// Negative test: Skip coalesced DMA when innermost dimension < subgroup size. This is to ensure we do not go down
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

// Negative test: Skip coalesced DMA when elements are not aligned to transfer size.
// With bf16 type, dma_sizes = [32, 128], and subgroup_size = 64:
// - 32-bit DMA: elementsPerLane = 2, elementsPerTransfer = 64 * 2 = 128
// - 128-bit DMA: elementsPerLane = 8, elementsPerTransfer = 64 * 8 = 512
// - minElementsPerTransfer = 128
// - 320 bf16 elements: 320 % 128 = 64 != 0, so alignment check fails.

#gpu_target_not_aligned = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_not_aligned = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_not_aligned}>
#translation_not_aligned = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_not_aligned_to_dma
// CHECK-SAME:    %[[SRC_BUF:[a-zA-Z0-9]+]]: memref<320xbf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<320xbf16>
func.func @copy_not_aligned_to_dma(%source_buffer: memref<320xbf16, #amdgpu.address_space<fat_raw_buffer>>, %init: tensor<320xbf16>) -> tensor<320xbf16>
  attributes {hal.executable.target = #exec_target_not_aligned, translation_info = #translation_not_aligned} {
  %source = iree_codegen.load_from_buffer %source_buffer : memref<320xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<320xbf16>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<320xbf16>)
    outs(%init : tensor<320xbf16>) -> tensor<320xbf16>

  // CHECK: linalg.copy
  // CHECK-NOT: iree_gpu.coalesced_gather_dma

  return %result : tensor<320xbf16>
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

// -----

// Test: Small innermost dimension with tensor.empty() output CAN be linearized.
// When output comes from tensor.empty(), we can use total elements instead of
// innermost dimension for the size check, enabling coalesced DMA.

#gpu_target_linearize = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32]
>>

#exec_target_linearize = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_linearize}>
#translation_linearize = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_small_innermost_linearized
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<128x16xf32>
func.func @copy_small_innermost_linearized(%source: tensor<128x16xf32>) -> tensor<128x16xf32>
  attributes {hal.executable.target = #exec_target_linearize, translation_info = #translation_linearize} {
  %empty = tensor.empty() : tensor<128x16xf32>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<128x16xf32>)
    outs(%empty : tensor<128x16xf32>) -> tensor<128x16xf32>

  // Innermost dimension (16) < minElementsPerTransfer (64), but since output is
  // tensor.empty(), we use total elements (2048) for the check, which passes.
  // With 4 warps (256/64), rows are tiled to 32 (128/4), columns kept whole at 16.

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<128x16xf32>

  // Warp-level forall: step (32, 16) distributes 128 rows across 4 warps
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (128, 16) step (32, 16)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[EMPTY]]) -> (tensor<128x16xf32>) {
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [32, 16] [1, 1]
  // CHECK-SAME:   : tensor<128x16xf32> to tensor<32x16xf32>
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [32, 16] [1, 1]
  // CHECK-SAME:   : tensor<128x16xf32> to tensor<32x16xf32>

  // Thread-level forall with 64 lanes
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<32x16xf32>) {
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[SLICE_SRC]] into %[[THREAD_INIT]] lane(%[[LANE]])
  // CHECK-SAME:       : tensor<32x16xf32>, tensor<32x16xf32>, index
  // CHECK:     }
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV0]], 0] [32, 16] [1, 1]
  // CHECK-SAME:     : tensor<32x16xf32> into tensor<128x16xf32>
  // CHECK:   }
  // CHECK: }

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<128x16xf32>
}

// -----

// Test: 1D tensor copy distributes warps across the single dimension.
// This tests the 1D tile size computation logic for flattened copies.

#gpu_target_1d = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32]
>>

#exec_target_1d = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_1d}>
#translation_1d = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_1d_tensor
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<2048xf32>
func.func @copy_1d_tensor(%source: tensor<2048xf32>) -> tensor<2048xf32>
  attributes {hal.executable.target = #exec_target_1d, translation_info = #translation_1d} {
  %empty = tensor.empty() : tensor<2048xf32>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<2048xf32>)
    outs(%empty : tensor<2048xf32>) -> tensor<2048xf32>

  // With 4 warps (256/64) and 2048 elements:
  // - Tile size = ceil(2048/4) = 512 elements per warp
  // - Step = 512, distributing the single dimension across warps

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2048xf32>

  // Warp-level forall: step (512) distributes 2048 elements across 4 warps
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV:.+]]) = (0) to (2048) step (512)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[EMPTY]]) -> (tensor<2048xf32>) {
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV]]] [512] [1]
  // CHECK-SAME:   : tensor<2048xf32> to tensor<512xf32>
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV]]] [512] [1]
  // CHECK-SAME:   : tensor<2048xf32> to tensor<512xf32>

  // Thread-level forall with 64 lanes
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK-SAME:   shared_outs(%[[THREAD_INIT:.+]] = %[[SLICE_DST]]) -> (tensor<512xf32>) {
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[SLICE_SRC]] into %[[THREAD_INIT]] lane(%[[LANE]])
  // CHECK-SAME:       : tensor<512xf32>, tensor<512xf32>, index
  // CHECK:     }
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][%[[IV]]] [512] [1]
  // CHECK-SAME:     : tensor<512xf32> into tensor<2048xf32>
  // CHECK:   }
  // CHECK: }

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<2048xf32>
}

// -----

// Negative test: Small innermost dimension with non-tensor.empty output should
// NOT be linearized. The copy should remain unchanged because:
// 1. Innermost dim (16) < minElementsPerTransfer (64)
// 2. Output is a function argument, not tensor.empty, so we can't linearize

#gpu_target_no_linearize = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32]
>>

#exec_target_no_linearize = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_no_linearize}>
#translation_no_linearize = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_small_innermost_no_linearize
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<128x16xf32>
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]: tensor<128x16xf32>
func.func @copy_small_innermost_no_linearize(%source: tensor<128x16xf32>, %dest: tensor<128x16xf32>) -> tensor<128x16xf32>
  attributes {hal.executable.target = #exec_target_no_linearize, translation_info = #translation_no_linearize} {
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<128x16xf32>)
    outs(%dest : tensor<128x16xf32>) -> tensor<128x16xf32>

  // Innermost dimension (16) < minElementsPerTransfer (64), and output is not
  // tensor.empty(), so linearization is not possible. The copy should remain.

  // CHECK: %[[RESULT:.+]] = linalg.copy
  // CHECK-SAME: ins(%[[SRC]] : tensor<128x16xf32>)
  // CHECK-SAME: outs(%[[DST]] : tensor<128x16xf32>)
  // CHECK: return %[[RESULT]]

  return %result : tensor<128x16xf32>
}

// -----

// Test: Copy with extract_slice input (source from a slice of a larger tensor).
// The copy should be converted to coalesced DMA when the input comes from an
// extract_slice with contiguous innermost dimensions.

#gpu_target_extract_input = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32]
>>

#exec_target_extract_input = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_extract_input}>
#translation_extract_input = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_with_extract_slice_input
// CHECK-SAME:    %[[LARGE_SRC:[a-zA-Z0-9]+]]: tensor<256x128xf32>
func.func @copy_with_extract_slice_input(%large_source: tensor<256x128xf32>) -> tensor<64x128xf32>
  attributes {hal.executable.target = #exec_target_extract_input, translation_info = #translation_extract_input} {
  // Extract a contiguous slice from the larger source tensor.
  // The innermost dimension (128) is fully taken, so this is a contiguous slice.
  %c32 = arith.constant 32 : index
  %slice = tensor.extract_slice %large_source[%c32, 0] [64, 128] [1, 1]
    : tensor<256x128xf32> to tensor<64x128xf32>

  %empty = tensor.empty() : tensor<64x128xf32>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%slice : tensor<64x128xf32>)
    outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>

  // The copy should be converted to coalesced DMA.
  // With 4 warps (256/64) and 64x128 tensor:
  // - Rows are tiled: step = 64/4 = 16
  // - Columns are NOT tiled (step = 128, full dimension) to ensure contiguous subviews

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64x128xf32>
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (64, 128) step (16, 128)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[EMPTY]]) -> (tensor<64x128xf32>) {

  // The source slice is further sliced for each warp
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]]

  // Thread-level forall with 64 lanes
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64)
  // CHECK:     iree_gpu.coalesced_gather_dma
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x128xf32>
}

// -----

// Test: Two copies both with use_global_load_dma and both DMA-convertible.
// Both should be converted to coalesced DMA.

#gpu_target_pair = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_pair = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_pair}>
#translation_pair = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_pair_both_convertible
func.func @copy_pair_both_convertible(
    %src0: tensor<64x128xf32>, %init0: tensor<64x128xf32>,
    %src1: tensor<32x256xf32>, %init1: tensor<32x256xf32>)
    -> (tensor<64x128xf32>, tensor<32x256xf32>)
  attributes {hal.executable.target = #exec_target_pair, translation_info = #translation_pair} {
  %r0 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src0 : tensor<64x128xf32>)
    outs(%init0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %r1 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src1 : tensor<32x256xf32>)
    outs(%init1 : tensor<32x256xf32>) -> tensor<32x256xf32>

  // Both copies should be converted since both are DMA-convertible.
  // CHECK: iree_gpu.coalesced_gather_dma
  // CHECK: iree_gpu.coalesced_gather_dma
  // CHECK-NOT: linalg.copy

  return %r0, %r1 : tensor<64x128xf32>, tensor<32x256xf32>
}

// -----

// Negative test: Two copies with use_global_load_dma, but one has misaligned
// innermost dimension. Neither should be converted.

#gpu_target_pair_one_bad = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_pair_one_bad = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_pair_one_bad}>
#translation_pair_one_bad = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_pair_one_unconvertible
func.func @copy_pair_one_unconvertible(
    %src0: tensor<64x128xf32>, %init0: tensor<64x128xf32>,
    %src1: tensor<64x32xf32>, %init1: tensor<64x32xf32>)
    -> (tensor<64x128xf32>, tensor<64x32xf32>)
  attributes {hal.executable.target = #exec_target_pair_one_bad, translation_info = #translation_pair_one_bad} {
  // First copy is DMA-convertible (128 % 64 == 0), but second is not (32 % 64 != 0).
  // Since not ALL copies are convertible, neither should be converted.
  %r0 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src0 : tensor<64x128xf32>)
    outs(%init0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %r1 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src1 : tensor<64x32xf32>)
    outs(%init1 : tensor<64x32xf32>) -> tensor<64x32xf32>

  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: linalg.copy
  // CHECK-SAME: lowering_config = #iree_gpu.derived_thread_config
  // CHECK: linalg.copy
  // CHECK-SAME: lowering_config = #iree_gpu.derived_thread_config

  return %r0, %r1 : tensor<64x128xf32>, tensor<64x32xf32>
}

// -----

// Test: Mixed attributes (1 use_global_load_dma + 1 derived_thread_config),
// both DMA-convertible. Both should be upgraded and converted to DMA.

#gpu_target_mixed_ok = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_mixed_ok = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_mixed_ok}>
#translation_mixed_ok = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_mixed_attrs_both_convertible
func.func @copy_mixed_attrs_both_convertible(
    %src0: tensor<64x128xf32>, %init0: tensor<64x128xf32>,
    %src1: tensor<32x256xf32>, %init1: tensor<32x256xf32>)
    -> (tensor<64x128xf32>, tensor<32x256xf32>)
  attributes {hal.executable.target = #exec_target_mixed_ok, translation_info = #translation_mixed_ok} {
  %r0 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src0 : tensor<64x128xf32>)
    outs(%init0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  // This copy has derived_thread_config but IS DMA-convertible.
  // The pass should upgrade it to use_global_load_dma and convert both.
  %r1 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
    ins(%src1 : tensor<32x256xf32>)
    outs(%init1 : tensor<32x256xf32>) -> tensor<32x256xf32>

  // Both copies should be converted since both are DMA-convertible.
  // CHECK: iree_gpu.coalesced_gather_dma
  // CHECK: iree_gpu.coalesced_gather_dma
  // CHECK-NOT: linalg.copy

  return %r0, %r1 : tensor<64x128xf32>, tensor<32x256xf32>
}

// -----

// Negative test: Mixed attributes (1 use_global_load_dma + 1 derived_thread_config),
// but the derived_thread_config copy is NOT DMA-convertible. Neither should be
// converted.

#gpu_target_mixed_bad = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_mixed_bad = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_mixed_bad}>
#translation_mixed_bad = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @copy_mixed_attrs_one_unconvertible
func.func @copy_mixed_attrs_one_unconvertible(
    %src0: tensor<64x128xf32>, %init0: tensor<64x128xf32>,
    %src1: tensor<64x32xf32>, %init1: tensor<64x32xf32>)
    -> (tensor<64x128xf32>, tensor<64x32xf32>)
  attributes {hal.executable.target = #exec_target_mixed_bad, translation_info = #translation_mixed_bad} {
  // First copy is DMA-convertible, but second is not (32 % 64 != 0).
  %r0 = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%src0 : tensor<64x128xf32>)
    outs(%init0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %r1 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
    ins(%src1 : tensor<64x32xf32>)
    outs(%init1 : tensor<64x32xf32>) -> tensor<64x32xf32>

  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: linalg.copy
  // CHECK-SAME: lowering_config = #iree_gpu.derived_thread_config
  // CHECK: linalg.copy
  // CHECK-SAME: lowering_config = #iree_gpu.derived_thread_config

  return %r0, %r1 : tensor<64x128xf32>, tensor<64x32xf32>
}
