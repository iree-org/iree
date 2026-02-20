// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma,canonicalize))" %s --split-input-file | FileCheck %s

#gpu_target_copy = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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
  // CHECK-DAG:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [4, 512] [1, 1]
  // CHECK-DAG:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [4, 512] [1, 1]

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
  // CHECK:   }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x512xf32>
}

// -----

#gpu_target_gather = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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
  // CHECK-DAG:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [1, 512] [1, 1]
  // CHECK-DAG:   %[[SLICE_INDICES:.+]] = tensor.extract_slice %[[INDICES]][%[[IV0]]] [1] [1]

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
  // CHECK:   }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: iree_linalg_ext.gather

  return %result : tensor<64x512xf32>
}

// -----

// Negative test: Skip coalesced DMA when innermost dimension < subgroup size. This is to ensure we do not go down
// the slow path (which is not implemented yet).

#gpu_target_small_inner = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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

#gpu_target_contiguous = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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
  // CHECK-DAG:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [16, 128] [1, 1]
  // CHECK-SAME:   : tensor<64x128xf32> to tensor<16x128xf32>
  // CHECK-DAG:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [16, 128] [1, 1]
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
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x128xf32>
}

// -----

// Test: Small innermost dimension with tensor.empty() output CAN be linearized.
// When output comes from tensor.empty(), we can use total elements instead of
// innermost dimension for the size check, enabling coalesced DMA.

#gpu_target_linearize = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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
  // CHECK-DAG:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], 0] [32, 16] [1, 1]
  // CHECK-SAME:   : tensor<128x16xf32> to tensor<32x16xf32>
  // CHECK-DAG:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[INIT_TILE]][%[[IV0]], 0] [32, 16] [1, 1]
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
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}

  // CHECK: return %[[WARP_RESULT]]
  // CHECK-NOT: linalg.copy

  return %result : tensor<128x16xf32>
}

// -----

// Test: 1D tensor copy distributes warps across the single dimension.
// This tests the 1D tile size computation logic for flattened copies.

#gpu_target_1d = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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

#gpu_target_no_linearize = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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

#gpu_target_extract_input = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
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

// Test: tensor.pad fusion into coalesced_gather_dma.
// When linalg.copy reads from tensor.pad, trace through to the original source
// and set in_bounds attribute based on padding.

#gpu_target_pad = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_pad = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_pad}>
#translation_pad = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_with_tensor_pad_fusion
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<121x64xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x64xf32>
func.func @copy_with_tensor_pad_fusion(%source: tensor<121x64xf32>, %init: tensor<4x64xf32>, %off: index, %sz: index, %high: index) -> tensor<4x64xf32>
  attributes {hal.executable.target = #exec_target_pad, translation_info = #translation_pad} {
  // Extract a dynamic slice.
  %extracted = tensor.extract_slice %source[%off, 0] [%sz, 64] [1, 1]
      : tensor<121x64xf32> to tensor<?x64xf32>

  // Pad to static size (only M dimension has padding).
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %extracted low[0, 0] high[%high, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<?x64xf32> to tensor<4x64xf32>

  // Copy from padded tensor.
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%padded : tensor<4x64xf32>)
    outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>

  // Key check: tensor.pad is fused - source is the extract_slice result, not the padded tensor.
  // in_bounds = [false, true] because M dim has dynamic padding, K dim has no padding.
  // CHECK: %[[EXTRACTED:.+]] = tensor.extract_slice %[[SRC]]
  // CHECK: scf.forall {{.*}} shared_outs(%[[OUTER_INIT:.+]] = %[[INIT]])
  // CHECK:   scf.forall (%[[LANE:.+]]) in (64) shared_outs(%[[INNER_INIT:.+]] = %[[OUTER_INIT]])
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[EXTRACTED]] into %[[INNER_INIT]] lane(%[[LANE]]) in_bounds [false, true]
  // CHECK-SAME:     : tensor<?x64xf32>, tensor<4x64xf32>, index
  // CHECK:     }
  // CHECK-NOT: tensor.pad

  return %result : tensor<4x64xf32>
}

// -----

// Test: tensor.pad fusion with multiple warps creates single-iteration wrapper forall.
// When tensor.pad is fused, subgroup-level tiling is skipped to ensure the DMA
// operates on the full padded buffer shape, not on smaller subviews.
// This is critical for correct delinearization in the lowering pass.

#gpu_target_pad_multi_warp = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_pad_multi_warp = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_pad_multi_warp}>
#translation_pad_multi_warp = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_with_tensor_pad_fusion_multi_warp
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<121x64xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x64xf32>
func.func @copy_with_tensor_pad_fusion_multi_warp(%source: tensor<121x64xf32>, %init: tensor<4x64xf32>, %off: index, %sz: index, %high: index) -> tensor<4x64xf32>
  attributes {hal.executable.target = #exec_target_pad_multi_warp, translation_info = #translation_pad_multi_warp} {
  // Extract a dynamic slice.
  %extracted = tensor.extract_slice %source[%off, 0] [%sz, 64] [1, 1]
      : tensor<121x64xf32> to tensor<?x64xf32>

  // Pad to static size (only M dimension has padding).
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %extracted low[0, 0] high[%high, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<?x64xf32> to tensor<4x64xf32>

  // Copy from padded tensor with 4 warps (256/64=4).
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%padded : tensor<4x64xf32>)
    outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>

  // Key check: With 4 warps available, normal tiling would create a warp-level
  // forall with step (1, 64) producing 4 iterations with 1x64 subviews.
  // For tensor.pad fusion, we instead create a single-iteration wrapper forall
  // with step (4, 64) - the full shape - so the DMA operates on 4x64 directly.
  // After canonicalization, identity extract_slices are eliminated.
  //
  // CHECK: %[[EXTRACTED:.+]] = tensor.extract_slice %[[SRC]]
  // CHECK: %[[WARP_RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) = (0, 0) to (4, 64) step (4, 64)
  // CHECK-SAME: shared_outs(%[[INIT_TILE:.+]] = %[[INIT]]) -> (tensor<4x64xf32>) {
  //
  // Thread-level forall with 64 lanes (uses outer forall's shared_out directly):
  // CHECK:   %[[THREAD_RESULT:.+]] = scf.forall (%[[LANE:.+]]) in (64) shared_outs(%[[INNER_INIT:.+]] = %[[INIT_TILE]])
  // CHECK:     scf.forall.in_parallel {
  // CHECK:       iree_gpu.coalesced_gather_dma %[[EXTRACTED]] into %[[INNER_INIT]] lane(%[[LANE]]) in_bounds [false, true]
  // CHECK-SAME:     : tensor<?x64xf32>, tensor<4x64xf32>, index
  // CHECK:     }
  // CHECK:   } {mapping = [#iree_gpu.lane_id<0>]}
  //
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice %[[THREAD_RESULT]] into %[[INIT_TILE]][0, 0] [4, 64] [1, 1]
  // CHECK:   }
  // CHECK: } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
  // CHECK-NOT: tensor.pad

  return %result : tensor<4x64xf32>
}

// -----

// Test: tensor.pad fusion bails out when source row size is not DWORD-aligned.
// On AMD CDNA, per-component range checking is performed for each DWORD.
// If a DWORD is partially out-of-bounds, the entire DWORD returns zero,
// causing incorrect results. We bail out to avoid the slow path.

#gpu_target_pad_unaligned = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_pad_unaligned = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_pad_unaligned}>
#translation_pad_unaligned = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_with_tensor_pad_unaligned_row
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<65x121xf16>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x124xf16>
func.func @copy_with_tensor_pad_unaligned_row(%source: tensor<65x121xf16>, %init: tensor<4x124xf16>, %off: index, %sz: index, %high_m: index) -> tensor<4x124xf16>
  attributes {hal.executable.target = #exec_target_pad_unaligned, translation_info = #translation_pad_unaligned} {
  // Extract a dynamic slice: tensor<?x121xf16>.
  // Row size = 121 * 2 bytes = 242 bytes, NOT 4-byte aligned.
  %extracted = tensor.extract_slice %source[%off, 0] [%sz, 121] [1, 1]
      : tensor<65x121xf16> to tensor<?x121xf16>

  // Pad to static size.
  %cst = arith.constant 0.0 : f16
  %padded = tensor.pad %extracted low[0, 0] high[%high_m, 3] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f16
  } : tensor<?x121xf16> to tensor<4x124xf16>

  // Copy from padded tensor.
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%padded : tensor<4x124xf16>)
    outs(%init : tensor<4x124xf16>) -> tensor<4x124xf16>

  // Source row size (121 * 2 = 242 bytes) is not DWORD-aligned.
  // Coalesced DMA bails out to avoid partial OOB in per-DWORD range checking.
  // The linalg.copy should remain unchanged.
  // CHECK: tensor.pad
  // CHECK: linalg.copy
  // CHECK-NOT: iree_gpu.coalesced_gather_dma

  return %result : tensor<4x124xf16>
}

// -----

// Test: Copy from load_from_buffer with fat_raw_buffer address space.
// DMA should be applied because fat_raw_buffer indicates the binding fits
// within the 2GB limit required for buffer instructions.

#gpu_target_fat_raw = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_fat_raw = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_fat_raw}>
#translation_fat_raw = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_from_fat_raw_buffer
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]: memref<64x512xbf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x512xbf16>
func.func @copy_from_fat_raw_buffer(
    %buf: memref<64x512xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %init: tensor<64x512xbf16>) -> tensor<64x512xbf16>
  attributes {hal.executable.target = #exec_target_fat_raw, translation_info = #translation_fat_raw} {
  %source = iree_codegen.load_from_buffer %buf
      : memref<64x512xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<64x512xbf16>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x512xbf16>)
    outs(%init : tensor<64x512xbf16>) -> tensor<64x512xbf16>

  // fat_raw_buffer source allows DMA.
  // 2 warps (128/64), 64 rows → step = 32 rows, 512 cols whole.
  // CHECK: %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUF]]
  // CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) {{.*}} shared_outs(%[[WARP_INIT:.+]] = %[[INIT]])
  // CHECK:   %[[SLICE_SRC:.+]] = tensor.extract_slice %[[SOURCE]][%[[IV0]], 0]
  // CHECK:   %[[SLICE_DST:.+]] = tensor.extract_slice %[[WARP_INIT]][%[[IV0]], 0]
  // CHECK:   scf.forall (%[[LANE:.+]]) in (64) shared_outs(%{{.+}} = %[[SLICE_DST]])
  // CHECK:     iree_gpu.coalesced_gather_dma %[[SLICE_SRC]] into %{{.+}} lane(%[[LANE]])
  // CHECK-NOT: linalg.copy

  return %result : tensor<64x512xbf16>
}

// -----

// Test: Small tensor copy from load_from_buffer with fat_raw_buffer.
// Even small tensors should get DMA when innermost dim >= min transfer size.

#gpu_target_fat_raw_small = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_fat_raw_small = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_fat_raw_small}>
#translation_fat_raw_small = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_from_fat_raw_buffer_small
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]: memref<4x256xbf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x256xbf16>
func.func @copy_from_fat_raw_buffer_small(
    %buf: memref<4x256xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %init: tensor<4x256xbf16>) -> tensor<4x256xbf16>
  attributes {hal.executable.target = #exec_target_fat_raw_small, translation_info = #translation_fat_raw_small} {
  %source = iree_codegen.load_from_buffer %buf
      : memref<4x256xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x256xbf16>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<4x256xbf16>)
    outs(%init : tensor<4x256xbf16>) -> tensor<4x256xbf16>

  // Small tensor (4x256 bf16) from fat_raw_buffer.
  // Innermost dim 256 >= min transfer (64*32/16=128), so DMA is applied.
  // 1 warp (64/64), 4 rows → step = 4 rows, 256 cols whole.
  // CHECK: %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUF]]
  // CHECK: scf.forall {{.*}} shared_outs(%{{.+}} = %[[INIT]])
  // CHECK:   scf.forall (%[[LANE:.+]]) in (64) shared_outs(%[[INNER_INIT:.+]] =
  // CHECK:     iree_gpu.coalesced_gather_dma %[[SOURCE]] into %[[INNER_INIT]] lane(%[[LANE]])
  // CHECK-NOT: linalg.copy

  return %result : tensor<4x256xbf16>
}

// -----

// Test: Copy from load_from_buffer with storage_buffer (non-fat_raw_buffer).
// DMA should NOT be applied because the source binding was not converted to
// fat_raw_buffer, indicating it exceeds the 2GB limit.

#gpu_target_storage_buf = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_storage_buf = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_storage_buf}>
#translation_storage_buf = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_from_non_fat_raw_buffer
// CHECK-SAME:    %[[BUF:[a-zA-Z0-9]+]]: memref<64x512xbf16, #hal.descriptor_type<storage_buffer>>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x512xbf16>
func.func @copy_from_non_fat_raw_buffer(
    %buf: memref<64x512xbf16, #hal.descriptor_type<storage_buffer>>,
    %init: tensor<64x512xbf16>) -> tensor<64x512xbf16>
  attributes {hal.executable.target = #exec_target_storage_buf, translation_info = #translation_storage_buf} {
  %source = iree_codegen.load_from_buffer %buf
      : memref<64x512xbf16, #hal.descriptor_type<storage_buffer>> -> tensor<64x512xbf16>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x512xbf16>)
    outs(%init : tensor<64x512xbf16>) -> tensor<64x512xbf16>

  // storage_buffer source: sourceIsNotFromFatRawBuffer returns true, DMA skipped.
  // The linalg.copy should remain unchanged.
  // CHECK: linalg.copy
  // CHECK-NOT: iree_gpu.coalesced_gather_dma

  return %result : tensor<64x512xbf16>
}

// -----

// Test: Copy from dispatch.tensor.load source.
// DMA should NOT be applied because dispatch.tensor.load indicates the binding
// was not bufferized to a memref (e.g., >2GB binding loaded via dispatch tensor
// path), so fat_raw_buffer is not available.

#pipeline_layout_dtl = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#gpu_target_dtl = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_dtl = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_dtl}>
#translation_dtl = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// CHECK-LABEL: func.func @copy_from_dispatch_tensor_load
func.func @copy_from_dispatch_tensor_load(%init: tensor<64x512xbf16>) -> tensor<64x512xbf16>
  attributes {hal.executable.target = #exec_target_dtl, translation_info = #translation_dtl} {
  %c0 = arith.constant 0 : index
  %binding = hal.interface.binding.subspan layout(#pipeline_layout_dtl) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xbf16>>
  %source = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0, 0], sizes = [64, 512], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xbf16>> -> tensor<64x512xbf16>
  %result = linalg.copy {lowering_config = #iree_gpu.use_global_load_dma}
    ins(%source : tensor<64x512xbf16>)
    outs(%init : tensor<64x512xbf16>) -> tensor<64x512xbf16>

  // dispatch.tensor.load source: sourceIsNotFromFatRawBuffer returns true,
  // DMA skipped. The linalg.copy should remain unchanged.
  // CHECK: linalg.copy
  // CHECK-NOT: iree_gpu.coalesced_gather_dma

  return %result : tensor<64x512xbf16>
}
