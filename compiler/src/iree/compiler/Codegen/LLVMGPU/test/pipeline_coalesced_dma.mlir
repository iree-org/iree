// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds)))))" \
// RUN:   %s | FileCheck %s

// Test: Lowering coalesced_gather_dma to amdgpu.gather_to_lds for copying from
// global memory (fat_raw_buffer) to workgroup memory (LDS).

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @coalesced_dma_to_lds
hal.executable public @coalesced_dma_to_lds {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @lower_coalesced_dma ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @lower_coalesced_dma
      // For the pass to generate gather_to_lds, the transfer size per thread must match
      // one of the dma_sizes (32 or 128 bits). With 4x256 f32 elements and 64 threads,
      // each thread handles 256/64 = 4 elements per row = 128 bits, which matches.
      func.func @lower_coalesced_dma()
        attributes {
          hal.executable.target = #executable_target_rocm_hsaco_fb,
          translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<4x256xf32, #hal.descriptor_type<storage_buffer>>
        %assumed = memref.assume_alignment %0, 64 : memref<4x256xf32, #hal.descriptor_type<storage_buffer>>
        %source = amdgpu.fat_raw_buffer_cast %assumed resetOffset : memref<4x256xf32, #hal.descriptor_type<storage_buffer>> to memref<4x256xf32, #amdgpu.address_space<fat_raw_buffer>>
        %dest = memref.alloc() : memref<4x256xf32, #gpu.address_space<workgroup>>
        // CHECK: scf.forall (%[[THREAD_IDX:.+]]) in (64)
        scf.forall (%arg6) in (64) {
          // With 4x256 f32 elements and 64 threads:
          // - Each thread handles 256/64 = 4 elements per row
          // - 4 f32 elements = 16 bytes = 128 bits (matches dma_sizes)
          // - 4 rows means 4 gather_to_lds ops per thread
          // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
          // CHECK-DAG: %[[OFFSET:.+]] = arith.muli %[[THREAD_IDX]], %[[C4]]
          // CHECK-COUNT-4: amdgpu.gather_to_lds
          // CHECK-NOT: iree_gpu.coalesced_gather_dma
          iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
            (memref<4x256xf32, #amdgpu.address_space<fat_raw_buffer>>,
            memref<4x256xf32, #gpu.address_space<workgroup>>, index) -> ()
        } {mapping = [#gpu.thread<linear_dim_0>]}
        return
      }
    }
  }
}

// -----

// Lowering coalesced_gather_dma with dimensions matching lds_matmul_coalesced_dma.mlir e2e test.
// The e2e test uses 32x64 f32 matmul operands where innermost dim (64) == subgroup size (64).
// This mirrors the copy-to-LDS pattern for matmul prefetching.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @coalesced_dma_matmul_operand
hal.executable public @coalesced_dma_matmul_operand {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @lower_coalesced_dma_matmul ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @lower_coalesced_dma_matmul
      func.func @lower_coalesced_dma_matmul()
        attributes {
          hal.executable.target = #executable_target_rocm_hsaco_fb,
          translation_info = #translation} {
        %c0 = arith.constant 0 : index
        // This shape (32x64 f32) mirrors the LHS of the e2e matmul test.
        // innermost dim (64) == subgroup size (64), so each thread handles 1 element per row.
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x64xf32, #hal.descriptor_type<storage_buffer>>
        %assumed = memref.assume_alignment %0, 64 : memref<32x64xf32, #hal.descriptor_type<storage_buffer>>
        %source = amdgpu.fat_raw_buffer_cast %assumed resetOffset : memref<32x64xf32, #hal.descriptor_type<storage_buffer>> to memref<32x64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %dest = memref.alloc() : memref<32x64xf32, #gpu.address_space<workgroup>>
        // CHECK: scf.forall (%[[THREAD_IDX:.+]]) in (64)
        scf.forall (%arg6) in (64) {
          // With 32x64 f32 elements and 64 threads:
          // - Each thread handles 64/64 = 1 element per row
          // - 1 f32 element = 4 bytes = 32 bits (matches dma_sizes)
          // - 32 rows means 32 gather_to_lds ops per thread
          // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
          // CHECK-DAG: %[[OFFSET:.+]] = arith.muli %[[THREAD_IDX]], %[[C1]]
          // CHECK-COUNT-32: amdgpu.gather_to_lds
          // CHECK-NOT: iree_gpu.coalesced_gather_dma
          iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
            (memref<32x64xf32, #amdgpu.address_space<fat_raw_buffer>>,
            memref<32x64xf32, #gpu.address_space<workgroup>>, index) -> ()
        } {mapping = [#gpu.thread<linear_dim_0>]}
        return
      }
    }
  }
}

// -----

// Lowering coalesced_gather_dma with f16 type.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @coalesced_dma_f16
hal.executable public @coalesced_dma_f16 {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @lower_coalesced_dma_f16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @lower_coalesced_dma_f16
      func.func @lower_coalesced_dma_f16()
        attributes {
          hal.executable.target = #executable_target_rocm_hsaco_fb,
          translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x128xf16, #hal.descriptor_type<storage_buffer>>
        %assumed = memref.assume_alignment %0, 64 : memref<2x128xf16, #hal.descriptor_type<storage_buffer>>
        %source = amdgpu.fat_raw_buffer_cast %assumed resetOffset : memref<2x128xf16, #hal.descriptor_type<storage_buffer>> to memref<2x128xf16, #amdgpu.address_space<fat_raw_buffer>>
        %dest = memref.alloc() : memref<2x128xf16, #gpu.address_space<workgroup>>
        // CHECK: scf.forall (%[[THREAD_IDX:.+]]) in (64)
        scf.forall (%arg6) in (64) {
          // For f16: 2x128 elements, 64 threads
          // Each thread handles 128/64 = 2 elements per row
          // Vector type is 2xf16 (4 bytes, fits in 32-bit DMA)
          // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
          // CHECK-DAG: %[[OFFSET:.+]] = arith.muli %[[THREAD_IDX]], %[[C2]]
          // CHECK-COUNT-2: amdgpu.gather_to_lds
          // CHECK-NOT: iree_gpu.coalesced_gather_dma
          iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
            (memref<2x128xf16, #amdgpu.address_space<fat_raw_buffer>>,
            memref<2x128xf16, #gpu.address_space<workgroup>>, index) -> ()
        } {mapping = [#gpu.thread<linear_dim_0>]}
        return
      }
    }
  }
}

// -----

// Test: Multiple DMA transfers per lane (N-transfer mode).
// When innermost dimension > subgroupSize * elementsPerLane, multiple GatherToLDS
// ops are generated to cover the entire dimension.
//
// With 4x128 f32 elements and 64 threads:
//   - innermost = 128, dma_sizes = [32, 128]
//   - dma_size=128: elementsPerLane=4, totalElementsPerTransfer=256, 128 % 256 != 0 -> skip
//   - dma_size=32: elementsPerLane=1, totalElementsPerTransfer=64, 128 % 64 = 0 -> 2 transfers
// Each row requires 2 gather_to_lds ops (128/64 = 2 transfers per row).
// 4 rows * 2 transfers = 8 total gather_to_lds ops.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @coalesced_dma_multi_transfer
hal.executable public @coalesced_dma_multi_transfer {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @lower_coalesced_dma_multi_transfer ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @lower_coalesced_dma_multi_transfer
      func.func @lower_coalesced_dma_multi_transfer()
        attributes {
          hal.executable.target = #executable_target_rocm_hsaco_fb,
          translation_info = #translation} {
        %c0 = arith.constant 0 : index
        // CHECK: %[[SRC:.+]] = amdgpu.fat_raw_buffer_cast
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<4x128xf32, #hal.descriptor_type<storage_buffer>>
        %assumed = memref.assume_alignment %0, 64 : memref<4x128xf32, #hal.descriptor_type<storage_buffer>>
        %source = amdgpu.fat_raw_buffer_cast %assumed resetOffset : memref<4x128xf32, #hal.descriptor_type<storage_buffer>> to memref<4x128xf32, #amdgpu.address_space<fat_raw_buffer>>
        // CHECK: %[[DST:.+]] = memref.alloc()
        %dest = memref.alloc() : memref<4x128xf32, #gpu.address_space<workgroup>>
        // CHECK: scf.forall (%[[LANE_ID:.+]]) in (64)
        scf.forall (%arg6) in (64) {
          // With 4x128 f32 elements and 64 threads:
          // - innermost=128, can't use 128-bit (128 % 256 != 0), use 32-bit
          // - elementsPerLane = 1, totalElementsPerTransfer = 64
          // - Each row needs 128/64 = 2 transfers (at offsets 0 and 64)
          // - 4 rows * 2 transfers = 8 gather_to_lds ops total
          // CHECK-DAG: %[[C1:.+]] = arith.constant 1
          // CHECK-DAG: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C1]]
          //
          // Row 0, Transfer 1: source[0, 0 + lane_offset], dest[0, 0]
          // CHECK: %[[SRC_COL0_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{c0.+}}, %[[SRC_COL0_T0]]], %[[DST]][%{{c0.+}}, %{{c0.+}}] : vector<1xf32>
          //
          // Row 0, Transfer 2: source[0, 64 + lane_offset], dest[0, 64]
          // CHECK: %[[C64:.+]] = arith.constant 64
          // CHECK: %[[SRC_COL0_T1:.+]] = arith.addi %[[C64]], %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{c0.+}}, %[[SRC_COL0_T1]]], %[[DST]][%{{c0.+}}, %[[C64]]] : vector<1xf32>
          //
          // Row 1, Transfer 1: source[1, 0 + lane_offset], dest[1, 0]
          // CHECK: %[[ROW1:.+]] = arith.constant 1
          // CHECK: %[[SRC_COL1_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[SRC_COL1_T0]]], %[[DST]][%[[ROW1]], %{{c0.+}}] : vector<1xf32>
          //
          // Row 1, Transfer 2: source[1, 64 + lane_offset], dest[1, 64]
          // CHECK: %[[SRC_COL1_T1:.+]] = arith.addi %{{c64.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{.+}}, %[[SRC_COL1_T1]]], %[[DST]][%{{.+}}, %{{c64.+}}] : vector<1xf32>
          //
          // Row 2, Transfer 1: source[2, 0 + lane_offset], dest[2, 0]
          // CHECK: %[[ROW2:.+]] = arith.constant 2
          // CHECK: %[[SRC_COL2_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW2]], %[[SRC_COL2_T0]]], %[[DST]][%[[ROW2]], %{{c0.+}}] : vector<1xf32>
          //
          // Row 2, Transfer 2: source[2, 64 + lane_offset], dest[2, 64]
          // CHECK: %[[SRC_COL2_T1:.+]] = arith.addi %{{c64.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{.+}}, %[[SRC_COL2_T1]]], %[[DST]][%{{.+}}, %{{c64.+}}] : vector<1xf32>
          //
          // Row 3, Transfer 1: source[3, 0 + lane_offset], dest[3, 0]
          // CHECK: %[[ROW3:.+]] = arith.constant 3
          // CHECK: %[[SRC_COL3_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW3]], %[[SRC_COL3_T0]]], %[[DST]][%[[ROW3]], %{{c0.+}}] : vector<1xf32>
          //
          // Row 3, Transfer 2: source[3, 64 + lane_offset], dest[3, 64]
          // CHECK: %[[SRC_COL3_T1:.+]] = arith.addi %{{c64.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{.+}}, %[[SRC_COL3_T1]]], %[[DST]][%{{.+}}, %{{c64.+}}] : vector<1xf32>
          // CHECK-NOT: amdgpu.gather_to_lds
          // CHECK-NOT: iree_gpu.coalesced_gather_dma
          iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
            memref<4x128xf32, #amdgpu.address_space<fat_raw_buffer>>,
            memref<4x128xf32, #gpu.address_space<workgroup>>, index
        } {mapping = [#gpu.thread<linear_dim_0>]}
        return
      }
    }
  }
}

// -----

// Test: Multiple DMA transfers with 128-bit DMA size.
// With 2x512 f32 elements and 64 threads:
//   - innermost = 512, dma_sizes = [32, 128]
//   - dma_size=128: elementsPerLane=4, totalElementsPerTransfer=256, 512 % 256 = 0 -> 2 transfers
// Each row requires 2 gather_to_lds ops using 128-bit (4 f32) transfers.
// 2 rows * 2 transfers = 4 total gather_to_lds ops.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx942", features = "", wgp = <
    compute = fp64|fp32|fp16|int64|int32|int16|int8,
    storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
    dot = dp4xi8toi32, mma = [], subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @coalesced_dma_multi_transfer_128bit
hal.executable public @coalesced_dma_multi_transfer_128bit {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @lower_coalesced_dma_multi_transfer_128bit ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // CHECK-LABEL: func.func @lower_coalesced_dma_multi_transfer_128bit
      func.func @lower_coalesced_dma_multi_transfer_128bit()
        attributes {
          hal.executable.target = #executable_target_rocm_hsaco_fb,
          translation_info = #translation} {
        %c0 = arith.constant 0 : index
        // CHECK: %[[SRC:.+]] = amdgpu.fat_raw_buffer_cast
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x512xf32, #hal.descriptor_type<storage_buffer>>
        %assumed = memref.assume_alignment %0, 64 : memref<2x512xf32, #hal.descriptor_type<storage_buffer>>
        %source = amdgpu.fat_raw_buffer_cast %assumed resetOffset : memref<2x512xf32, #hal.descriptor_type<storage_buffer>> to memref<2x512xf32, #amdgpu.address_space<fat_raw_buffer>>
        // CHECK: %[[DST:.+]] = memref.alloc()
        %dest = memref.alloc() : memref<2x512xf32, #gpu.address_space<workgroup>>
        // CHECK: scf.forall (%[[LANE_ID:.+]]) in (64)
        scf.forall (%arg6) in (64) {
          // With 2x512 f32 elements and 64 threads:
          // - innermost=512, use 128-bit (512 % 256 = 0)
          // - elementsPerLane = 4, totalElementsPerTransfer = 256
          // - Each row needs 512/256 = 2 transfers (at offsets 0 and 256)
          // - 2 rows * 2 transfers = 4 gather_to_lds ops total
          // CHECK-DAG: %[[C4:.+]] = arith.constant 4
          // CHECK-DAG: %[[LANE_OFFSET:.+]] = arith.muli %[[LANE_ID]], %[[C4]]
          //
          // Row 0, Transfer 1: source[0, 0 + lane_offset], dest[0, 0]
          // CHECK: %[[SRC_COL0_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{c0.+}}, %[[SRC_COL0_T0]]], %[[DST]][%{{c0.+}}, %{{c0.+}}] : vector<4xf32>
          //
          // Row 0, Transfer 2: source[0, 256 + lane_offset], dest[0, 256]
          // CHECK: %[[C256:.+]] = arith.constant 256
          // CHECK: %[[SRC_COL0_T1:.+]] = arith.addi %[[C256]], %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{c0.+}}, %[[SRC_COL0_T1]]], %[[DST]][%{{c0.+}}, %[[C256]]] : vector<4xf32>
          //
          // Row 1, Transfer 1: source[1, 0 + lane_offset], dest[1, 0]
          // CHECK: %[[ROW1:.+]] = arith.constant 1
          // CHECK: %[[SRC_COL1_T0:.+]] = arith.addi %{{c0.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%[[ROW1]], %[[SRC_COL1_T0]]], %[[DST]][%[[ROW1]], %{{c0.+}}] : vector<4xf32>
          //
          // Row 1, Transfer 2: source[1, 256 + lane_offset], dest[1, 256]
          // CHECK: %[[SRC_COL1_T1:.+]] = arith.addi %{{c256.+}}, %[[LANE_OFFSET]]
          // CHECK: amdgpu.gather_to_lds %[[SRC]][%{{.+}}, %[[SRC_COL1_T1]]], %[[DST]][%{{.+}}, %{{c256.+}}] : vector<4xf32>
          // CHECK-NOT: amdgpu.gather_to_lds
          // CHECK-NOT: iree_gpu.coalesced_gather_dma
          iree_gpu.coalesced_gather_dma %source into %dest lane(%arg6) :
            memref<2x512xf32, #amdgpu.address_space<fat_raw_buffer>>,
            memref<2x512xf32, #gpu.address_space<workgroup>>, index
        } {mapping = [#gpu.thread<linear_dim_0>]}
        return
      }
    }
  }
}
