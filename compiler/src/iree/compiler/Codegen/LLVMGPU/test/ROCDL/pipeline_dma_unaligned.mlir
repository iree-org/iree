// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-llvmgpu-use-direct-load \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline, builtin.module(iree-codegen-llvmgpu-configuration-pipeline, iree-codegen-llvmgpu-rocdl-lowering-pipeline{include-llvm-lowering=false}), iree-codegen-translation-postprocessing-pipeline)))" \
// RUN:   %s | FileCheck %s

// Pipeline test for the GPUPushDownDMABoundsToConsumers + buffer_resource_cast
// validBytes hack on non-DWORD-aligned innermost rows. Ensures the chain
// pushdown -> bubble -> bufferize survives end-to-end:
//
//   1. GPUPushDownDMABoundsToConsumers wraps the DMA source with
//      iree_gpu.buffer_resource_cast carrying validBytes = roundUp(N + 4, 4).
//   2. GPUBubbleResourceCasts leaves casts with validBytes alone (so they
//      remain on the immediate root tensor).
//   3. Bufferization peels the binding's pre-existing fat_raw_buffer_cast
//      and emits a fresh amdgpu.fat_raw_buffer_cast with the explicit
//      validBytes attribute.
//   4. AMDGPULowerCoalescedDMAToGatherLDS lowers the DMA op to
//      amdgpu.gather_to_lds against the new buffer descriptor.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm",
  "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<
  arch = "gfx950", features = "", wgp = <
    compute = fp32|fp16, storage = b32|b16, subgroup = shuffle,
    dot = none, mma = [<MFMA_F32_16x16x32_F16>],
    subgroup_size_choices = [64, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128, simds_per_wgp = 4,
    vgpr_space_bits = 8192, dma_sizes = [32, 128]>>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>],
  flags = Indirect>

// CHECK-LABEL: hal.executable public @matmul_64x43x64
hal.executable public @matmul_64x43x64 {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @matmul_64x43x64 ordinal(0) layout(#pipeline_layout)
        count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // The LHS (64x43xf16, 86 bytes/row, NOT DWORD-aligned) is the source
      // that needs the validBytes override. The RHS (43x64xf16, 128 bytes/row,
      // DWORD-aligned) does not.
      //
      // CHECK-LABEL: func.func @matmul_64x43x64
      //
      // The LHS binding gets a fat_raw_buffer_cast with explicit validBytes
      // = roundUp(64*43*2, 4) + 4 = 5508. Without the hack the descriptor
      // defaults to the raw 5504-byte buffer size and the last lane's
      // partial-DWORD straddle gets HW-zeroed.
      //
      // CHECK-DAG: %[[VB:.+]] = arith.constant 5508 : i64
      // CHECK-DAG: amdgpu.fat_raw_buffer_cast %{{.+}} validBytes(%[[VB]]) {{.*}} : memref<64x43xf16, #hal.descriptor_type<storage_buffer>> to memref<64x43xf16, #amdgpu.address_space<fat_raw_buffer>>
      //
      // The RHS binding (DWORD-aligned innermost) is cast without an
      // override.
      // CHECK-DAG: amdgpu.fat_raw_buffer_cast %{{.+}} resetOffset : memref<43x64xf16, #hal.descriptor_type<storage_buffer>> to memref<43x64xf16, #amdgpu.address_space<fat_raw_buffer>>
      //
      // Both LHS and RHS bindings are loaded via gather_to_lds against their
      // fat_raw_buffer descriptors.
      // CHECK: amdgpu.gather_to_lds {{.*}} memref<64x43xf16, #amdgpu.address_space<fat_raw_buffer>>
      // CHECK: amdgpu.gather_to_lds {{.*}} memref<43x64xf16, #amdgpu.address_space<fat_raw_buffer>>
      func.func @matmul_64x43x64() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %lhs_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags("ReadOnly|Indirect")
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x43xf16>>
        %rhs_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0) flags("ReadOnly|Indirect")
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<43x64xf16>>
        %out_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            alignment(64) offset(%c0) flags(Indirect)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
        %lhs = iree_tensor_ext.dispatch.tensor.load %lhs_b, offsets = [0, 0],
            sizes = [64, 43], strides = [1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x43xf16>>
            -> tensor<64x43xf16>
        %rhs = iree_tensor_ext.dispatch.tensor.load %rhs_b, offsets = [0, 0],
            sizes = [43, 64], strides = [1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<43x64xf16>>
            -> tensor<43x64xf16>
        %empty = tensor.empty() : tensor<64x64xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
            -> tensor<64x64xf32>
        %res = linalg.matmul
            ins(%lhs, %rhs : tensor<64x43xf16>, tensor<43x64xf16>)
            outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
        iree_tensor_ext.dispatch.tensor.store %res, %out_b,
            offsets = [0, 0], sizes = [64, 64], strides = [1, 1]
            : tensor<64x64xf32>
            -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
        return
      }
    }
  }
}
