// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:     %s | FileCheck %s

// Test that the vector distribute pipeline correctly handles attention with
// a K1 dimension (63) that is not aligned to the reduction tile size (64),
// requiring masking.

// CHECK-LABEL: func.func @attention_dispatch_0_attention_20x16x63x4096x64()
// CHECK:   arith.cmpi slt
// CHECK:   scf.for
// CHECK:     arith.select
// CHECK:   scf.yield
hal.executable private @attention_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>) {
    hal.executable.export public @attention_dispatch_0_attention_20x16x63x4096x64 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_20x16x63x4096x64() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [512, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x16x63xf16, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<20x16x63xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x63xf16, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4096x63xf16, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<20x4096x63xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4096x63xf16, #amdgpu.address_space<fat_raw_buffer>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4096x64xf16, #hal.descriptor_type<storage_buffer>>
        %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<20x4096x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4096x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>>
        %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %8 = iree_codegen.load_from_buffer %1 : memref<20x16x63xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x16x63xf16>
        %9 = iree_codegen.load_from_buffer %3 : memref<20x4096x63xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4096x63xf16>
        %10 = iree_codegen.load_from_buffer %5 : memref<20x4096x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4096x64xf16>
        %11 = tensor.empty() : tensor<20x16x64xf16>
        %12 = iree_linalg_ext.attention {
          decomposition_config = {
            pv_attrs = {
              lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 64], [1, 0, 4, 3]], subgroup_basis = [[1, 1, 1, 1, 8], [0, 1, 4, 3]], thread = [0, 0, 0, 8]}>
            },
            qk_attrs = {
              lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 64], [1, 0, 3, 4]], subgroup_basis = [[1, 1, 1, 1, 8], [0, 1, 2, 4]], thread = [0, 0, 8, 0]}>
            }
          },
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> ()>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
          ],
          lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 0, 512, 0], workgroup = [1, 1, 0, 0, 16], reduction = [0, 0, 64, 0, 0]}>
        } ins(%8, %9, %10, %cst : tensor<20x16x63xf16>, tensor<20x4096x63xf16>, tensor<20x4096x64xf16>, f16) outs(%11 : tensor<20x16x64xf16>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<20x16x64xf16>
        iree_codegen.store_to_buffer %12, %7 : tensor<20x16x64xf16> into memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
      iree_codegen.dispatch_config @attention_dispatch_0_attention_20x16x63x4096x64 {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        iree_codegen.yield %x, %y, %z : index, index, index
      }
    }
  }
}

// -----

// Test that the vector distribute pipeline correctly handles attention with
// a K2 dimension (4080) that is not aligned to the tile size (64), requiring
// masking.

// CHECK-LABEL: func.func @attention_dispatch_0_attention_20x16x64x4080x64()
// CHECK:   scf.for
// CHECK:     arith.cmpi slt
// CHECK:     arith.select
// CHECK:   scf.yield
hal.executable private @attention_dispatch_1 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>) {
    hal.executable.export public @attention_dispatch_0_attention_20x16x64x4080x64 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_20x16x64x4080x64() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>>
        %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>>
        %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %8 = iree_codegen.load_from_buffer %1 : memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x16x64xf16>
        %9 = iree_codegen.load_from_buffer %3 : memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4080x64xf16>
        %10 = iree_codegen.load_from_buffer %5 : memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4080x64xf16>
        %11 = tensor.empty() : tensor<20x16x64xf16>
        %12 = iree_linalg_ext.attention {
          decomposition_config = {
            pv_attrs = {
              attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 3, 4]]}>},
            qk_attrs = {
              attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 2, 3]]}>
            }
          },
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> ()>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
          ],
          lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1, 2], reduction = [0, 0, 0, 64, 0], workgroup = [1, 16, 0, 0, 64]}>
        } ins(%8, %9, %10, %cst : tensor<20x16x64xf16>, tensor<20x4080x64xf16>, tensor<20x4080x64xf16>, f16) outs(%11 : tensor<20x16x64xf16>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<20x16x64xf16>
        iree_codegen.store_to_buffer %12, %7 : tensor<20x16x64xf16> into memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
      iree_codegen.dispatch_config @attention_dispatch_0_attention_20x16x64x4080x64 {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        iree_codegen.yield %x, %y, %z : index, index, index
      }
    }
  }
}

// -----

// Test that the vector distribute pipeline correctly handles layernorm with
// a reduction dimension (508) that is not aligned to the reduction tile size
// (512), requiring masking.

// CHECK-LABEL: func.func @layernorm_dispatch_0_reduction_2x508_f16()
// CHECK:   arith.cmpi slt
// CHECK:   arith.select
hal.executable private @layernorm_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>) {
    hal.executable.export public @layernorm_dispatch_0_reduction_2x508_f16 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @layernorm_dispatch_0_reduction_2x508_f16() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
        %cst = arith.constant 0.000000e+00 : f16
        %cst_0 = arith.constant 5.080000e+02 : f16
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x508xf16, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<2x508xf16, #hal.descriptor_type<storage_buffer>> to memref<2x508xf16, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<508xf16, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<508xf16, #hal.descriptor_type<storage_buffer>> to memref<508xf16, #amdgpu.address_space<fat_raw_buffer>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<508xf16, #hal.descriptor_type<storage_buffer>>
        %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<508xf16, #hal.descriptor_type<storage_buffer>> to memref<508xf16, #amdgpu.address_space<fat_raw_buffer>>
        %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<2x508xf16, #hal.descriptor_type<storage_buffer>>
        %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<2x508xf16, #hal.descriptor_type<storage_buffer>> to memref<2x508xf16, #amdgpu.address_space<fat_raw_buffer>>
        %8 = iree_codegen.load_from_buffer %1 : memref<2x508xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2x508xf16>
        %9 = iree_codegen.load_from_buffer %3 : memref<508xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<508xf16>
        %10 = iree_codegen.load_from_buffer %5 : memref<508xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<508xf16>
        %11 = tensor.empty() : tensor<2x508xf16>
        %12 = tensor.empty() : tensor<2xf16>
        %13 = linalg.fill ins(%cst : f16) outs(%12 : tensor<2xf16>) -> tensor<2xf16>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%8 : tensor<2x508xf16>) outs(%13 : tensor<2xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], reduction = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %out: f16):
          %18 = arith.addf %in, %out : f16
          linalg.yield %18 : f16
        } -> tensor<2xf16>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %14 : tensor<2x508xf16>, tensor<2xf16>) outs(%11 : tensor<2x508xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], serial = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %18 = arith.divf %in_2, %cst_0 : f16
          %19 = arith.subf %in, %18 : f16
          linalg.yield %19 : f16
        } -> tensor<2x508xf16>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%15 : tensor<2x508xf16>) outs(%13 : tensor<2xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], reduction = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %out: f16):
          %18 = arith.mulf %in, %in : f16
          %19 = arith.addf %18, %out : f16
          linalg.yield %19 : f16
        } -> tensor<2xf16>
        %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %16, %9, %10 : tensor<2x508xf16>, tensor<2xf16>, tensor<508xf16>, tensor<508xf16>) outs(%11 : tensor<2x508xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], serial = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
          %18 = arith.divf %in_2, %cst_0 : f16
          %19 = arith.truncf %cst_1 : f32 to f16
          %20 = arith.addf %18, %19 : f16
          %21 = math.rsqrt %20 : f16
          %22 = arith.mulf %in, %21 : f16
          %23 = arith.mulf %22, %in_3 : f16
          %24 = arith.addf %23, %in_4 : f16
          linalg.yield %24 : f16
        } -> tensor<2x508xf16>
        iree_codegen.store_to_buffer %17, %7 : tensor<2x508xf16> into memref<2x508xf16, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
      iree_codegen.dispatch_config @layernorm_dispatch_0_reduction_2x508_f16 {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        iree_codegen.yield %x, %y, %z : index, index, index
      }
    }
  }
}
