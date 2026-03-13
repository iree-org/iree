// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target{for-rocdl=true})))))" %s | FileCheck %s

// Test: Scaled matmul (f4E2M1FN * f4E2M1FN with f8E8M0FNU scales) compiles
// through the full pipeline with DMA + XOR swizzle config. This validates
// that the pipeline handles DMA with inverse source swizzle for data
// operands, including sub-byte type narrow type emulation.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [512, 1, 1]
  subgroup_size = 64,
  {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_num_stages = 2,
      no_reduce_shared_memory_bank_conflicts = true>
  }
>
#config = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.scaled_mma_layout<
    intrinsic = MFMA_SCALE_F32_16x16x128_B32,
    lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN,
    acc_elem_type = f32>,
  promote_operands = [0, 1, 2, 3],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<256, 32>>,
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<256, 32>>,
    #iree_gpu.derived_thread_config,
    #iree_gpu.derived_thread_config],
  reduction = [0, 0, 1, 1],
  subgroup = [4, 8, 0, 0],
  workgroup = [256, 256, 0, 0]
}>
#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @scaled_matmul_dma ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scaled_matmul_dma()
        attributes {translation_info = #translation_info} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x32xf4E2M1FN>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x32xf4E2M1FN>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        %A = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x32xf4E2M1FN>> -> tensor<1024x512x32xf4E2M1FN>
        %B = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1024, 512, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x32xf4E2M1FN>> -> tensor<1024x512x32xf4E2M1FN>
        %A_scales = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>> -> tensor<1024x512xf8E8M0FNU>
        %B_scales = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf8E8M0FNU>> -> tensor<1024x512xf8E8M0FNU>
        %empty = tensor.empty() : tensor<1024x1024xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %result = linalg.generic {
          indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
          iterator_types = ["parallel", "parallel", "reduction", "reduction"]
        } ins(%A, %B, %A_scales, %B_scales : tensor<1024x512x32xf4E2M1FN>, tensor<1024x512x32xf4E2M1FN>, tensor<1024x512xf8E8M0FNU>, tensor<1024x512xf8E8M0FNU>) outs(%fill : tensor<1024x1024xf32>) attrs = {lowering_config = #config} {
        ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
          %s1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
          %s2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
          %m = arith.mulf %s1, %s2 : f32
          %r = arith.addf %out, %m : f32
          linalg.yield %r : f32
        } -> tensor<1024x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %4, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        return
      }
    }
  }
}

// Verify pipeline completes and produces:
//   - gather_to_lds for LHS/RHS data operands (f4E2M1FN) with XOR swizzle
//     applied to source indices (inverse swizzle on read side)
//   - scaled MFMA 16x16x128 compute ops
//   - no linalg.copy remaining for the data operands

// CHECK-LABEL: func.func @scaled_matmul_dma
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf8E8M0FNU, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf4E2M1FN, #gpu.address_space<workgroup>>
// Verify XOR swizzle applied to source: arith.xori emitted in pipelined prologue,
// before gather_to_lds (swizzled indices are hoisted by the software pipeline).
//       CHECK:   arith.xori
// Verify DMA path: LDS store via gather_to_lds (uses swizzled indices computed above)
//       CHECK:   amdgpu.gather_to_lds
// Verify scaled MFMA compute ops are generated
//       CHECK:   amdgpu.scaled_mfma 16x16x128
// Verify no plain linalg.copy remains for data operands
//   CHECK-NOT:   linalg.copy
