// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 2, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_256x256x256_f16_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f32
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) -> (vector<2x2x8x1x1x1xf32>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 wmma ops.
// CHECK-COUNT-32:     amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x8x1x1x1xf32>
//  Since each subgroup handles 2 * 2 tiles, and for each tile, each lane holds 4 values.
//  we will have 32 writes. We cannot do contiguous writes since the outputs columns has interleaved
//  thread ids.
//  CHECK-COUNT-32:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<1x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], mma_kind = #iree_gpu.mma_layout<WMMAR3_F16_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 2, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_256x256x256_f16_f16 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f16() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
      return
    }
  }
}
}

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f16
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) -> (vector<2x2x16x1x1x1xf16>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 wmma ops.
// CHECK-COUNT-32:     amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<16xf16>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x16x1x1x1xf16>
//  Since each subgroup handles 2 * 2 tiles, and for each tile, each lane holds 4 values.
//  we will have 32 writes. We cannot do contiguous writes since the outputs columns has interleaved
//  thread ids.
//  CHECK-COUNT-32:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<1x1xf16>, memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>>

// -----

hal.executable private @matvec_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
        %5 = tensor.empty() : tensor<32000x2xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32000x2xf32>) -> tensor<32000x2xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<32000x4096xf16>, tensor<2x4096xf16>) outs(%6 : tensor<32000x2xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 512], subgroup_basis = [[1, 1, 4], [0, 1, 2]], thread = [0, 0, 4], thread_basis = [[1, 1, 32], [0, 1, 2]], workgroup = [16, 1, 0]}>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<32000x2xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 2], strides = [1, 1] : tensor<32000x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf32>>
        return
      }
    }
  }
}
//    CHECK-LABEL: func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32
//          CHECK:   scf.forall ({{.*}}) = (0, 0) to (32000, 2) step (16, 1)
// CHECK-COUNT-16:     gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32
// CHECK-COUNT-16:     gpu.subgroup_reduce  add {{.*}} cluster(size = 4) : (f32) -> f32
