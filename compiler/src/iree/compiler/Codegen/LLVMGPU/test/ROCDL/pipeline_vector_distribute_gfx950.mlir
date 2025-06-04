// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s --check-prefix=MEMORY

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x512_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x512_f16_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x512_f16_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x512xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x512xf16>> -> tensor<256x512xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf16>> -> tensor<512x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x512xf16>, tensor<512x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Basic pipeline test to make sure it generates the instructions we expect.

//    CHECK-LABEL: func.func @matmul_256x256x512_f16_f32()
//          CHECK:   scf.for {{.*}} = %c0 to %c512 step %c256 iter_args({{.*}}) -> (vector<2x2x1x1x4x1xf32>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x1x1x4x1xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x512_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x512_f16_f16 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x512_f16_f16() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x512xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x512xf16>> -> tensor<256x512xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf16>> -> tensor<512x256xf16>
      %5 = tensor.empty() : tensor<256x256xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x512xf16>, tensor<512x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
      return
    }
  }
}
}

//    CHECK-LABEL: func.func @matmul_256x256x512_f16_f16()
//          CHECK:   scf.for {{.*}} = %c0 to %c512 step %c256 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x2x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<2x2x1x1x4x1xf16> to vector<2x2x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<2x2x1x1x4x1xf32> to vector<2x2x1x1x4x1xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<2x2x1x1x4x1xf16>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 64, 0], reduction = [0, 0, 0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, subgroup_m_count = 1, subgroup_n_count = 4}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @expanded_matmul_transpose_b_executable {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @expanded_matmul_transpose_b layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @expanded_matmul_transpose_b() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 2048], strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64x2048xf16>> -> tensor<2x64x2048xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 64, 2048], strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x2048xf16>> -> tensor<10x64x2048xf16>

        %5 = tensor.empty() : tensor<2x10x64x64xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
        %7 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
          lowering_config = #config
        } ins(%3, %4 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
        ^bb0(%lhs: f16, %rhs: f16, %out: f16):
          %mul = arith.mulf %lhs, %rhs : f16
          %add = arith.addf %mul, %out : f16
          linalg.yield %add : f16
        } -> tensor<2x10x64x64xf16>

        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1]
          : tensor<2x10x64x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        return
      }
    }
  }
}

//          CHECK: func @expanded_matmul_transpose_b
// This has more than 2 iteartions. So we have prefetching enabled for this case. Due to
// prefetching, we have one iteration peeled of so upper bound is 2048 - 256 = 1792.
//          CHECK:   scf.for {{.*}} = %c0 to %c1792 step %c256 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<4x1x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<4x1x1x1x4x1xf16> to vector<4x1x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<4x1x1x1x4x1xf32> to vector<4x1x1x1x4x1xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<4x1x1x1x4x1xf16>
// CHECK-COUNT-32:   amdgpu.mfma
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<2x10x64x64xf16, #amdgpu.address_space<fat_raw_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable @matmul_multiple_k {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_multiple_k layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_multiple_k() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x64x2048xf16>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128x64x2048xf16>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 64, 2048], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x64x2048xf16>> -> tensor<2x128x64x2048xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [10, 128, 64, 2048], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128x64x2048xf16>> -> tensor<10x128x64x2048xf16>
        %5 = tensor.empty() : tensor<2x10x64x64xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : tensor<2x128x64x2048xf16>, tensor<10x128x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 0, 1, 128], workgroup = [1, 1, 64, 64, 0, 0], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4}>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %8, %out : f16
          linalg.yield %9 : f16
        } -> tensor<2x10x64x64xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1] : tensor<2x10x64x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        return
      }
    }
  }
}

// Check if we can handle multiple reduction dimensions and that they generate
// one coalesced loop.

// CHECK-LABEL: func.func @matmul_multiple_k
// CHECK:          scf.for %[[IV:.+]] = %c0 to %c2048 step %c1
// CHECK:            affine.delinearize_index %[[IV]] into (128, 16)
// CHECK-COUNT-32:   amdgpu.mfma
// CHECK:            scf.yield
// CHECK-COUNT-4:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<2x10x64x64xf16, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic f8, f8 -> f32 matmul. (intrinsic with shape, m = 16, n = 16, k = 128)

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 1024], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x128_F8E4M3FN>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_16x16x128_f8_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_16x16x32_f8_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_16x16x128_f8_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FN>, tensor<256x256xf8E4M3FN>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for f8 inputs.

//    CHECK-LABEL: func.func @matmul_256x256x256_16x16x128_f8_f32()
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 2 times
// along the K dimension. So in total 8 mfma ops.
// CHECK-COUNT-8:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<4xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic i8, i8 -> i32 matmul.

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 512], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x64_I8>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_i8_i32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_i8_i32() attributes {translation_info = #translation} {
      %cst = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xi32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %5 = tensor.empty() : tensor<256x256xi32>
      %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<256x256xi32>) -> tensor<256x256xi32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xi32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for integer inputs.

//    CHECK-LABEL: func.func @matmul_256x256x256_i8_i32()
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 4 times
// along the K dimension. So in total 16 mfma ops.
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 64 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<16xi8>, vector<16xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic f8, f8 -> f32 matmul. (intrinsic with shape, m = 32, n = 32, k = 64)

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 1024], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x64_F8E4M3FN>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_32x32x64_f8_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_32x32x64_f8_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_32x32x64_f8_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FN>> -> tensor<256x256xf8E4M3FN>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FN>, tensor<256x256xf8E4M3FN>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for f8 inputs.

//    CHECK-LABEL: func.func @matmul_256x256x256_32x32x64_f8_f32()
// Each subgroup handles 1 * 1 tiles, and for each tile we accumulate (256/64) = 4 times
// along the K dimension. So in total 4 mfma ops.
//  CHECK-COUNT-4:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<16xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic i8, i8 -> i32 matmul_transpose_b.

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 512], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x64_I8>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_transpose_b_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_transpose_b_256x256x256_i8_i32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_transpose_b_256x256x256_i8_i32() attributes {translation_info = #translation} {
      %cst = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xi32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %5 = tensor.empty() : tensor<256x256xi32>
      %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %7 = linalg.matmul_transpose_b {lowering_config = #config} ins(%3, %4 : tensor<256x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<256x256xi32>) -> tensor<256x256xi32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xi32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for integer inputs.

//    CHECK-LABEL: func.func @matmul_transpose_b_256x256x256_i8_i32()
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 4 times
// along the K dimension. So in total 16 mfma ops.
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 64 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<16xi8>, vector<16xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 64, 0, 0, 64], reduction = [0, 0, 0, 128, 0], promote_operands = [0, 1, 2]}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_20x4096x64x4096x64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x4096x64x4096x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index){
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x4096x64x4096x64() attributes {translation_info = #translation} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x4096x64xf16>
        %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>],
                     lowering_config = #config,
                     decomposition_config = {
                      qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, subgroup_m_count = 2, subgroup_n_count = 1, promote_operands = [0, 1]}>},
                      pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 1, promote_operands = [1]}>}
                     }}
                     ins(%4, %5, %6, %cst : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x4096x64xf16>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                     } -> tensor<20x4096x64xf16>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        return
      }
    }
  }
}

// Basic test to make sure we can handle attention

// CHECK-LABEL: func.func @attention_20x4096x64x4096x64()

// Make sure the Q matmul global read, shared memory write and shared memory
// read is hoisted out.
// CHECK: transfer_read
// CHECK: transfer_write
// CHECK: gpu.barrier
// CHECK: transfer_read

// CHECK: scf.for %{{.*}} = %c0 to %c4096 step %c128
// CHECK-SAME: -> (vector<2x1x1xf32>, vector<2x1x1xf32>, vector<2x4x1x1x1x4xf32>)
// CHECK-COUNT-32:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<4xf32>
// CHECK-COUNT-16:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @attention_20x4096x64x4096x64()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 64], reduction = [0, 0, 0, 0, 64, 0], promote_operands = [0, 1, 2]}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_mfma_32x32x16 {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_mfma_32x32x16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_mfma_32x32x16() attributes {translation_info = #translation} {
        %cst = arith.constant 1.0 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x64x4608x128xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4608x128xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4608x128xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x4608x24x128xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [24, 64, 4608, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x64x4608x128xf16>> -> tensor<24x64x4608x128xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [24, 4608, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4608x128xf16>> -> tensor<24x4608x128xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [24, 4608, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4608x128xf16>> -> tensor<24x4608x128xf16>
        %7 = tensor.empty() : tensor<64x4608x24x128xf16>
        %8 = tensor.empty() : tensor<24x64x4608x128xf16>
        %9 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
                                                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d3)>,
                                                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5)>,
                                                         affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                                                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>],
                                                         lowering_config = #config,
                                                         decomposition_config = {
                                                          qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x16_F16>, subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [0, 1]}>},
                                                          pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1]}>}
                                                         }}
        ins(%4, %5, %6, %cst : tensor<24x64x4608x128xf16>, tensor<24x4608x128xf16>, tensor<24x4608x128xf16>, f16) outs(%8 : tensor<24x64x4608x128xf16>) {
              ^bb0(%score: f32):
                iree_linalg_ext.yield %score : f32
             } -> tensor<24x64x4608x128xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<24x64x4608x128xf16>) outs(%7 : tensor<64x4608x24x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<64x4608x24x128xf16>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0, 0, 0], sizes = [64, 4608, 24, 128], strides = [1, 1, 1, 1] : tensor<64x4608x24x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x4608x24x128xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @attention_mfma_32x32x16()
// CHECK: scf.for %{{.*}} = %c0 to %c4608 step %c64
// CHECK-SAME: -> (vector<1x1x1xf32>, vector<1x1x1xf32>, vector<1x2x1x4x1x4xf32>)
// CHECK-COUNT-16:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
// CHECK-COUNT-8:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @attention_mfma_32x32x16()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

hal.executable private @matvec_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
        %5 = tensor.empty() : tensor<32000x2xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32000x2xf32>) -> tensor<32000x2xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<32000x4096xf16>, tensor<2x4096xf16>) outs(%6 : tensor<32000x2xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 512], subgroup_basis = [[1, 1, 2], [0, 1, 2]], thread = [0, 0, 4], thread_basis = [[1, 1, 64], [0, 1, 2]], workgroup = [16, 1, 0]}>} {
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
//   MEMORY-LABEL: func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32
//    CHECK-LABEL: func.func @matvec_dispatch_0_matmul_transpose_b_32000x2x4096_f16xf16xf32
//          CHECK:   scf.forall ({{.*}}) = (0, 0) to (32000, 2) step (16, 1)
// CHECK-COUNT-16:     gpu.subgroup_reduce  add {{.*}} cluster(size = 64) : (f32) -> f32
// CHECK-COUNT-16:     gpu.subgroup_reduce  add {{.*}} cluster(size = 2) : (f32) -> f32
