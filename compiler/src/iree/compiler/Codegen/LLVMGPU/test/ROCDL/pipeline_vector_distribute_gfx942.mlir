// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s --check-prefix=MEMORY

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

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

// Basic pipeline test to make sure it generates the instructions we expect.

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f32()
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) -> (vector<2x2x1x1x4x1xf32>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x1x1x4x1xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

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

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f16()
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x2x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<2x2x1x1x4x1xf16> to vector<2x2x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<2x2x1x1x4x1xf32> to vector<2x2x1x1x4x1xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<2x2x1x1x4x1xf16>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 64, 0], reduction = [0, 0, 0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @expanded_matmul_transpose_b_executable {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @expanded_matmul_transpose_b layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
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
// prefetching, we have one iteration peeled of so upper bound is 2048 - 128 = 1920.
//          CHECK:   scf.for {{.*}} = %c0 to %c1920 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<4x1x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] : vector<4x1x1x1x4x1xf16> to vector<4x1x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
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

// Basic f8, f8 -> f32 matmul. (intrinsic with shape, m = 16, n = 16, k = 32)

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_16x16x32_f8_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_256x256x256_16x16x32_f8_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_16x16x32_f8_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FNUZ>, tensor<256x256xf8E4M3FNUZ>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for f8 inputs.

//    CHECK-LABEL: func.func @matmul_256x256x256_16x16x32_f8_f32()
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic i8, i8 -> i32 matmul.

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_256x256x256_i8_i32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
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
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic f8, f8 -> f32 matmul. (intrinsic with shape, m = 32, n = 32, k = 16)

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x16_F8E4M3FNUZ>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_256x256x256_32x32x16_f8_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_256x256x256_32x32x16_f8_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_32x32x16_f8_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FNUZ>, tensor<256x256xf8E4M3FNUZ>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for f8 inputs.

//    CHECK-LABEL: func.func @matmul_256x256x256_32x32x16_f8_f32()
// Each subgroup handles 1 * 1 tiles, and for each tile we accumulate (256/16) = 16 times
// along the K dimension. So in total 16 mfma ops.
//  CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// Basic i8, i8 -> i32 matmul_transpose_b.

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 256], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_transpose_b_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_transpose_b_256x256x256_i8_i32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
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
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 128, 0, 0, 0], reduction = [0, 0, 0, 0, 1, 1, 32], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @conv_nhwc_dispatch_0 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @conv_nhwc layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
        %5 = tensor.empty() : tensor<2x256x512x256xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @conv_nhwc
//          CHECK:   scf.for {{.*}} = %c0 to %c215 step %c1 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x4x1x1x4x1xf32>)
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield
// CHECK-COUNT-16:   amdgpu.mfma
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} : vector<4x1xf32>, memref<2x256x512x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 64, 1, 64, 0], reduction = [0, 0, 0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
hal.executable public @main_dispatch_expanded_matmul {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @generic_2x1024x20x64x1280_f16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_2x1024x20x64x1280_f16() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%2) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x1024x1280xf16>>
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x64x1280xf16>>
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x1024x20x64xf16>>
        %7 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [2, 1024, 1280], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x1024x1280xf16>> -> tensor<2x1024x1280xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [20, 64, 1280], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x64x1280xf16>> -> tensor<20x64x1280xf16>
        %9 = tensor.empty() : tensor<2x1024x20x64xf16>
        %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<2x1024x20x64xf16>) -> tensor<2x1024x20x64xf16>
        %11 = linalg.generic {
          indexing_maps = [#map, #map1, #map2],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
          lowering_config = #config
        } ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
          outs(%10 : tensor<2x1024x20x64xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %12 = arith.mulf %in, %in_0 : f16
          %13 = arith.addf %out, %12 : f16
          linalg.yield %13 : f16
        } -> tensor<2x1024x20x64xf16>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0, 0, 0], sizes = [2, 1024, 20, 64], strides = [1, 1, 1, 1] : tensor<2x1024x20x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x1024x20x64xf16>>
        return
      }
    }
  }
}


//    CHECK-LABEL: func.func @generic_2x1024x20x64x1280_f16
// This has more than 2 iteartions. So we have prefetching enabled for this case. Due to
// prefetching, we have one iteration peeled of so upper bound is 1280 - 128 = 1152.
//          CHECK:   scf.for {{.*}} = %c0 to %c1152 step %c128 iter_args({{.*}}) -> (vector<2x2x1x1x4x1xf16>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x1x1x4x1xf16>
// CHECK-COUNT-32:   amdgpu.mfma
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} : vector<4x1xf16>

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 16, 16, 0], reduction = [0, 0, 0, 16], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUPadAndVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @unaligned_mk_batch_matmul_64x978x1281x1281_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @unaligned_mk_batch_matmul_64x978x1281x1281_f16_f16 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @unaligned_nk_batch_matmul() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x968x1281xf16>> -> tensor<64x968x1281xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1281, 1281], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1281x1281xf16>> -> tensor<64x1281x1281xf16>
      %5 = tensor.empty() : tensor<64x968x1281xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
      %7 = linalg.batch_matmul {lowering_config = #config} ins(%3, %4 : tensor<64x968x1281xf16>, tensor<64x1281x1281xf16>) outs(%6 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : tensor<64x968x1281xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
      return
    }
  }
}
}
// Basic pipeline test to make sure it generates the instructions we expect.

// CHECK-LABEL: func.func @unaligned_nk_batch_matmul()
// CHECK-DAG:     %[[RHS_SHARED:.+]] = memref.alloc() : memref<1x16x20xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[RHS_SHARED_SUB:.+]] =  memref.subview %[[RHS_SHARED]][0, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK-DAG:     %[[LHS_SHARED:.+]] = memref.alloc() : memref<1x16x20xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[LHS_SHARED_SUB:.+]] =  memref.subview %[[LHS_SHARED]][0, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK-DAG:     %[[LHS_GLOBAL_BIND:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[ASSUMED_LHS_GLOBAL_BIND:.+]] = memref.assume_alignment %[[LHS_GLOBAL_BIND]], 64
// CHECK-DAG:     %[[LHS_GLOBAL:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_LHS_GLOBAL_BIND]] resetOffset : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>> to memref<64x968x1281xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-DAG:     %[[RHS_GLOBAL_BIND:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<64x1281x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[ASSUMED_RHS_GLOBAL_BIND:.+]] = memref.assume_alignment %[[RHS_GLOBAL_BIND]], 64
// CHECK-DAG:     %[[RHS_GLOBAL:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_RHS_GLOBAL_BIND]] resetOffset : memref<64x1281x1281xf16, #hal.descriptor_type<storage_buffer>> to memref<64x1281x1281xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-DAG:     %[[OUT_GLOBAL_BIND:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(64) offset(%c0) : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[ASSUMED_OUT_GLOBAL_BIND:.+]] = memref.assume_alignment %[[OUT_GLOBAL_BIND]], 64
// CHECK-DAG:     %[[OUT_GLOBAL:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_OUT_GLOBAL_BIND]] resetOffset : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>> to memref<64x968x1281xf16, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-DAG:     %[[LHS_GLOBAL_SUB:.+]] = memref.subview %[[LHS_GLOBAL]]
// CHECK-DAG:     %[[LHS_LOAD:.+]] = vector.transfer_read %[[LHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK-DAG:     %[[RHS_GLOBAL_SUB:.+]] = memref.subview %[[RHS_GLOBAL]]
// CHECK-DAG:     %[[RHS_LOAD:.+]] = vector.transfer_read %[[RHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK:         vector.transfer_write %[[LHS_LOAD]], %[[LHS_SHARED]]
// CHECK:         vector.transfer_write %[[RHS_LOAD]], %[[RHS_SHARED]]
// CHECK:         %[[RES:.+]] scf.for {{.*}} = %c0 to %c1280 step %c16 iter_args({{.*}}) -> (vector<1x1x1x1x1x1x1x4x1xf16>)
// CHECK-DAG:       %[[LHS_GLOBAL_SUB:.+]] = memref.subview %[[LHS_GLOBAL]]
// CHECK-DAG:       %[[LHS_LOAD:.+]] = vector.transfer_read %[[LHS_GLOBAL_SUB]]
// CHECK-DAG:       %[[RHS_GLOBAL_SUB:.+]] = memref.subview %[[RHS_GLOBAL]]
// CHECK-DAG:       %[[RHS_LOAD:.+]] = vector.transfer_read %[[RHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK:           gpu.barrier
// CHECK-DAG:       %{{.+}} = vector.transfer_read %[[LHS_SHARED]]
// CHECK-DAG:       %{{.+}} = vector.transfer_read %[[RHS_SHARED]]
// CHECK:           amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:           %[[TRUNCF:.+]] = arith.truncf %{{.+}} : vector<1x1x1x1x1x1x1x4x1xf32> to vector<1x1x1x1x1x1x1x4x1xf16>
// CHECK:           gpu.barrier
// CHECK:           vector.transfer_write %[[LHS_LOAD]], %[[LHS_SHARED]]
// CHECK:           vector.transfer_write %[[RHS_LOAD]], %[[RHS_SHARED]]
// CHECK:           scf.yield %[[TRUNCF]]
// CHECK:         }
// CHECK:         gpu.barrier
// CHECK:         amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:         %[[OUT_GLOBAL_SUB:.+]] = memref.subview %[[OUT_GLOBAL]]
// CHECK:         vector.transfer_write %{{.+}}, %[[OUT_GLOBAL_SUB]]

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 16, 32, 0], reduction = [0, 0, 0, 8], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, subgroup_m_count = 1, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUPadAndVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable public @pad_batch_matmul {
  hal.executable.variant public @rocm_hsaco_fb target(#hal.executable.target<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @pad_batch_matmul ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pad_batch_matmul() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x16x24xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x24x24xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x16x24xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [196, 16, 24], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x16x24xf32>> -> tensor<196x16x24xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [196, 24, 24], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x24x24xf32>> -> tensor<196x24x24xf32>
        %5 = tensor.empty() : tensor<196x16x24xf32>
        %6 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%5 : tensor<196x16x24xf32>) -> tensor<196x16x24xf32>
        %7 = linalg.batch_matmul {lowering_config = #config} ins(%3, %4 : tensor<196x16x24xf32>, tensor<196x24x24xf32>) outs(%6 : tensor<196x16x24xf32>) -> tensor<196x16x24xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [196, 16, 24], strides = [1, 1, 1] : tensor<196x16x24xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x16x24xf32>>
        return
      }
    }
  }
}

// This test checks if we can handle an unaligned batch matmul which has sizes
// smaller than the chosen tile sizes. We just want to make sure we can compile
// this example. We also check if the correct transfer_read/transfer_write are
// produced with in_bounds attrs for the padded dimensions.

// CHECK-LABEL:   @pad_batch_matmul
// CHECK:           scf.for
// LHS
// CHECK:             vector.transfer_read
// CHECK-SAME:        in_bounds = [true, true, true]
// CHECK-SAME:        memref<196x16x24xf32
// CHECK-SAME:        vector<1x1x1xf32>
// RHS
// CHECK:             vector.transfer_read
// CHECK-SAME:        in_bounds = [true, true, false]
// CHECK-SAME:        memref<1x8x24xf32
// CHECK-SAME:        vector<1x1x2xf32>
// CHECK:           scf.yield
// OUTPUT
// CHECK:           vector.transfer_write
// CHECK-SAME:      in_bounds = [true, true, false]
// CHECK-SAME:      vector<1x4x1xf32>
// CHECK-SAME:      memref<1x16x24xf32

// -----

// This test ensures that we are generating contraction schedules does not only work on contraction,
// but also will be compatible with transfer_read layouts anchors.
// Currently the transfer_read layout anchors expects WorkgroupSize % (WgTileSize / numelPerThread) == 0.
// this test ensure that this constraint is satisfied.

// NOTE: This test is not exhaustive of all possible ways the above condition is breaking,
//       but rather is an example of a matmul shape from a model that broke our compilation heuristic.

#config = #iree_gpu.lowering_config<{workgroup = [1, 16, 128, 0], reduction = [0, 0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @contract_schedule_considering_read_layout {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @contract_schedule_considering_read_layout ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @contract_schedule_considering_read_layout() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %3 = arith.index_castui %0 : i32 to index
        %4 = arith.index_castui %1 : i32 to index
        %5 = arith.index_castui %2 : i32 to index
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%3) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x160x1536xf16>>
        %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x1536x1536xf16>>
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x160x1536xf16>>
        %9 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [2, 160, 1536], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x160x1536xf16>> -> tensor<2x160x1536xf16>
        %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [2, 1536, 1536], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x1536x1536xf16>> -> tensor<2x1536x1536xf16>
        %11 = tensor.empty() : tensor<2x160x1536xf16>
        %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<2x160x1536xf16>) -> tensor<2x160x1536xf16>
        %13 = linalg.batch_matmul {lowering_config = #config} ins(%9, %10 : tensor<2x160x1536xf16>, tensor<2x1536x1536xf16>) outs(%12 : tensor<2x160x1536xf16>) -> tensor<2x160x1536xf16>
        iree_tensor_ext.dispatch.tensor.store %13, %8, offsets = [0, 0, 0], sizes = [2, 160, 1536], strides = [1, 1, 1] : tensor<2x160x1536xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x160x1536xf16>>
        return
      }
    }
  }
}
// Basic pipeline test to make sure it generates the instructions we expect.

// CHECK-LABEL: func.func @contract_schedule_considering_read_layout()
// CHECK-DAG:     %[[RHS_SHARED:.+]] = memref.alloc() : memref<128x132xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[RHS_SHARED_SUB:.+]] =  memref.subview %[[RHS_SHARED]][0, 0] [128, 128] [1, 1]
// CHECK-DAG:     %[[LHS_SHARED:.+]] = memref.alloc() : memref<16x132xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[LHS_SHARED_SUB:.+]] =  memref.subview %[[LHS_SHARED]][0, 0] [16, 128] [1, 1]
// CHECK:   scf.for {{.*}} = %c0 to %c1408 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<1x2x1x1x4x1xf16>)
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:     scf.yield
// CHECK-COUNT-16:   amdgpu.mfma

// -----

// This test ensures that we can generate and decompose the right instructions from V(Virtual) MFMAs.
// (intrinsic with shape, m = 32, n = 32, k = 16)

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32() attributes {translation_info = #translation} {
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

// CHECK-LABEL: func @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32
// CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<1x1x4x1x4x1xf32>)

// Validate that VMFMA is decomposed into coalesced read and 2 MFMAs:

// CHECK:       %[[A_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x8xf16> to vector<8xf16>
// CHECK:       %[[B_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x8x1xf16> to vector<8xf16>
// CHECK:       %[[C_CAST:.+]] = vector.shape_cast %{{.+}} : vector<4x1x4x1xf32> to vector<16xf32>
// CHECK:       %[[A_SLICE_0:.+]] = vector.extract_strided_slice %[[A_CAST]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[B_SLICE_0:.+]] = vector.extract_strided_slice %[[B_CAST]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[MFMA_0:.*]] = amdgpu.mfma %[[A_SLICE_0]] * %[[B_SLICE_0]] + %[[C_CAST]]
// CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[A_SLICE_1:.+]] = vector.extract_strided_slice %[[A_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[B_SLICE_1:.+]] = vector.extract_strided_slice %[[B_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[MFMA_1:.+]] = amdgpu.mfma %[[A_SLICE_1]] * %[[B_SLICE_1]] + %[[MFMA_0]]
// CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>

// Ensure right number of instructions are being generated.

// CHECK-COUNT-14: vector.extract_strided_slice
// CHECK-NEXT: amdgpu.mfma
// CHECK:     scf.yield

// -----

// This test ensures we can generate correct instructions from V(Virtual) MFMAs that has only different read layouts.
// (intrinsic with shape m = 16, n = 16, k = 32)

#config = #iree_gpu.lowering_config<{workgroup = [32, 32, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F8E4M3FNUZ>, subgroup_m_count = 1, subgroup_n_count = 1}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @virtual_intrinsic_256x256x256_32x32x16_f8E4M3FNUZ_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @virtual_intrinsic_256x256x256_32x32x16_f8E4M3FNUZ_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @virtual_intrinsic_256x256x256_32x32x16_f8E4M3FNUZ_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FNUZ>, tensor<256x256xf8E4M3FNUZ>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Basic pipeline test to make sure it generates the instructions we expect.

//    CHECK-LABEL: func.func @virtual_intrinsic_256x256x256_32x32x16_f8E4M3FNUZ_f32()
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) ->  (vector<1x1x4x1x4x1xf32>)
// Each subgroup handles 1 * 1 tiles, and for each tile we accumulate (128 / 16) = 8 times
// along the K dimension. So in total 8 mfma ops.
// CHECK-COUNT-8:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
//          CHECK:     scf.yield %{{.+}} : vector<1x1x4x1x4x1xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #amdgpu.address_space<fat_raw_buffer>>

// -----

// This test ensures we can generate correct instructions from V(Virtual) MFMAs that has only different read layouts.

#config = #iree_gpu.lowering_config<{workgroup = [32, 32, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_16x16x32_F8E4M3FNUZ>, subgroup_m_count = 2, subgroup_n_count = 2}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf8E4M3FNUZ>> -> tensor<256x256xf8E4M3FNUZ>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf8E4M3FNUZ>, tensor<256x256xf8E4M3FNUZ>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @virtual_intrinsic_256x256x256_16x16x32xf8E4M3FNUZ_f32
// CHECK-DAG:     %[[ALLOC_LHS:.+]] = memref.alloc() : memref<32x136xf8E4M3FNUZ, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[ALLOC_RHS:.+]] = memref.alloc() : memref<128x40xf8E4M3FNUZ, #gpu.address_space<workgroup>>
// CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<1x1x1x1x4x1xf32>)

// Validate that VMFMA do 2 interleaved reads, combine them for every MFMA:

// CHECK-COUNT-6: vector.transfer_read %[[ALLOC_LHS]]
// CHECK:       %[[SLICE_LHS_0:.+]] = vector.transfer_read %[[ALLOC_LHS]]
// CHECK:       %[[VECTOR_LHS_0:.+]] = vector.insert_strided_slice %[[SLICE_LHS_0]], %{{.*}}
// CHECK:       %[[SLICE_LHS_1:.+]] = vector.transfer_read %[[ALLOC_LHS]]
// CHECK:       %[[VECTOR_LHS_1:.+]] = vector.insert_strided_slice %[[SLICE_LHS_1]], %[[VECTOR_LHS_0]] {{.*}} : vector<1x4xf8E4M3FNUZ> into vector<1x4x1x2x1x4xf8E4M3FNUZ>

// CHECK-COUNT-6: vector.transfer_read %[[ALLOC_RHS]]
// CHECK:       %[[SLICE_RHS_0:.+]] = vector.transfer_read %[[ALLOC_RHS]]
// CHECK:       %[[VECTOR_RHS_0:.+]] = vector.insert_strided_slice %[[SLICE_RHS_0]], %{{.*}}
// CHECK:       %[[SLICE_RHS_1:.+]] = vector.transfer_read %[[ALLOC_RHS]]
// CHECK:       %[[VECTOR_RHS_1:.+]] = vector.insert_strided_slice %[[SLICE_RHS_1]], %[[VECTOR_RHS_0]] {{.*}} : vector<4x1xf8E4M3FNUZ> into vector<4x1x2x1x4x1xf8E4M3FNUZ>

// CHECK:       %[[EXTRACT_LHS:.+]] = vector.extract %[[VECTOR_LHS_1]][{{.*}}, {{.*}}] : vector<1x2x1x4xf8E4M3FNUZ> from vector<1x4x1x2x1x4xf8E4M3FNUZ>
// CHECK:       %[[EXTRACT_RHS:.+]] = vector.extract %[[VECTOR_RHS_1]][{{.*}}, {{.*}}] : vector<2x1x4x1xf8E4M3FNUZ> from vector<4x1x2x1x4x1xf8E4M3FNUZ>

// CHECK:       %[[LHS_CAST:.+]] = vector.shape_cast %[[EXTRACT_LHS]] : vector<1x2x1x4xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[RHS_CAST:.+]] = vector.shape_cast %[[EXTRACT_RHS]] : vector<2x1x4x1xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       amdgpu.mfma %[[LHS_CAST]] * %[[RHS_CAST]] + %{{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32}
// CHECK-SAME:     : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>

// Ensure right number of instructions are being generated.
// CHECK-COUNT-3: amdgpu.mfma

// CHECK:     scf.yield

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>,
                                                    #hal.pipeline.binding<storage_buffer>,
                                                    #hal.pipeline.binding<storage_buffer>,
                                                    #hal.pipeline.binding<storage_buffer>,
                                                    #hal.pipeline.binding<storage_buffer>]>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64, {}>
#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 64], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [64, 128, 0]}>

hal.executable public @matmul_gather_rhs {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_gather_rhs ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gather_rhs() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xi64>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf16>> -> tensor<4096x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xi64>> -> tensor<4096x64xi64>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf16>> -> tensor<4096x64xf16>
        %7 = tensor.empty() : tensor<4096x4096xf16>
        %8 = tensor.empty() : tensor<4096x4096xf32>
        %9 = tensor.empty() : tensor<4096x64xf16>
        %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<4096x64xi64>) outs(%9 : tensor<4096x64xf16>) {
        ^bb0(%in: i64, %out: f16):
          %14 = linalg.index 0 : index
          %15 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %4[%14, %15] : tensor<4096x64xf16>
          linalg.yield %extracted : f16
        } -> tensor<4096x64xf16>
        %11 = linalg.fill ins(%cst : f32) outs(%8 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
        %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3],
                              iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%6, %10 : tensor<4096x64xf16>, tensor<4096x64xf16>)
              outs(%11 : tensor<4096x4096xf32>)
              attrs = {lowering_config = #config} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %14 = arith.extf %in : f16 to f32
          %15 = arith.extf %in_0 : f16 to f32
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        } -> tensor<4096x4096xf32>
        %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<4096x4096xf32>) outs(%7 : tensor<4096x4096xf16>) {
        ^bb0(%in: f32, %out: f16):
          %14 = arith.truncf %in : f32 to f16
          linalg.yield %14 : f16
        } -> tensor<4096x4096xf16>
        iree_tensor_ext.dispatch.tensor.store %13, %3, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_gather_rhs
// CHECK: vector.gather
// CHECK-COUNT-32: amdgpu.mfma

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 64, 0, 0, 64], reduction = [0, 0, 0, 64, 0], promote_operands = [0, 1, 2]}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_20x4096x64x4096x64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x4096x64x4096x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
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
                      qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 1, promote_operands = [0, 1]}>},
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

// CHECK: scf.for %{{.*}} = %c0 to %c4096 step %c64
// CHECK-SAME: -> (vector<2x1x1xf32>, vector<2x1x1xf32>, vector<2x4x1x1x1x4xf32>)
// CHECK-COUNT-48:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @attention_20x4096x64x4096x64()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 64], reduction = [0, 0, 0, 0, 64, 0], promote_operands = [0, 1, 2]}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_multiple_m_transpose {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_multiple_m_transpose ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_multiple_m_transpose() attributes {translation_info = #translation} {
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
                                                          qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 1, promote_operands = [0, 1]}>},
                                                          pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 1, promote_operands = [1]}>}
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

// CHECK-LABEL: func.func @attention_multiple_m_transpose()
// CHECK: scf.for %{{.*}} = %c0 to %c4608 step %c64
// CHECK-SAME: -> (vector<2x1x1xf32>, vector<2x1x1xf32>, vector<2x4x1x1x1x4xf32>)
// CHECK-COUNT-96:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @attention_multiple_m_transpose()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 64], reduction = [0, 0, 0, 0, 32, 0], promote_operands = [0, 1, 2]}>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_mfma_32x32x8 {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_mfma_32x32x8 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_mfma_32x32x8() attributes {translation_info = #translation} {
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
                                                          qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [0, 1]}>},
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

// CHECK-LABEL: func.func @attention_mfma_32x32x8()
// CHECK: scf.for %{{.*}} = %c0 to %c4608 step %c32
// CHECK-SAME: -> (vector<1x1x1xf32>, vector<1x1x1xf32>, vector<1x2x1x4x1x4xf32>)
// CHECK-COUNT-24:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @attention_mfma_32x32x8()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

!Q = tensor<1x16x64xf16>
!K_SK     = tensor<1x4x256x64xf16>
!V_SK     = tensor<1x4x256x64xf16>
!O_SK     = tensor<1x4x16x64xf32>
!ROWRED_SK= tensor<1x4x16xf32>

#config = #iree_gpu.lowering_config<{ workgroup = [1, 1, 16, 0, 0, 0], reduction = [0, 0, 0, 0, 0, 32],promote_operands = [0, 1, 2] }>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64] subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @online_attention_split_k2 {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @online_attention_split_k2 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @online_attention_split_k2() attributes {translation_info = #translation} {
        %cst = arith.constant 1.0 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:!Q>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:!K_SK>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:!V_SK>
        %out_arg = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:!O_SK>
        %max_arg = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:!ROWRED_SK>
        %sum_arg = hal.interface.binding.subspan layout(#pipeline_layout) binding(5) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:!ROWRED_SK>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 16, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:!Q> -> !Q
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 4, 256, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:!K_SK> -> !K_SK
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 256, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:!V_SK> -> !V_SK
        %empty_o = tensor.empty() : !O_SK
        %empty_rowmax = tensor.empty() : !ROWRED_SK
        %empty_rowsum = tensor.empty() : !ROWRED_SK
        %out:3 = iree_linalg_ext.online_attention {indexing_maps = [affine_map<(b1, b2, m, n, k1, k2) -> (b1, m, k1)>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, k2, k1)>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, k2, n)>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> ()>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, m, n)>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, m)>,
                                                                    affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, m)>],
                                                                  lowering_config = #config,
                                                                  decomposition_config = {
                                                                    qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [0, 1]}>},
                                                                    pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [1]}>}
                                                                  }}
        ins(%4, %5, %6, %cst : !Q, !K_SK, !V_SK, f16) outs(%empty_o, %empty_rowmax, %empty_rowsum: !O_SK, !ROWRED_SK, !ROWRED_SK) {
              ^bb0(%score: f32):
                iree_linalg_ext.yield %score : f32
             } -> !O_SK, !ROWRED_SK, !ROWRED_SK
        iree_tensor_ext.dispatch.tensor.store %out#0, %out_arg, offsets = [0, 0, 0, 0], sizes = [1, 4, 16, 64], strides = [1, 1, 1, 1] : !O_SK -> !iree_tensor_ext.dispatch.tensor<writeonly:!O_SK>
        iree_tensor_ext.dispatch.tensor.store %out#1, %max_arg, offsets = [0, 0, 0], sizes = [1, 4, 16], strides = [1, 1, 1] : !ROWRED_SK -> !iree_tensor_ext.dispatch.tensor<writeonly:!ROWRED_SK>
        iree_tensor_ext.dispatch.tensor.store %out#2, %sum_arg, offsets = [0, 0, 0], sizes = [1, 4, 16], strides = [1, 1, 1] : !ROWRED_SK -> !iree_tensor_ext.dispatch.tensor<writeonly:!ROWRED_SK>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @online_attention_split_k2()
// CHECK: scf.for %{{.*}} = %c0 to %c256 step %c32
// CHECK-SAME: -> (vector<1x1x1xf32>, vector<1x1x1xf32>, vector<1x4x1x1x1x4xf32>)
// CHECK-COUNT-16:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK: scf.yield

// Check that we only use alloc for Q, K, and V. No shared memory for S is
// needed because the intrinsic layout mathes.
// MEMORY-LABEL: func.func @online_attention_split_k2()
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT: memref.alloc

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 64, {}>

#qk_config = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64}>}
#pv_config = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [1], subgroup_m_count = 2 : i64, subgroup_n_count = 1 : i64}>}
#config = #iree_gpu.lowering_config<{promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 0, 64], workgroup = [1, 1, 64, 64, 0, 0]}>

module {
  hal.executable public @attention_gather_k {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @attention_gather_k ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @attention_gather_k() attributes {translation_info = #translation} {
          %cst = arith.constant 1.250000e-01 : f16
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xi64>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>>
          %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>>
          %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x4096x64xf16>>
          %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>> -> tensor<2x10x4096x64xf16>
          %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xi64>> -> tensor<2x10x4096x64xi64>
          %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>> -> tensor<2x10x4096x64xf16>
          %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>> -> tensor<2x10x4096x64xf16>
          %9 = tensor.empty() : tensor<2x10x4096x64xf16>
          %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<2x10x4096x64xi64>) outs(%9 : tensor<2x10x4096x64xf16>) {
          ^bb0(%in: i64, %out: f16):
            %12 = linalg.index 0 : index
            %13 = linalg.index 1 : index
            %14 = arith.index_cast %in : i64 to index
            %15 = linalg.index 3 : index
            %extracted = tensor.extract %5[%12, %13, %14, %15] : tensor<2x10x4096x64xf16>
            linalg.yield %extracted : f16
          } -> tensor<2x10x4096x64xf16>
          %11 = iree_linalg_ext.attention {
              indexing_maps = [#map1, #map2, #map3, #map4, #map5],
              decomposition_config = { qk_attrs = #qk_config, pv_attrs = #pv_config },
              lowering_config = #config} ins(%7, %10, %8, %cst : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, f16) outs(%9 : tensor<2x10x4096x64xf16>) {
          ^bb0(%arg0: f32):
            iree_linalg_ext.yield %arg0 : f32
          } -> tensor<2x10x4096x64xf16>
          iree_tensor_ext.dispatch.tensor.store %11, %4, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : tensor<2x10x4096x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x10x4096x64xf16>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @attention_gather_k
// CHECK: scf.for %{{.*}} = %c0 to %c4096 step %c64
// CHECK:      vector.gather
// CHECK-SAME: into vector<4x1x1x1x1x8xf16>
// CHECK: scf.yield

// MEMORY-LABEL: func.func @attention_gather_k
// MEMORY-COUNT-3: memref.alloc
// MEMORY-NOT:     memref.alloc

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
