// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 4],
  thread = [8, 4],
  promote_operands = [0, 1]
}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// Note that current barrier placement logic is observedly poor. Some cleanup
// analysis should be able to simplify the below to just two barriers.

// CHECK-LABEL: func @matmul_transpose_b
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   memref.alloc() : memref<64x8xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x8xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c1280 step %c4 {{.*}} -> (vector<8x4xf32>)
//       CHECK:       gpu.barrier
//   CHECK-DAG:       %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<2xf16>
//   CHECK-DAG:       vector.transfer_write %[[LHS_RD]], %[[LHS_ALLOC:[A-Za-z0-9]+]]
//   CHECK-DAG:       %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<2xf16>
//   CHECK-DAG:       vector.transfer_write %[[RHS_RD]], %[[RHS_ALLOC:[A-Za-z0-9]+]]
//       CHECK:       gpu.barrier
//   CHECK-DAG:       %[[LHS_MM:.+]] = vector.transfer_read %[[LHS_ALLOC]]{{.*}} vector<8x4xf16>
//   CHECK-DAG:       %[[RHS_MM:.+]] = vector.transfer_read %[[RHS_ALLOC]]{{.*}} vector<4x4xf16>
//       CHECK:       %[[MM:.+]] = vector.contract {{.*}} %[[LHS_MM]], %[[RHS_MM]]
//       CHECK:       scf.yield %[[MM]]
//       CHECK:     vector.transfer_write %[[LOOP]], %[[B2]]
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [4, 4, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_mfma ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_mfma()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_mfma
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
//       CHECK:       gpu.barrier
//   CHECK-DAG:       %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<8xf16>
//   CHECK-DAG:       vector.transfer_write %[[LHS_RD]]
//   CHECK-DAG:       %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<8xf16>
//   CHECK-DAG:       vector.transfer_write %[[RHS_RD]]
//       CHECK:       gpu.barrier
//   CHECK-DAG:       vector.transfer_read {{.*}} #gpu.address_space<workgroup>>, vector<2x1x2x4xf16>
//   CHECK-DAG:       vector.transfer_read {{.*}} #gpu.address_space<workgroup>>, vector<2x1x2x4xf16>
//   CHECK-DAG:       vector.transpose %{{.*}}, [0, 2, 1, 3] : vector<2x1x2x4xf16>
//   CHECK-DAG:       vector.transpose %{{.*}}, [0, 2, 1, 3] : vector<2x1x2x4xf16>
// CHECK-COUNT-4:     amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:       scf.yield
//       CHECK:     %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 2, 1, 3] : vector<2x2x4x1xf32> to vector<2x4x2x1xf32>
//       CHECK:     vector.transfer_write %[[LOOP_T]], %[[B2]]
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [4, 4, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_wmma ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_wmma()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_wmma
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x8x1x1xf32>)
//       CHECK:       gpu.barrier
//   CHECK-DAG:       vector.transfer_read %[[B0]]{{.*}} vector<8xf16>
//   CHECK-DAG:       vector.transfer_read %[[B0]]{{.*}} vector<8xf16>
//   CHECK-DAG:       vector.transfer_read %[[B1]]{{.*}} vector<8xf16>
//   CHECK-DAG:       vector.transfer_read %[[B1]]{{.*}} vector<8xf16>
//       CHECK:       gpu.barrier
//   CHECK-DAG:       vector.transfer_read {{.*}} vector<2x1x2x16xf16>
//   CHECK-DAG:       vector.transfer_read {{.*}} vector<2x1x2x16xf16>
//   CHECK-DAG:       vector.transpose %{{.*}}, [0, 2, 1, 3] : vector<2x1x2x16xf16>
//   CHECK-DAG:       vector.transpose %{{.*}}, [0, 2, 1, 3] : vector<2x1x2x16xf16>
// CHECK-COUNT-8:     amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//       CHECK:       scf.yield
//       CHECK:     %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 2, 3, 1, 4] : vector<2x2x8x1x1xf32> to vector<2x8x1x2x1xf32>
//       CHECK:     vector.transfer_write %[[LOOP_T]], %[[B2]]
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [4, 4, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  promote_operands = [0, 1]
}>

!eltype = f32
!aeltype = f32

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_mfma_16x16x4 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_mfma_16x16x4()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>> -> tensor<2048x1280x!eltype>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>> -> tensor<10240x1280x!eltype>
        %5 = tensor.empty() : tensor<2048x10240x!aeltype>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280x!eltype>, tensor<10240x1280x!eltype>)
          outs(%6 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240x!aeltype> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_mfma_16x16x4
//   CHECK-DAG:   memref.alloc() : memref<64x10xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x10xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c320 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
// CHECK-COUNT-8:     amdgpu.mfma {{.*}}blocks = 1 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:       scf.yield
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [4, 4, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
  promote_operands = [0, 1]
}>

!eltype = f8E4M3FNUZ
!aeltype = f32

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_mfma_16x16x32_f8 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_mfma_16x16x32_f8()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>> -> tensor<2048x1280x!eltype>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>> -> tensor<10240x1280x!eltype>
        %5 = tensor.empty() : tensor<2048x10240x!aeltype>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280x!eltype>, tensor<10240x1280x!eltype>)
          outs(%6 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240x!aeltype> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_mfma_16x16x32_f8
//   CHECK-DAG:   memref.alloc() : memref<64x72xf8E4M3FNUZ, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x72xf8E4M3FNUZ, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c40 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
// CHECK-COUNT-8:     amdgpu.mfma {{.*}}blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:       scf.yield
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [2, 2, 0],
  reduction = [0, 0, 2],
  subgroup = [1, 1],
  mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>,
  promote_operands = [0, 1]
}>

!eltype = i8
!aeltype = i32

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_mfma_32x32x16_i8 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_mfma_32x32x16_i8()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>> -> tensor<2048x1280x!eltype>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>> -> tensor<10240x1280x!eltype>
        %5 = tensor.empty() : tensor<2048x10240x!aeltype>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280x!eltype>, tensor<10240x1280x!eltype>)
          outs(%6 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240x!aeltype> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_mfma_32x32x16_i8
//   CHECK-DAG:   memref.alloc() : memref<64x40xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x40xi8, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c80 step %c2 {{.*}} -> (vector<1x1x4x4x1xi32>)
// CHECK-COUNT-2:     amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
//       CHECK:       scf.yield
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [4, 4, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F16>,
  promote_operands = [0, 1]
}>

!eltype = f16
!aeltype = f16

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_wmma_f16_16x16x16_f16 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_wmma_f16_16x16x16_f16()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280x!eltype>> -> tensor<2048x1280x!eltype>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280x!eltype>> -> tensor<10240x1280x!eltype>
        %5 = tensor.empty() : tensor<2048x10240x!aeltype>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280x!eltype>, tensor<10240x1280x!eltype>)
          outs(%6 : tensor<2048x10240x!aeltype>) -> tensor<2048x10240x!aeltype>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240x!aeltype> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240x!aeltype>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_transpose_b_wmma_f16_16x16x16_f16
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.forall ({{.*}}) in (32, 160) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x16x1x1xf16>)
// CHECK-COUNT-8:     amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<16xf16>
//       CHECK:       scf.yield
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#lowering_config = #iree_gpu.lowering_config<{
  reduction = [0, 0, 0, 0, 1, 3, 3],
  thread = [1, 1, 1, 1, 0, 0, 0],
  workgroup = [1, 1, 4, 8, 0, 0, 0]
}>

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [8, 4, 1] subgroup_size = 32>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_nchw_fused ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nchw_fused() attributes {translation_info = #translation_info} {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant dense<1.0> : tensor<1x64xf32>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x64x58x58xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64x3x3xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x64x56x56xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 64, 58, 58], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x64x58x58xf32>> -> tensor<1x64x58x58xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 64, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64x3x3xf32>> -> tensor<64x64x3x3xf32>
        %5 = tensor.empty() : tensor<1x64x56x56xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
        %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #lowering_config, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%6 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %cst_0 : tensor<1x64x56x56xf32>, tensor<1x64xf32>) outs(%5 : tensor<1x64x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in, %in_1 : f32
          %10 = arith.cmpf ugt, %9, %cst : f32
          %11 = arith.select %10, %9, %cst : f32
          linalg.yield %11 : f32
        } -> tensor<1x64x56x56xf32>
        flow.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [1, 64, 56, 56], strides = [1, 1, 1, 1] : tensor<1x64x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x64x56x56xf32>>
        return
      }
    }
  }
}

// Verify that it compiles, meaning the consumer was successfully fused into
// the producer's (convolution's) distributed scf.forall loop.
// CHECK-LABEL: func @conv_nchw_fused
//       CHECK:   %[[ALLOCA:.+]] = memref.alloca() : memref<1x1x1x1xf32, #gpu.address_space<private>>
//       CHECK:   scf.forall ({{.*}}) in (64, 14, 7) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c64 step %c1
//       CHECK:       linalg.conv_2d_nchw_fchw
//  CHECK-SAME:         outs(%[[ALLOCA]] : memref<1x1x1x1xf32, #gpu.address_space<private>>)
//       CHECK:     arith.addf
//       CHECK:     arith.cmpf
//       CHECK:     arith.select
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#lowering_config = #iree_gpu.lowering_config<{
  reduction = [0, 0, 4],
  thread = [1, 4, 0],
  workgroup = [4, 32, 0],
  promote_operands = [0, 1]
}>

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [8, 4, 1] subgroup_size = 32>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @skinny_matmul_config ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @skinny_matmul_config() attributes {translation_info = #translation_info} {
        %cst = arith.constant 0.000000e+00 : f32
        %c102227904 = arith.constant 102227904 : index
        %c111444672 = arith.constant 111444672 : index
        %c4014080 = arith.constant 4014080 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c102227904) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c4014080) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x3136xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c111444672) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<128x3136xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 3136], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x3136xf32>> -> tensor<256x3136xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
        %7 = tensor.empty() : tensor<128x3136xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<128x3136xf32>) -> tensor<128x3136xf32>
        %9 = linalg.matmul {lowering_config = #lowering_config} ins(%4, %5 : tensor<128x256xf32>, tensor<256x3136xf32>) outs(%8 : tensor<128x3136xf32>) -> tensor<128x3136xf32>
        %10 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%9, %6 : tensor<128x3136xf32>, tensor<128xf32>) outs(%7 : tensor<128x3136xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.cmpf ugt, %11, %cst : f32
          %13 = arith.select %12, %11, %cst : f32
          linalg.yield %13 : f32
        } -> tensor<128x3136xf32>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 3136], strides = [1, 1] : tensor<128x3136xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x3136xf32>>
        return
      }
    }
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 8 + s2 * 32)>
// CHECK: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 8 + s1)>

// CHECK-LABEL: func @skinny_matmul_config

//   CHECK-DAG:   %[[IDX:.+]] = gpu.thread_id  x
//   CHECK-DAG:   %[[IDY:.+]] = gpu.thread_id  y
//   CHECK-DAG:   %[[IDZ:.+]] = gpu.thread_id  z
//       CHECK:   %[[LINID0:.+]] = affine.apply #[[$MAP]]()[%[[IDX]], %[[IDY]], %[[IDZ]]]
//       CHECK:   %[[IDS:.+]]:2 = affine.delinearize_index %[[LINID0:.+]] into (%c4, %c8) : index, index
//       CHECK:   %[[LINID1:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#0, %[[IDS]]#1]
//       CHECK:   scf.forall ({{.*}}) in (32, 98) {
//       CHECK:     scf.for %{{.*}} = %c0 to %c256 step %c4 {{.*}} -> (vector<1x4xf32>)
//       CHECK:       scf.for %{{.*}} = %[[LINID1]] to %c4 step %c32
//       CHECK:         %[[READ:.+]] = vector.transfer_read {{.*}} : memref<128x256xf32, {{.*}}storage_buffer>>, vector<4xf32>
//       CHECK:         vector.transfer_write %[[READ]], %{{.*}} : vector<4xf32>, memref<4x6xf32, #gpu.address_space<workgroup>>
//       CHECK:       vector.contract
//       CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 2, 1] subgroup_size = 32>

#lowering_config = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<WMMA_I32_16x16x16_I8>,
  reduction = [0, 0, 4],
  subgroup = [2, 4, 0],
  workgroup = [4, 8, 0],
  promote_operands = [0, 1]
}>

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_fused_multi_result ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_fused_multi_result() attributes {translation_info = #translation_info} {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3136x64xi8>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256xi8>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xi32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<256x3136xi8>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<256x3136xi8>>
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<3136x256xi8>>
        %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3136, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3136x64xi8>> -> tensor<3136x64xi8>
        %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xi8>> -> tensor<64x256xi8>
        %8 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xi32>> -> tensor<256xi32>
        %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [256, 3136], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x3136xi8>> -> tensor<256x3136xi8>
        %10 = tensor.empty() : tensor<3136x256xi8>
        %11 = tensor.empty() : tensor<256x3136xi8>
        %12 = tensor.empty() : tensor<3136x256xi32>
        %13 = linalg.fill ins(%c0_i32 : i32) outs(%12 : tensor<3136x256xi32>) -> tensor<3136x256xi32>
        %14 = linalg.matmul {lowering_config = #lowering_config} ins(%6, %7 : tensor<3136x64xi8>, tensor<64x256xi8>) outs(%13 : tensor<3136x256xi32>) -> tensor<3136x256xi32>
        %15:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %8, %9 : tensor<3136x256xi32>, tensor<256xi32>, tensor<256x3136xi8>) outs(%11, %10 : tensor<256x3136xi8>, tensor<3136x256xi8>) {
        ^bb0(%in: i32, %in_4: i32, %in_5: i8, %out: i8, %out_6: i8):
          %16 = arith.addi %in, %in_4 : i32
          %17 = arith.extsi %in_5 : i8 to i32
          %18 = arith.addi %16, %17 : i32
          %19 = arith.minsi %16, %17 : i32
          %20 = arith.trunci %18 : i32 to i8
          %21 = arith.trunci %19 : i32 to i8
          linalg.yield %20, %21 : i8, i8
        } -> (tensor<256x3136xi8>, tensor<3136x256xi8>)
        flow.dispatch.tensor.store %15#0, %4, offsets = [0, 0], sizes = [256, 3136], strides = [1, 1] : tensor<256x3136xi8> -> !flow.dispatch.tensor<writeonly:tensor<256x3136xi8>>
        flow.dispatch.tensor.store %15#1, %5, offsets = [0, 0], sizes = [3136, 256], strides = [1, 1] : tensor<3136x256xi8> -> !flow.dispatch.tensor<writeonly:tensor<3136x256xi8>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_fused_multi_result
// CHECK-DAG:      memref.alloc() : memref<64x72xi8, #gpu.address_space<workgroup>>
// CHECK-DAG:      memref.alloc() : memref<64x136xi8, #gpu.address_space<workgroup>>
// CHECK-COUNT-32: amdgpu.wmma {{.*}} : vector<16xi8>, vector<16xi8>, vector<8xi32>
// CHECK:          vector.transfer_write {{.*}} : vector<4x1x2x8x1xi8>, memref<16x16x196x8x2xi8, #hal.descriptor_type<storage_buffer>>
// CHECK:          vector.transfer_write {{.*}} : vector<2x8x1x4x1xi8>, memref<196x8x2x16x16xi8, #hal.descriptor_type<storage_buffer>>

// -----

#lowering_config = #iree_gpu.lowering_config<{
  thread = [1, 1], workgroup = [1, 1]
}>

#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @small_elementwise ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @small_elementwise() attributes {translation_info = #translation_info} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x3xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x3xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x3xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x3xf32>> -> tensor<1x3xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x3xf32>> -> tensor<1x3xf32>
        %5 = tensor.empty() : tensor<1x3xf32>
        %6 = linalg.add {lowering_config = #lowering_config} ins(%3, %4 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%5 : tensor<1x3xf32>) -> tensor<1x3xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x3xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @small_elementwise
//       CHECK:   %[[B:.+]] = hal.interface.binding.subspan layout({{.*}}) binding(2)
//       CHECK:   %[[ADD:.+]] = arith.addf %{{.*}}, %{{.*}} : vector<1xf32>
//       CHECK:   vector.transfer_write %[[ADD]], %[[B]]

// -----

#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>],
  flags = Indirect
>
#translation_info = #iree_codegen.translation_info<
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_shared_memory = false,
      no_reduce_shared_memory_bank_conflicts = true>
  }
>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 1, 0],
  reduction = [0, 0, 1],
  promote_operands = [0, 1]
}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b_mfma ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @multi_mma_data_tiled_unrolled_MFMA_F32_16x16x4_F32()
        attributes {translation_info = #translation_info} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x1x8x4x16x4xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x1x4x2x4x16x4xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<readwrite:tensor<4x4x8x4x2x4x16x4xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [4, 1, 8, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1x8x4x16x4xf32>> -> tensor<4x1x8x4x16x4xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [4, 1, 4, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1x4x2x4x16x4xf32>> -> tensor<4x1x4x2x4x16x4xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [4, 4, 8, 4, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x4x8x4x2x4x16x4xf32>> -> tensor<4x4x8x4x2x4x16x4xf32>
        %6 = iree_gpu.multi_mma %3, %4, %5 {
          lowering_config = #config,
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = [
            #iree_gpu.iterator_type<parallel>,
            #iree_gpu.iterator_type<parallel>,
            #iree_gpu.iterator_type<reduction>],
          kind = #iree_gpu.data_tiled_mma_layout<
            intrinsic = MFMA_F32_16x16x4_F32,
            unroll_m = 8,
            unroll_n = 2,
            unroll_n_to_subgroups = 4,
            unroll_k = 4>}
          : tensor<4x1x8x4x16x4xf32>, tensor<4x1x4x2x4x16x4xf32> into tensor<4x4x8x4x2x4x16x4xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [4, 4, 8, 4, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x4x8x4x2x4x16x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x4x8x4x2x4x16x4xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @multi_mma_data_tiled_unrolled_MFMA_F32_16x16x4_F32()
// CHECK-DAG:  %[[BINDING_A:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:  %[[BINDING_B:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:  %[[BINDING_C:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK-DAG:  %[[A_ALLOC:.+]] = memref.alloc() : memref<1x1x8x4x16x4xf32, #gpu.address_space<workgroup>>
// CHECK-DAG:  %[[B_ALLOC:.+]] = memref.alloc() : memref<1x1x4x2x4x16x4xf32, #gpu.address_space<workgroup>>
// CHECK:      gpu.barrier
// CHECK-DAG:  %[[A_GLOBAL_LOAD:.+]] = vector.transfer_read %[[BINDING_A]]{{.*}} vector<4xf32>
// CHECK-DAG:  %[[B_GLOBAL_LOAD:.+]] = vector.transfer_read %[[BINDING_B]]{{.*}} vector<4xf32>
// CHECK-DAG:  vector.transfer_write %[[A_GLOBAL_LOAD]], %[[A_ALLOC]]
// CHECK-DAG:  vector.transfer_write %[[B_GLOBAL_LOAD]], %[[B_ALLOC]]
// CHECK:      gpu.barrier
// CHECK-DAG:  %[[A_READ:.+]] = vector.transfer_read %[[A_ALLOC]]{{.*}} vector<8x1x1x4xf32>
// CHECK-DAG:  %[[B_READ:.+]] = vector.transfer_read %[[B_ALLOC]]{{.*}} vector<2x1x1x4xf32>
// CHECK-DAG:  %[[C_READ:.+]] = vector.transfer_read %[[BINDING_C]]{{.*}} vector<8x1x2x1x1x4xf32>
// CHECK-DAG:  %[[C_00_0:.+]] = vector.extract %[[C_READ]][0, 0, 0, 0, 0] : vector<4xf32> from vector<8x1x2x1x1x4xf32>
// CHECK-DAG:  %[[C_01_0:.+]] = vector.extract %[[C_READ]][0, 0, 1, 0, 0] : vector<4xf32> from vector<8x1x2x1x1x4xf32>
// CHECK-DAG:  %[[C_70_0:.+]] = vector.extract %[[C_READ]][7, 0, 0, 0, 0] : vector<4xf32> from vector<8x1x2x1x1x4xf32>
// CHECK-DAG:  %[[C_71_0:.+]] = vector.extract %[[C_READ]][7, 0, 1, 0, 0] : vector<4xf32> from vector<8x1x2x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT00:.+]] = vector.extract %[[A_READ]][0, 0, 0, 0] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT01:.+]] = vector.extract %[[A_READ]][0, 0, 0, 1] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT02:.+]] = vector.extract %[[A_READ]][0, 0, 0, 2] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT03:.+]] = vector.extract %[[A_READ]][0, 0, 0, 3] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT70:.+]] = vector.extract %[[A_READ]][7, 0, 0, 0] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT71:.+]] = vector.extract %[[A_READ]][7, 0, 0, 1] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT72:.+]] = vector.extract %[[A_READ]][7, 0, 0, 2] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[A_EXTRACT73:.+]] = vector.extract %[[A_READ]][7, 0, 0, 3] : f32 from vector<8x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT00:.+]] = vector.extract %[[B_READ]][0, 0, 0, 0] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT01:.+]] = vector.extract %[[B_READ]][0, 0, 0, 1] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT02:.+]] = vector.extract %[[B_READ]][0, 0, 0, 2] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT03:.+]] = vector.extract %[[B_READ]][0, 0, 0, 3] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT10:.+]] = vector.extract %[[B_READ]][1, 0, 0, 0] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT11:.+]] = vector.extract %[[B_READ]][1, 0, 0, 1] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT12:.+]] = vector.extract %[[B_READ]][1, 0, 0, 2] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[B_EXTRACT13:.+]] = vector.extract %[[B_READ]][1, 0, 0, 3] : f32 from vector<2x1x1x4xf32>
// CHECK-DAG:  %[[C_00_1:.+]] = amdgpu.mfma %[[A_EXTRACT00]] * %[[B_EXTRACT00]] + %[[C_00_0]]
// CHECK-DAG:  %[[C_00_2:.+]] = amdgpu.mfma %[[A_EXTRACT01]] * %[[B_EXTRACT01]] + %[[C_00_1]]
// CHECK-DAG:  %[[C_00_3:.+]] = amdgpu.mfma %[[A_EXTRACT02]] * %[[B_EXTRACT02]] + %[[C_00_2]]
// CHECK-DAG:  %[[C_00_4:.+]] = amdgpu.mfma %[[A_EXTRACT03]] * %[[B_EXTRACT03]] + %[[C_00_3]]
// CHECK-DAG:  %[[C_01_1:.+]] = amdgpu.mfma %[[A_EXTRACT00]] * %[[B_EXTRACT10]] + %[[C_01_0]]
// CHECK-DAG:  %[[C_01_2:.+]] = amdgpu.mfma %[[A_EXTRACT01]] * %[[B_EXTRACT11]] + %[[C_01_1]]
// CHECK-DAG:  %[[C_01_3:.+]] = amdgpu.mfma %[[A_EXTRACT02]] * %[[B_EXTRACT12]] + %[[C_01_2]]
// CHECK-DAG:  %[[C_01_4:.+]] = amdgpu.mfma %[[A_EXTRACT03]] * %[[B_EXTRACT13]] + %[[C_01_3]]
// CHECK-DAG:  %[[C_70_1:.+]] = amdgpu.mfma %[[A_EXTRACT70]] * %[[B_EXTRACT00]] + %[[C_70_0]]
// CHECK-DAG:  %[[C_70_2:.+]] = amdgpu.mfma %[[A_EXTRACT71]] * %[[B_EXTRACT01]] + %[[C_70_1]]
// CHECK-DAG:  %[[C_70_3:.+]] = amdgpu.mfma %[[A_EXTRACT72]] * %[[B_EXTRACT02]] + %[[C_70_2]]
// CHECK-DAG:  %[[C_70_4:.+]] = amdgpu.mfma %[[A_EXTRACT73]] * %[[B_EXTRACT03]] + %[[C_70_3]]
// CHECK-DAG:  %[[C_71_1:.+]] = amdgpu.mfma %[[A_EXTRACT70]] * %[[B_EXTRACT10]] + %[[C_71_0]]
// CHECK-DAG:  %[[C_71_2:.+]] = amdgpu.mfma %[[A_EXTRACT71]] * %[[B_EXTRACT11]] + %[[C_71_1]]
// CHECK-DAG:  %[[C_71_3:.+]] = amdgpu.mfma %[[A_EXTRACT72]] * %[[B_EXTRACT12]] + %[[C_71_2]]
// CHECK-DAG:  %[[C_71_4:.+]] = amdgpu.mfma %[[A_EXTRACT73]] * %[[B_EXTRACT13]] + %[[C_71_3]]
// CHECK:  vector.insert_strided_slice %[[C_00_4]], {{.*}}offsets = [0, 0, 0, 0, 0]{{.*}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:  vector.insert_strided_slice %[[C_01_4]], {{.*}}offsets = [0, 1, 0, 0, 0]{{.*}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:  vector.insert_strided_slice %[[C_70_4]], {{.*}}offsets = [7, 0, 0, 0, 0]{{.*}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:  vector.insert_strided_slice %[[C_71_4]], {{.*}}offsets = [7, 1, 0, 0, 0]{{.*}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:  vector.transfer_write

// -----

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  reduction = [0, 0, 4],
  thread = [1, 4, 0],
  workgroup = [1, 128, 0]
}>
#translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>

hal.executable public @main {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @small_m_matmul ordinal(0) layout(#layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @small_m_matmul() attributes {translation_info = #translation_info} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x1000xf32>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1000x512xf32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<4x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 1000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1000xf32>> -> tensor<4x1000xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1000, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1000x512xf32>> -> tensor<1000x512xf32>
        %5 = tensor.empty() : tensor<4x512xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x512xf32>) -> tensor<4x512xf32>
        %7 = linalg.matmul {lowering_config = #lowering_config}
          ins(%3, %4 : tensor<4x1000xf32>, tensor<1000x512xf32>)
          outs(%6 : tensor<4x512xf32>) -> tensor<4x512xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [4, 512], strides = [1, 1] : tensor<4x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x512xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @small_m_matmul
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<1x6xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<4x130xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c1000 step %c4 {{.*}} -> (vector<1x4xf32>)
//       CHECK:     gpu.barrier
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %c1 step %c32
//       CHECK:       %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<4xf32>
//  CHECK-NEXT:       vector.transfer_write %[[LHS_RD]], %[[LHS_ALLOC]]
//       CHECK:     gpu.barrier
//   CHECK-DAG:     %[[LHS_MM:.+]] = vector.transfer_read %[[LHS_ALLOC]]{{.*}} vector<4xf32>
//   CHECK-DAG:     %[[RHS_MM:.+]] = vector.transfer_read %[[RHS_ALLOC]]{{.*}} vector<4x4xf32>
//       CHECK:     vector.contract {{.*}} %[[LHS_MM]], %[[RHS_MM]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 64, 0],
  reduction = [0, 0, 2],
  thread = [1, 1, 0]
}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @small_matvec ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @small_matvec()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10x10xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10x1xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<readwrite:tensor<10x1xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [10, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10x10xf32>> -> tensor<10x10xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10x1xf32>> -> tensor<10x1xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [10, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<10x1xf32>> -> tensor<10x1xf32>
        %6 = linalg.matmul {lowering_config = #config}
          ins(%3, %4 : tensor<10x10xf32>, tensor<10x1xf32>)
          outs(%5 : tensor<10x1xf32>) -> tensor<10x1xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [10, 1], strides = [1, 1] : tensor<10x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<10x1xf32>>
        return
      }
    }
  }
}

// Note that current barrier placement logic is observedly poor. Some cleanup
// analysis should be able to simplify the below to just two barriers.

// CHECK-LABEL: func @small_matvec
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %c1 step %c64

// Verify that the write does not get hoisted out of the single threaded
// for loop.
//       CHECK:       vector.transfer_write %{{.*}}, %[[B2]]{{.*}} memref<10x1xf32, #hal.descriptor_type<storage_buffer>>
//  CHECK-NEXT:     }
//  CHECK-NEXT:   } {mapping = [#iree_codegen.workgroup_mapping<x>]}
//  CHECK-NEXT:   return
