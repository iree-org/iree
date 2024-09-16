// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
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
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c1280 step %c4 {{.*}} -> (vector<8x4xf32>)
//   CHECK-DAG:     %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<2xf16>
//   CHECK-DAG:     vector.transfer_write %[[LHS_RD]], %[[LHS_ALLOC:[A-Za-z0-9]+]]
//   CHECK-DAG:     %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<2xf16>
//   CHECK-DAG:     vector.transfer_write %[[RHS_RD]], %[[RHS_ALLOC:[A-Za-z0-9]+]]
//       CHECK:     gpu.barrier
//   CHECK-DAG:     %[[LHS_MM:.+]] = vector.transfer_read %[[LHS_ALLOC]]{{.*}} vector<8x4xf16>
//   CHECK-DAG:     %[[RHS_MM:.+]] = vector.transfer_read %[[RHS_ALLOC]]{{.*}} vector<4x4xf16>
//       CHECK:     gpu.barrier
//       CHECK:     %[[MM:.+]] = vector.contract {{.*}} %[[LHS_MM]], %[[RHS_MM]]
//       CHECK:     scf.yield %[[MM]]
//       CHECK:   vector.transfer_write %[[LOOP]], %[[B2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 2], subgroup = [2, 2], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
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
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
//   CHECK-DAG:     %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<8xf16>
//   CHECK-DAG:     vector.transfer_write %[[LHS_RD]]
//   CHECK-DAG:     %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<8xf16>
//   CHECK-DAG:     vector.transfer_write %[[RHS_RD]]
//       CHECK:     gpu.barrier
//       CHECK:     %[[LHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x1x2x4xf16>
//       CHECK:     %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x1x2x4xf16>
//       CHECK:     gpu.barrier
//   CHECK-DAG:     vector.transpose %[[LHS_MM]], [0, 2, 1, 3] : vector<2x1x2x4xf16>
//   CHECK-DAG:     vector.transpose %[[RHS_MM]], [0, 2, 1, 3] : vector<2x1x2x4xf16>
// CHECK-COUNT-4:   amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:     scf.yield
//       CHECK:   %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 2, 1, 3] : vector<2x2x4x1xf32> to vector<2x4x2x1xf32>
//       CHECK:   vector.transfer_write %[[LOOP_T]], %[[B2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{workgroup = [1, 64, 64, 0], reduction = [0, 0, 0, 2], subgroup = [1, 2, 2], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_igemm_im2col ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_igemm_im2col() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 2, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x16x16x1280xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>> -> tensor<2x34x34x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x1280xf16>
        %5 = tensor.empty() : tensor<2x16x16x1280xf32>
        %6 = tensor.empty() : tensor<2x256x11520xf16>
        %7 = iree_linalg_ext.im2col
            strides = [2, 2] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [0] k_offset = [0]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          ins(%3 : tensor<2x34x34x1280xf16>)
          outs(%6 : tensor<2x256x11520xf16>) -> tensor<2x256x11520xf16>
        %collapsed = tensor.collapse_shape %4 [[0, 1, 2], [3]] : tensor<3x3x1280x1280xf16> into tensor<11520x1280xf16>
        %collapsed_0 = tensor.collapse_shape %5 [[0], [1, 2], [3]] : tensor<2x16x16x1280xf32> into tensor<2x256x1280xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%collapsed_0 : tensor<2x256x1280xf32>) -> tensor<2x256x1280xf32>
        %9 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%7, %collapsed : tensor<2x256x11520xf16>, tensor<11520x1280xf16>)
          outs(%8 : tensor<2x256x1280xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%in: f16, %in_1: f16, %out: f32):
          %10 = arith.extf %in : f16 to f32
          %11 = arith.extf %in_1 : f16 to f32
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        } -> tensor<2x256x1280xf32>
        %expanded = tensor.expand_shape %9 [[0], [1, 2], [3]] output_shape [2, 16, 16, 1280] : tensor<2x256x1280xf32> into tensor<2x16x16x1280xf32>
        flow.dispatch.tensor.store %expanded, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 16, 1280], strides = [1, 1, 1, 1] : tensor<2x16x16x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x16x16x1280xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func @conv_igemm_im2col
//     CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//     CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//     CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//     CHECK-DAG:   memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
//     CHECK-DAG:   memref.alloc() : memref<32x68xf16, #gpu.address_space<workgroup>>
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C720:.+]] = arith.constant 720 : index
//     CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//         CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C720]] step %[[C2]] {{.*}} -> (vector<1x2x2x4x1xf32>)
//     CHECK-DAG:     %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<8xf16>
//     CHECK-DAG:     vector.transfer_write %[[LHS_RD]]
//     CHECK-DAG:     %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<8xf16>
//     CHECK-DAG:     vector.transfer_write %[[RHS_RD]]
//         CHECK:     gpu.barrier
//         CHECK:     %[[LHS_MM0:.+]] = vector.transfer_read {{.*}} vector<2x1x2x4xf16>
//         CHECK:     %[[LHS_MM1:.+]] = vector.broadcast {{.*}} vector<2x1x2x4xf16> to vector<1x2x1x2x4xf16>
//         CHECK:     %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x4x2x1xf16>
//         CHECK:     gpu.barrier
//     CHECK-DAG:     vector.transpose %[[LHS_MM1]], [0, 1, 3, 2, 4] : vector<1x2x1x2x4xf16> to vector<1x2x2x1x4xf16>
//     CHECK-DAG:     vector.transpose %[[RHS_MM]], [0, 2, 3, 1] : vector<2x4x2x1xf16> to vector<2x2x1x4xf16>
// CHECK-COUNT-4:     amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//         CHECK:   %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 1, 3, 2, 4] : vector<1x2x2x4x1xf32> to vector<1x2x4x2x1xf32>
//         CHECK:   %[[EXTRACT:.+]] = vector.extract %[[LOOP_T]][0] : vector<2x4x2x1xf32> from vector<1x2x4x2x1xf32>
//         CHECK:   vector.transfer_write %[[EXTRACT]], %[[B2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>}>
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
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x8x1x1xf32>)
//   CHECK-DAG:     %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<2x8xf16>
//   CHECK-DAG:     vector.transfer_write %[[LHS_RD]]
//   CHECK-DAG:     %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<2x8xf16>
//   CHECK-DAG:     vector.transfer_write %[[RHS_RD]]
//       CHECK:     gpu.barrier
//       CHECK:     %[[LHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x1x2x16xf16>
//       CHECK:     %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x1x2x16xf16>
//       CHECK:     gpu.barrier
//   CHECK-DAG:     vector.transpose %[[LHS_MM]], [0, 2, 1, 3] : vector<2x1x2x16xf16>
//   CHECK-DAG:     vector.transpose %[[RHS_MM]], [0, 2, 1, 3] : vector<2x1x2x16xf16>
// CHECK-COUNT-8:   amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//       CHECK:     scf.yield
//       CHECK:   %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 2, 3, 1, 4] : vector<2x2x8x1x1xf32> to vector<2x8x1x2x1xf32>
//       CHECK:   vector.transfer_write %[[LOOP_T]], %[[B2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>}>

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
//       CHECK:   scf.for %{{.*}} = %c0 to %c320 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
// CHECK-COUNT-8:   amdgpu.mfma {{.*}}blocks = 1 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:     scf.yield

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>}>

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
//       CHECK:   scf.for %{{.*}} = %c0 to %c40 step %c2 {{.*}} -> (vector<2x2x4x1xf32>)
// CHECK-COUNT-8:   amdgpu.mfma {{.*}}blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32
//       CHECK:     scf.yield

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>}>

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
//       CHECK:   scf.for %{{.*}} = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x4x4x1xi32>)
// CHECK-COUNT-8:   amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
//       CHECK:     scf.yield

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 2],
  subgroup = [2, 2],
  mma_kind = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F16>}>

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
//       CHECK:   scf.for %{{.*}} = %c0 to %c80 step %c2 {{.*}} -> (vector<2x2x16x1x1xf16>)
// CHECK-COUNT-8:   amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<16xf16>
//       CHECK:     scf.yield

// -----

#lowering_config = #iree_gpu.lowering_config<{
  reduction = [0 : index, 0 : index, 0 : index, 0 : index, 1 : index, 3 : index, 3 : index],
  thread = [1 : index, 1 : index, 1 : index, 1 : index, 0 : index, 0 : index, 0 : index],
  workgroup = [1 : index, 1 : index, 4 : index, 8 : index, 0 : index, 0 : index, 0 : index]
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
//       CHECK:   %[[ALLOCA2:.+]] = memref.alloca() : memref<1x1x1x1xf32, #gpu.address_space<private>>
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c1
//       CHECK:     linalg.conv_2d_nchw_fchw
//  CHECK-SAME:       outs(%[[ALLOCA2]] : memref<1x1x1x1xf32, #gpu.address_space<private>>)
//       CHECK:   arith.addf
//       CHECK:   arith.cmpf
//       CHECK:   arith.select

// -----

#lowering_config = #iree_gpu.lowering_config<{
  reduction = [0 : index, 0 : index, 4 : index],
  thread = [1 : index, 4 : index, 0 : index],
  workgroup = [4 : index, 32 : index, 0 : index]
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
//       CHECK:   %[[IDS:.+]]:2 = affine.delinearize_index %[[LINID0:.+]] into (%c8, %c4) : index, index
//       CHECK:   %[[LINID1:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#1, %[[IDS]]#0]
//       CHECK:   scf.for %{{.*}} = %c0 to %c256 step %c4 {{.*}} -> (vector<1x4xf32>)
//       CHECK:     scf.for %{{.*}} = %[[LINID1]] to %c4 step %c32
//       CHECK:       %[[READ:.+]] = vector.transfer_read {{.*}} : memref<128x256xf32, {{.*}}storage_buffer>>, vector<4xf32>
//       CHECK:       vector.transfer_write %[[READ]], %{{.*}} : vector<4xf32>, memref<4x6xf32, #gpu.address_space<workgroup>>
//       CHECK:     vector.contract

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
  reduction = [0 : index, 0 : index, 4 : index],
  subgroup = [2 : index, 4 : index, 0 : index],
  workgroup = [64 : index, 128 : index, 0 : index]}>

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
  thread = [1 : index, 1 : index],
  workgroup = [1 : index, 1 : index]
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
