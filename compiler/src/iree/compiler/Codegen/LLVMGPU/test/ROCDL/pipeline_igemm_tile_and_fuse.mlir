// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_shared_memory = false,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 4, 16, 256, 0],
  reduction = [0, 0, 0, 0, 2],
  subgroup = [1, 4, 1, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_igemm_im2col ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_stride_2() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x16x16x1280xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>> -> tensor<2x34x34x1280xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x1280xf16>
        %5 = tensor.empty() : tensor<2x16x16x1280xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x16x16x1280xf32>) -> tensor<2x16x16x1280xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%6 : tensor<2x16x16x1280xf32>) -> tensor<2x16x16x1280xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 16, 1280], strides = [1, 1, 1, 1] : tensor<2x16x16x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x16x16x1280xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_nhwc
//      CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//      CHECK-DAG:   %[[ASSUMED_B0:.+]] = memref.assume_alignment %[[B0]], 64
//      CHECK-DAG:   %[[BUF0:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B0]]
//      CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   %[[ASSUMED_B1:.+]] = memref.assume_alignment %[[B1]], 64
//      CHECK-DAG:   %[[BUF1:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B1]]
//      CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   %[[ASSUMED_B2:.+]] = memref.assume_alignment %[[B2]], 64
//      CHECK-DAG:   %[[BUF2:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B2]]
//      CHECK-DAG:   memref.alloc() : memref<1x4x16x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<32x260xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C720:.+]] = arith.constant 720 : index
//      CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//          CHECK:   scf.forall ({{.*}}) in (2, 4, 5) {
//          CHECK:     %[[LOOP:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C720]] step %[[C2]] {{.*}} -> (vector<1x4x1x4x4x1xf32>)
//          CHECK:       gpu.barrier
//      CHECK-DAG:       %[[LHS_RD:.+]] = vector.transfer_read %[[BUF0]]{{.*}} vector<8xf16>
//      CHECK-DAG:       vector.transfer_write %[[LHS_RD]]
//      CHECK-DAG:       %[[RHS_RD:.+]] = vector.transfer_read %[[BUF1]]{{.*}} vector<8xf16>
//      CHECK-DAG:       vector.transfer_write %[[RHS_RD]]
//          CHECK:       gpu.barrier
//      CHECK-DAG:       %[[LHS_MM0:.+]] = vector.transfer_read {{.*}} vector<4x1x1x2x4xf16>
//      CHECK-DAG:       %[[LHS_MM1:.+]] = vector.broadcast {{.*}} vector<4x1x1x2x4xf16> to vector<1x4x1x1x2x4xf16>
//      CHECK-DAG:       %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x4x4x1xf16>
//      CHECK-DAG:       vector.transpose %[[LHS_MM1]], [0, 1, 2, 4, 3, 5] : vector<1x4x1x1x2x4xf16> to vector<1x4x1x2x1x4xf16>
//      CHECK-DAG:       vector.transpose %[[RHS_MM]], [0, 2, 3, 1] : vector<2x4x4x1xf16> to vector<2x4x1x4xf16>
// CHECK-COUNT-32:       amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//          CHECK:     %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 1, 2, 4, 3, 5] : vector<1x4x1x4x4x1xf32> to vector<1x4x1x4x4x1xf32>
//          CHECK:     %[[EXTRACT:.+]] = vector.extract %[[LOOP_T]][0] : vector<4x1x4x4x1xf32> from vector<1x4x1x4x4x1xf32>
//          CHECK:     vector.transfer_write %[[EXTRACT]], %[[BUF2]]
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// TODO(Max191): Add tests for more convolution types

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_shared_memory = false,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  padding = [2, 1, 32, 16, 16],
  workgroup = [2, 1, 32, 16, 0],
  reduction = [0, 0, 0, 0, 1],
  subgroup = [1, 1, 1, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1, 2]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_dispatch_0_conv_2d_nhwc_hwcf_2x17x17x1281x3x3x1281_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_unaligned_stride_2() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x35x35x1281xf16>>          %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1281x1281xf16>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x17x17x1281xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 35, 35, 1281], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x35x35x1281xf16>> -> tensor<2x35x35x1281xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1281, 1281], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1281x1281xf16>> -> tensor<3x3x1281x1281xf16>
        %5 = tensor.empty() : tensor<2x17x17x1281xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x17x17x1281xf32>) -> tensor<2x17x17x1281xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<2x35x35x1281xf16>, tensor<3x3x1281x1281xf16>) outs(%6 : tensor<2x17x17x1281xf32>) -> tensor<2x17x17x1281xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 17, 17, 1281], strides = [1, 1, 1, 1] : tensor<2x17x17x1281xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x17x17x1281xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_nhwc_unaligned
//      CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//      CHECK-DAG:   %[[ASSUMED_B0:.+]] = memref.assume_alignment %[[B0]], 64
//      CHECK-DAG:   %[[BUF0:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B0]]
//      CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   %[[ASSUMED_B1:.+]] = memref.assume_alignment %[[B1]], 64
//      CHECK-DAG:   %[[BUF1:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B1]]
//      CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   %[[ASSUMED_B2:.+]] = memref.assume_alignment %[[B2]], 64
//      CHECK-DAG:   %[[BUF2:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B2]]
//      CHECK-DAG:   memref.alloc() : memref<2x1x32x18xf32, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<16x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<2x1x32x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C721:.+]] = arith.constant 721 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//          CHECK:   scf.forall ({{.*}}) in (17, 81) {
//          CHECK:     %[[LOOP:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C721]] step %[[C1]] {{.*}} -> (vector<1x1x1x1x4x1xf32>)
//          CHECK:       gpu.barrier
//      CHECK-DAG:       %[[LHS_MM0:.+]] = vector.transfer_read {{.*}} vector<4xf16>
//      CHECK-DAG:       %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<4xf16>
// CHECK-COUNT-1:       amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//          CHECK:     %[[LOOP_T:.+]] = vector.shape_cast %[[LOOP]] : vector<1x1x1x1x4x1xf32> to vector<4xf32>
//          CHECK:     vector.transfer_write %[[LOOP_T]]
// Note there is a writeback loop here that is skipped to simplify the test.
//       CHECK:        memref.copy {{.*}}#gpu.address_space<workgroup>> to {{.*}}#amdgpu.address_space<fat_raw_buffer>
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
