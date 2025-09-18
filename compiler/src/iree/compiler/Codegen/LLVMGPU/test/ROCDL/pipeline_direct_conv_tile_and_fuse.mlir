// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [512, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_shared_memory = false,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = false>
  }>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 1, 32, 64, 0, 0, 0],
  subgroup = [0, 0, 1, 1, 0, 0, 0],
  reduction = [0, 0, 0, 0, 1, 1, 4],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_igemm_im2col ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_fhwc() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x50x34x576xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3x576xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x48x32x576xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 48, 32, 576], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x50x34x576xf16>> -> tensor<16x50x34x576xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [576, 3, 3, 576], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3x576xf16>> -> tensor<576x3x3x576xf16>
        %5 = tensor.empty() : tensor<16x48x32x576xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32>
        %7 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<16x50x34x576xf16>, tensor<576x3x3x576xf16>) outs(%6 : tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [16, 48, 32, 576], strides = [1, 1, 1, 1] : tensor<16x48x32x576xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x48x32x576xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_nhwc_fhwc
//      CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//      CHECK-DAG:   %[[ASSUMED_B0:.+]] = memref.assume_alignment %[[B0]], 64
//      CHECK-DAG:   %[[BUF0:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B0]]
//      CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   %[[ASSUMED_B1:.+]] = memref.assume_alignment %[[B1]], 64
//      CHECK-DAG:   %[[BUF1:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B1]]
//      CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   %[[ASSUMED_B2:.+]] = memref.assume_alignment %[[B2]], 64
//      CHECK-DAG:   %[[BUF2:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_B2]]
//      CHECK-DAG:   memref.alloc() : memref<64x1x1x68xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<1x1x32x68xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//      CHECK-DAG:   %[[C36:.+]] = arith.constant 36 : index
//          CHECK:   scf.forall ({{.*}}) in (16, 48, 9) {
//          CHECK:     %[[LOOP1:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {{.*}} -> (vector<1x1x1x1x4x1xf32>)
//          CHECK:       %[[LOOP2:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {{.*}} -> (vector<1x1x1x1x4x1xf32>)
//          CHECK:         %[[LOOP3:.+]] = scf.for %[[IV3:.+]] = %[[C0]] to %[[C36]] step %[[C4]] {{.*}} -> (vector<1x1x1x1x4x1xf32>)
//          CHECK:           gpu.barrier
//      CHECK-DAG:           %[[LHS_RD:.+]] = vector.transfer_read %[[BUF0]]{{.*}} vector<4xf16>
//      CHECK-DAG:           vector.transfer_write %[[LHS_RD]]
//      CHECK-DAG:           %[[RHS_RD:.+]] = vector.transfer_read %[[BUF1]]{{.*}} vector<8xf16>
//      CHECK-DAG:           vector.transfer_write %[[RHS_RD]]
//          CHECK:           gpu.barrier
//      CHECK-DAG:           %[[LHS_MM:.+]] = vector.transfer_read {{.*}} vector<4x4xf16>
//      CHECK-DAG:           %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<4x4xf16>
//  CHECK-COUNT-4:           amdgpu.mfma {{.*}}blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//          CHECK:     vector.transfer_write %{{.*}}, %[[BUF2]]
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
