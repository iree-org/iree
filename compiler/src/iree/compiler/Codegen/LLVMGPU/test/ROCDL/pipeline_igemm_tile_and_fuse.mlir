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
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 64, 256, 0],
  reduction = [0, 0, 0, 2],
  subgroup = [1, 4, 4, 0],
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
//      CHECK-DAG:   hal.interface.binding.subspan layout({{.+}}) binding(0)
//      CHECK-DAG:   hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<32x260xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C360:.+]] = arith.constant 360 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//          CHECK:   scf.forall ({{.*}}) in (2, 4, 5) {
//          CHECK:     %[[LOOP:.+]] = scf.for {{.+}} = %[[C0]] to %[[C360]] step %[[C1]] {{.*}} -> (vector<1x4x4x4x1xf32>)
//          CHECK:       gpu.barrier memfence [#gpu.address_space<workgroup>]
//      CHECK-DAG:       vector.transfer_read {{.*}}#amdgpu.address_space<fat_raw_buffer>{{.*}} vector<8xf16>
//      CHECK-DAG:       vector.transfer_write {{.*}} vector<8xf16>
//          CHECK:       gpu.barrier memfence [#gpu.address_space<workgroup>]
//      CHECK-DAG:       vector.transfer_read {{.*}} vector<4x1x2x4xf16>
//      CHECK-DAG:       %[[LHS_MM1:.+]] = vector.shape_cast {{.*}} vector<4x1x2x4xf16> to vector<1x4x2x1x4xf16>
//      CHECK-DAG:       %[[RHS_MM:.+]] = vector.transfer_read {{.*}} vector<2x4x4x1xf16>
//      CHECK-DAG:       vector.transpose %[[RHS_MM]], [0, 2, 3, 1] : vector<2x4x4x1xf16> to vector<2x4x1x4xf16>
// CHECK-COUNT-32:       amdgpu.mfma 16x16x16
//          CHECK:     %[[LOOP_T:.+]] = vector.transpose %[[LOOP]], [0, 1, 3, 2, 4] : vector<1x4x4x4x1xf32> to vector<1x4x4x4x1xf32>
//          CHECK:     %[[CAST:.+]] = vector.shape_cast %[[LOOP_T]] : vector<1x4x4x4x1xf32> to vector<4x4x4x1xf32>
//          CHECK:     vector.transfer_write %[[CAST]]
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
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  padding = [1, 32, 16, 16],
  workgroup = [1, 32, 16, 0],
  reduction = [0, 0, 0, 1],
  subgroup = [1, 1, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1, 2]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_dispatch_0_conv_2d_nhwc_hwcf_2x17x17x1281x3x3x1281_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
//      CHECK-DAG:   memref.assume_alignment %[[B0]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//      CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   memref.assume_alignment %[[B1]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//      CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   memref.assume_alignment %[[B2]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//      CHECK-DAG:   memref.alloc() : memref<1x32x18xf32, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<16x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C721:.+]] = arith.constant 721 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//          CHECK:   scf.forall ({{.*}}) in (2, 10, 81) {
//          CHECK:     scf.for {{.+}} = %[[C0]] to %[[C721]] step %[[C1]]
//          CHECK:       gpu.barrier memfence [#gpu.address_space<workgroup>]
//      CHECK-DAG:       vector.transfer_read {{.*}} vector<1xf16>
//      CHECK-DAG:       vector.transfer_read {{.*}} vector<1xf16>
//  CHECK-COUNT-1:       amdgpu.mfma 16x16x16
// Note there is a writeback loop here that is skipped to simplify the test.
//          CHECK:        memref.copy {{.*}}#gpu.address_space<workgroup>> to {{.*}}#amdgpu.address_space<fat_raw_buffer>
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

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
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  padding = [1, 32, 16, 16],
  workgroup = [1, 32, 16, 0],
  reduction = [0, 0, 0, 1],
  subgroup = [1, 1, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_dispatch_0_conv_2d_nhwc_hwcf_2x17x17x1281x3x3x1281_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_unaligned_stride_2_nocpromo() attributes {translation_info = #translation} {
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
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C721:.+]] = arith.constant 721 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK-DAG:   memref.alloc() : memref<16x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//      CHECK-DAG:   memref.assume_alignment %[[B0]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//      CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//      CHECK-DAG:   memref.assume_alignment %[[B1]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//      CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//      CHECK-DAG:   memref.assume_alignment %[[B2]], 64
//      CHECK-DAG:   amdgpu.fat_raw_buffer_cast
//          CHECK:   scf.forall ({{.*}}) in (2, 10, 81) {
//          CHECK:     scf.for {{.+}} = %[[C0]] to %[[C721]] step %[[C1]]
//          CHECK:       gpu.barrier memfence [#gpu.address_space<workgroup>]
//      CHECK-DAG:       vector.transfer_read {{.*}} vector<1xf16>
//      CHECK-DAG:       vector.transfer_read {{.*}} vector<1xf16>
//  CHECK-COUNT-1:       amdgpu.mfma 16x16x16
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly">,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  padding = [2, 32, 64, 64],
  workgroup = [2, 32, 64, 0],
  reduction = [0, 0, 0, 4],
  subgroup = [2, 2, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>,
  promote_operands = [0, 1]
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_2d_input_backward_16x1x21x192_nhwc ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_2d_input_backward_16x1x21x192_nhwc() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x21x384xbf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x192xbf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x21x192xbf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 21, 384], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x21x384xbf16>> -> tensor<16x21x384xbf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x192xbf16>> -> tensor<384x192xbf16>
        %5 = tensor.empty() : tensor<16x21x192xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x21x192xf32>) -> tensor<16x21x192xf32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<16x21x384xbf16>, tensor<384x192xbf16>) outs(%6 : tensor<16x21x192xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %10 = arith.extf %in : bf16 to f32
          %11 = arith.extf %in_0 : bf16 to f32
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %out, %12 : f32
          linalg.yield %13 : f32
        } -> tensor<16x21x192xf32>
        %8 = tensor.empty() : tensor<16x21x192xbf16>
        %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<16x21x192xf32>) outs(%8 : tensor<16x21x192xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %10 = arith.truncf %in : f32 to bf16
          linalg.yield %10 : bf16
        } -> tensor<16x21x192xbf16>
        iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0, 0], sizes = [16, 21, 192], strides = [1, 1, 1] : tensor<16x21x192xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x21x192xbf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @conv_2d_input_backward_16x1x21x192_nhwc
// CHECK-DAG: memref.alloc() : memref<64x68xbf16, #gpu.address_space<workgroup>>
// CHECK-DAG: memref.alloc() : memref<2x32x68xbf16, #gpu.address_space<workgroup>>
// CHECK-NOT: memref.alloca

// -----

// Test NHWC conv with small channels (C=3) and padding_conv.
// This exercises the full iGEMM pipeline with spatial padding:
//   GPUPadConvsPass → ConvolutionToIGEMM → FoldInputPadIntoIm2col →
//   tiling → VectorizeIm2colOp (with padding masks) → MFMA
// The key verification points are:
//   1. No tensor.pad ops remain (folded into im2col)
//   2. No iree_linalg_ext.im2col ops remain (vectorized)
//   3. Masked vector.transfer_read with arith.cmpi bounds checks
//   4. MFMA instructions are generated
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
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  padding = [1, 128, 64, 32],
  workgroup = [1, 128, 64, 0],
  reduction = [0, 0, 0, 2],
  subgroup = [1, 2, 2, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_nhwc_small_channel ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_small_channel() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x26x19x3xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x3x64xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x24x17x64xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 26, 19, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x26x19x3xf16>> -> tensor<16x26x19x3xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x3x64xf16>> -> tensor<3x3x3x64xf16>
        %5 = tensor.empty() : tensor<16x24x17x64xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x24x17x64xf32>) -> tensor<16x24x17x64xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<16x26x19x3xf16>, tensor<3x3x3x64xf16>) outs(%6 : tensor<16x24x17x64xf32>) -> tensor<16x24x17x64xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [16, 24, 17, 64], strides = [1, 1, 1, 1] : tensor<16x24x17x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x24x17x64xf32>>
        return
      }
    }
  }
}

// Verify padding was folded into im2col and im2col was vectorized:
//   - No tensor.pad or im2col ops remain
//   - Bounds checking with arith.cmpi
//   - MFMA instructions generated for the GEMM
//    CHECK-LABEL: func @conv_nhwc_small_channel
//      CHECK-NOT:   tensor.pad
//      CHECK-NOT:   iree_linalg_ext.im2col
//      CHECK-DAG:   memref.alloc() : memref<32x68xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<1x128x36xf16, #gpu.address_space<workgroup>>
//          CHECK:   scf.forall ({{.*}}) in (16, 4) {
//          CHECK:     vector.transfer_read {{.*}} : memref<16x26x19x3xf16, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:     gpu.barrier
//  CHECK-COUNT-8:     amdgpu.mfma 16x16x16
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

// Test NHWC conv with stride=2 and aligned channels.
// No spatial padding needed (channel count 128 is multiple of MFMA K tile).
// This verifies the non-padded im2col vectorization path (no masks).
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
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 64, 128, 0],
  reduction = [0, 0, 0, 2],
  subgroup = [1, 4, 2, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_nhwc_stride2_no_pad ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_stride2_no_pad() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x128xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x16x16x128xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x128xf16>> -> tensor<3x3x128x128xf16>
        %5 = tensor.empty() : tensor<2x16x16x128xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x16x16x128xf32>) -> tensor<2x16x16x128xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<2x34x34x128xf16>, tensor<3x3x128x128xf16>) outs(%6 : tensor<2x16x16x128xf32>) -> tensor<2x16x16x128xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 16, 128], strides = [1, 1, 1, 1] : tensor<2x16x16x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x16x16x128xf32>>
        return
      }
    }
  }
}

// Non-padded conv: im2col vectorized without masks.
// No tensor.pad, no im2col, no spatial bounds checking.
//    CHECK-LABEL: func @conv_nhwc_stride2_no_pad
//      CHECK-NOT:   tensor.pad
//      CHECK-NOT:   iree_linalg_ext.im2col
//      CHECK-DAG:   memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   memref.alloc() : memref<32x132xf16, #gpu.address_space<workgroup>>
//          CHECK:   scf.forall ({{.*}}) in (2, 4) {
//          CHECK:     scf.for {{.*}} iter_args({{.*}}) -> (vector<1x4x2x4x1xf32>)
//          CHECK:       gpu.barrier
//      CHECK-DAG:       vector.transfer_read {{.*}}#amdgpu.address_space<fat_raw_buffer>{{.*}} vector<8xf16>
//      CHECK-DAG:       vector.transfer_write {{.*}} vector<8xf16>
//          CHECK:       gpu.barrier
// CHECK-COUNT-16:       amdgpu.mfma 16x16x16
//          CHECK:     %[[LOOP_T:.+]] = vector.transpose {{.*}} [0, 1, 3, 2, 4] : vector<1x4x2x4x1xf32> to vector<1x4x4x2x1xf32>
//          CHECK:     %[[CAST:.+]] = vector.shape_cast %[[LOOP_T]] : vector<1x4x4x2x1xf32> to vector<4x4x2x1xf32>
//          CHECK:     vector.transfer_write %[[CAST]]
//          CHECK:   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
