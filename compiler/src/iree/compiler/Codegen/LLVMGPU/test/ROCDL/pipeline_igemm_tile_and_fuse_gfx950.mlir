// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target{for-rocdl=true})))))" %s | FileCheck %s

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
  workgroup = [1, 4, 16, 256, 0],
  reduction = [0, 0, 0, 0, 2],
  subgroup = [1, 4, 1, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @conv_nhwc_f16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_nhwc_f16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_f16() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>> -> tensor<2x34x34x1280xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x1280xf16>
        %5 = tensor.empty() : tensor<2x32x32x1280xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config} ins(%3, %4 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%6 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_nhwc_f16
//          CHECK:   scf.forall
//          CHECK:     scf.for {{.*}} iter_args
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<2x34x34x1280xf16, #amdgpu.address_space<fat_raw_buffer>>{{.*}}vector<8xf16>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<1x4x16x{{.*}}xf16, {{.*}}#gpu.address_space<workgroup>>
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<11520x1280xf16, #amdgpu.address_space<fat_raw_buffer>>{{.*}}vector<8xf16>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<64x{{.*}}xf16, {{.*}}#gpu.address_space<workgroup>>
//          CHECK:       gpu.barrier
//          CHECK:       vector.transfer_read {{.*}}#gpu.address_space<workgroup>
//          CHECK:       amdgpu.transpose_load {{.*}}#gpu.address_space<workgroup>{{.*}}vector<4xf16>
//          CHECK:       amdgpu.mfma 16x16x32 {{.*}} vector<8xf16>, vector<8xf16>, vector<4xf32>
//          CHECK:       scf.yield

// -----

#pipeline_layout_unaligned = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_unaligned = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config_unaligned = #iree_gpu.lowering_config<{
  padding = [2, 1, 32, 16, 32],
  workgroup = [2, 1, 32, 16, 0],
  reduction = [0, 0, 0, 0, 1],
  subgroup = [1, 1, 1, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1]
}>
hal.executable private @conv_nhwc_unaligned_f16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_nhwc_unaligned_f16 ordinal(0) layout(#pipeline_layout_unaligned) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc_unaligned_f16() attributes {translation_info = #translation_unaligned} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout_unaligned) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x35x35x1281xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout_unaligned) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1281x1281xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout_unaligned) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x17x17x1281xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 35, 35, 1281], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x35x35x1281xf16>> -> tensor<2x35x35x1281xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1281, 1281], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1281x1281xf16>> -> tensor<3x3x1281x1281xf16>
        %5 = tensor.empty() : tensor<2x17x17x1281xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x17x17x1281xf32>) -> tensor<2x17x17x1281xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, lowering_config = #config_unaligned} ins(%3, %4 : tensor<2x35x35x1281xf16>, tensor<3x3x1281x1281xf16>) outs(%6 : tensor<2x17x17x1281xf32>) -> tensor<2x17x17x1281xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 17, 17, 1281], strides = [1, 1, 1, 1] : tensor<2x17x17x1281xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x17x17x1281xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_nhwc_unaligned_f16
//          CHECK:   scf.forall
//          CHECK:     scf.for {{.*}} iter_args
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<2x35x35x1281xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<2x1x32x{{.*}}xf16, {{.*}}#gpu.address_space<workgroup>>
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<11529x1281xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<32x{{.*}}xf16, {{.*}}#gpu.address_space<workgroup>>
//          CHECK:       gpu.barrier
//          CHECK:       vector.transfer_read {{.*}}#gpu.address_space<workgroup>
//          CHECK:       amdgpu.transpose_load {{.*}}#gpu.address_space<workgroup>{{.*}}vector<4xf16>
//          CHECK:       amdgpu.mfma 16x16x32 {{.*}} vector<8xf16>, vector<8xf16>, vector<4xf32>
//          CHECK:       scf.yield

// -----

#pipeline_layout_backward = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly">,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_backward = #iree_codegen.translation_info<pipeline =
  LLVMGPUTileAndFuse
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_num_stages = 0,
       no_reduce_shared_memory_bank_conflicts = false,
       use_igemm_convolution = true>
  }>
#config_backward = #iree_gpu.lowering_config<{
  padding = [2, 32, 64, 64],
  workgroup = [2, 32, 64, 0],
  reduction = [0, 0, 0, 2],
  subgroup = [2, 2, 1, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_BF16>,
  promote_operands = [0, 1]
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
hal.executable private @conv_input_backward_bf16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_input_backward_bf16 ordinal(0) layout(#pipeline_layout_backward) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_input_backward_bf16() attributes {translation_info = #translation_backward} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout_backward) binding(0) alignment(64) offset(%c0) flags("ReadOnly") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x21x384xbf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout_backward) binding(1) alignment(64) offset(%c0) flags("ReadOnly") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x192xbf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout_backward) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x21x192xbf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 21, 384], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x21x384xbf16>> -> tensor<16x21x384xbf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x192xbf16>> -> tensor<384x192xbf16>
        %5 = tensor.empty() : tensor<16x21x192xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x21x192xf32>) -> tensor<16x21x192xf32>
        %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<16x21x384xbf16>, tensor<384x192xbf16>) outs(%6 : tensor<16x21x192xf32>) attrs =  {lowering_config = #config_backward} {
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

//    CHECK-LABEL: func @conv_input_backward_bf16
//          CHECK:   scf.forall
//          CHECK:     scf.for {{.*}} iter_args
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<16x21x384xbf16, #amdgpu.address_space<fat_raw_buffer>>{{.*}}vector<8xbf16>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<2x32x{{.*}}xbf16, {{.*}}#gpu.address_space<workgroup>>
//      CHECK-DAG:       vector.transfer_read {{.*}}memref<384x192xbf16, #amdgpu.address_space<fat_raw_buffer>>{{.*}}vector<8xbf16>
//      CHECK-DAG:       vector.transfer_write {{.*}}memref<64x{{.*}}xbf16, {{.*}}#gpu.address_space<workgroup>>
//          CHECK:       gpu.barrier
//          CHECK:       vector.transfer_read {{.*}}#gpu.address_space<workgroup>
//          CHECK:       amdgpu.transpose_load {{.*}}#gpu.address_space<workgroup>{{.*}}vector<4xbf16>
//          CHECK:       amdgpu.mfma 16x16x32 {{.*}} vector<8xbf16>, vector<8xbf16>, vector<4xf32>
//          CHECK:       scf.yield
