// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" --iree-codegen-llvmgpu-enable-implicit-gemm | FileCheck %s

hal.executable @nchw_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @nchw_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @nchw_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x128x258x258xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128x3x3xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 128, 258, 258], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x128x258x258xf32>> -> tensor<8x128x258x258xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [256, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128x3x3xf32>> -> tensor<256x128x3x3xf32>
      %5 = tensor.empty() : tensor<8x256x256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x128x258x258xf32>, tensor<256x128x3x3xf32>) outs(%6 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 256], strides = [1, 1, 1, 1] : tensor<8x256x256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      return
    }
  }
}
}

//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUImplicitGEMM>
//       CHECK: hal.executable.export public @nchw_convolution
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-LABEL: func @nchw_convolution

// -----

hal.executable @nhwc_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @nhwc_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @nhwc_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x258x258x128xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 258, 258, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x258x258x128xf32>> -> tensor<8x258x258x128xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>> -> tensor<3x3x128x256xf32>
      %5 = tensor.empty() : tensor<8x256x256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x258x258x128xf32>, tensor<3x3x128x256xf32>) outs(%6 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 256], strides = [1, 1, 1, 1] : tensor<8x256x256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      return
    }
  }
}
}
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUImplicitGEMM>
//       CHECK: hal.executable.export public @nhwc_convolution
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-LABEL: func @nhwc_convolution

// -----

hal.executable @unaligned_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @unaligned_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @unaligned_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x258x258x132xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x132x264xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x264xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 258, 258, 132], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x258x258x132xf32>> -> tensor<8x258x258x132xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 132, 264], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x132x264xf32>> -> tensor<3x3x132x264xf32>
      %5 = tensor.empty() : tensor<8x256x256x264xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x264xf32>) -> tensor<8x256x256x264xf32>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x258x258x132xf32>, tensor<3x3x132x264xf32>) outs(%6 : tensor<8x256x256x264xf32>) -> tensor<8x256x256x264xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 264], strides = [1, 1, 1, 1] : tensor<8x256x256x264xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x264xf32>>
      return
    }
  }
}
}

//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUImplicitGEMM>
//       CHECK: hal.executable.export public @unaligned_convolution
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-LABEL: func @unaligned_convolution
