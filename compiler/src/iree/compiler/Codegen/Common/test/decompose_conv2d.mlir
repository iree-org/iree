// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-convolution-to-lower-dim-ops))" --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 4, 0, 0], [0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0]]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
#translation = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @restrict_num_workgroups() attributes {hal.executable.target = #executable_target_system_elf_arm_64_, translation_info = #translation} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x1x4x4xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1x4x4xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<1x1x1x4xf32>>
    %input = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 1, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x4x4xf32>> -> tensor<1x1x4x4xf32>
    %filter = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 4, 4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4x4xf32>> -> tensor<1x4x4xf32>
    %5 = tensor.empty() : tensor<1x1x1x4xf32>
    %output = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x4x4xf32>) outs(%output : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1] : tensor<1x1x1x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x1x4xf32>>
    return
  }
}

//   CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 0], [1, 1, 4, 0], [0, 0, 0, 4], [0, 0, 0, 0]]>
//   CHECK:    linalg.depthwise_conv_1d_nwc_wc
//   CHECK-SAME: lowering_config = #[[CONFIG]]
//   CHECK-SAME: ins({{.*}}, {{.*}} : tensor<1x4x4xf32>, tensor<4x4xf32>) outs({{.*}} : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
