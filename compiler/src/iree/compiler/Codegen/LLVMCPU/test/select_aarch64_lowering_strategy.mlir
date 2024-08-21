// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_default() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors_default()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @i4_i4_i32_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xi4>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xi4>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%0, %1}
  %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi4>>{%0, %2} -> tensor<?x?xi4>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi4>>{%2, %1} -> tensor<?x?xi4>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xi4>, tensor<?x?xi4>) outs(%9 : tensor<?x?xi32>) -> tensor<?x?xi32>
  flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%0, %1}
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 32, 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @i4_i4_i32_matmul()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @batch_matmul_tensors() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %1, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %3, %2}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%0, %1, %2}
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%0, %1, %3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %1, %3} -> tensor<?x?x?xf32>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%0, %3, %2], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %3, %2} -> tensor<?x?x?xf32>
  %9 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = linalg.batch_matmul ins(%7, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%10 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0], sizes = [%0, %1, %2], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%0, %1, %2}
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64, 0], [1, 64, 64, 0], [0, 0, 0, 0], [1, 8, 16, 0], [0, 0, 0, 1], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @batch_matmul_tensors()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_static() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<196x240xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<240x40xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [196, 240], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<196x240xf32>> -> tensor<196x240xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [240, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<240x40xf32>> -> tensor<240x40xf32>
  %5 = tensor.empty() : tensor<196x40xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<196x40xf32>) -> tensor<196x40xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<196x240xf32>, tensor<240x40xf32>) outs(%6 : tensor<196x40xf32>) -> tensor<196x40xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [196, 40], strides = [1, 1] : tensor<196x40xf32> -> !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[14, 40, 0], [14, 40, 0], [0, 0, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_static()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @conv_static() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c607520 = arith.constant 607520 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x51x41x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(32) offset(%c607520) : !flow.dispatch.tensor<readonly:tensor<3x3x512x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x25x20x512xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 51, 41, 512], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x51x41x512xf32>> -> tensor<1x51x41x512xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 512, 512], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x512x512xf32>> -> tensor<3x3x512x512xf32>
  %5 = tensor.empty() : tensor<1x25x20x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x51x41x512xf32>, tensor<3x3x512x512xf32>) outs(%6 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 25, 20, 512], strides = [1, 1, 1, 1] : tensor<1x25x20x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x25x20x512xf32>>
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 5, 20, 64, 0, 0, 0], [1, 1, 2, 2, 0, 0, 0], [0, 0, 0, 0, 1, 1, 16], [0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @restrict_num_workgroups() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 11, 11, 576], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>> -> tensor<1x11x11x576xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [5, 5, 576], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>> -> tensor<5x5x576xf32>
  %5 = tensor.empty() : tensor<1x7x7x576xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%6 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 576], strides = [1, 1, 1, 1] : tensor<1x7x7x576xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 7, 7, 64, 0, 0], [1, 1, 1, 4, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//       CHECK: func.func @restrict_num_workgroups()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_aarch_i8_i8_i32_static() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xi8>> -> tensor<128x384xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>> -> tensor<384x1536xi8>
  %5 = tensor.empty() : tensor<128x1536xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x384xi8>, tensor<384x1536xi8>) outs(%6 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : tensor<128x1536xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_aarch_i8_i8_i32_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_aarch_i8_i8_i32_dynamic() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(32) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1}
  %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%0, %2} -> tensor<?x?xi8>
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%2, %1} -> tensor<?x?xi8>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
  %9 = linalg.matmul ins(%6, %7 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%8 : tensor<?x?xi32>) -> tensor<?x?xi32>
  flow.dispatch.tensor.store %9, %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1}
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_aarch_i8_i8_i32_dynamic()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @pack() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x48x8x1xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf32>> -> tensor<20x40xf32>
  %3 = tensor.empty() : tensor<4x48x8x1xf32>
  %pack = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<20x40xf32> -> tensor<4x48x8x1xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 48, 8, 1], strides = [1, 1, 1, 1] : tensor<4x48x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x48x8x1xf32>>
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16], [1, 1]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//       CHECK: func.func @pack()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   tensor.pack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @unpack_outer_dynamic() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
  %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
  %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
  %unpack = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  flow.dispatch.tensor.store %unpack, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [32, 16]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//       CHECK: func.func @unpack_outer_dynamic()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   tensor.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @mmt4d_384x384x512_4x1x4_dispatch_0() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c96 = arith.constant 96 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>> -> tensor<96x384x4x1xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 384, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>> -> tensor<128x384x4x1xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>> -> tensor<96x128x4x4xf32>
  %6 = linalg.mmt4d ins(%3, %4 : tensor<96x384x4x1xf32>, tensor<128x384x4x1xf32>) outs(%5 : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [96, 128, 4, 4], strides = [1, 1, 1, 1] : tensor<96x128x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16, 0, 0, 0, 0], [1, 1, 0, 4, 4, 0], [0, 0, 1, 0, 0, 1]]
//       CHECK: func.func @mmt4d_384x384x512_4x1x4_dispatch_0()
//       CHECK:   linalg.mmt4d
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
