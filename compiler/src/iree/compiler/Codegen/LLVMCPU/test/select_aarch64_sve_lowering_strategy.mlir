// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file --iree-llvmcpu-disable-arm-sme-tiling %s | FileCheck %s --check-prefixes=CHECK,DISABLE-ARM-SME

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @matmul_tensors()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @static_tensors_non_pow_two_sizes() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<15x14xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<14x7xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 14], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<15x14xf32>> -> tensor<15x14xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [14, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<14x7xf32>> -> tensor<14x7xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>> -> tensor<15x7xf32>
  %6 = linalg.matmul ins(%3, %4 : tensor<15x14xf32>, tensor<14x7xf32>) outs(%5 : tensor<15x7xf32>) -> tensor<15x7xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : tensor<15x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<15x7xf32>>
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[5, 7, 0], [5, [8], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @static_tensors_non_pow_two_sizes()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @static_tensors_1x1() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
  %6 = linalg.matmul ins(%3, %4 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @static_tensors_1x1()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  return
}

//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      DISABLE-ARM-SME: func.func @matmul_tensors()
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], {{\[}}[8], [8], 0], [0, 0, 1], [0, 0, 0]]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       WITH-SME: func.func @matmul_tensors()
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @matmul_with_fill() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = arith.index_castui %0 : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.index_castui %3 : i32 to index
  %9 = arith.index_castui %4 : i32 to index
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x256xi8>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
  %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024xf32>>
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%9) : !flow.dispatch.tensor<writeonly:tensor<1024x256xf32>>
  %15 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x256xi8>> -> tensor<1024x256xi8>
  %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
  %17 = flow.dispatch.tensor.load %12, offsets = [0], sizes = [1024], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1024xf32>> -> tensor<1024xf32>
  %18 = flow.dispatch.tensor.load %13, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
  %19 = tensor.empty() : tensor<1024x256xf32>
  %20 = tensor.empty() : tensor<1024x256xi32>
  %21 = linalg.fill ins(%c0_i32 : i32) outs(%20 : tensor<1024x256xi32>) -> tensor<1024x256xi32>
  %22 = linalg.matmul ins(%15, %16 : tensor<1024x256xi8>, tensor<256x256xi8>) outs(%21 : tensor<1024x256xi32>) -> tensor<1024x256xi32>
  %23 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %17, %18 : tensor<1024x256xi32>, tensor<1024xf32>, tensor<256xf32>) outs(%19 : tensor<1024x256xf32>) {
  ^bb0(%in: i32, %in_0: f32, %in_1: f32, %out: f32):
    %24 = arith.sitofp %in : i32 to f32
    %25 = arith.mulf %24, %in_0 : f32
    %26 = arith.mulf %25, %in_1 : f32
    linalg.yield %26 : f32
  } -> tensor<1024x256xf32>
  flow.dispatch.tensor.store %23, %14, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : tensor<1024x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x256xf32>>
  return
}

// CHECK-DAG:  #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, [16]], [0, 0], [0, 0]]>
// CHECK-DAG:  #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
// CHECK:      func.func @matmul_with_fill()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
func.func @depthwise_conv() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
  %5 = tensor.empty() : tensor<1x28x28x72xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1] : tensor<1x28x28x72xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
  return
}

// CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 28, 28, 8, 0, 0], [1, 1, 4, [4], 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
// CHECK:      func.func @depthwise_conv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK:      linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]
