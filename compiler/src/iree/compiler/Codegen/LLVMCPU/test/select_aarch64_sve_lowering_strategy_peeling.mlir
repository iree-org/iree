// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-vector-pproc-strategy=peel \
// RUN:   --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @matmul_tensors() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
    %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
    %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//       CHECK: func.func @matmul_tensors()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @static_tensors_non_pow_two_sizes() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<15x14xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<14x7xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 14], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<15x14xf32>> -> tensor<15x14xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [14, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<14x7xf32>> -> tensor<14x7xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>> -> tensor<15x7xf32>
    %6 = linalg.matmul ins(%3, %4 : tensor<15x14xf32>, tensor<14x7xf32>) outs(%5 : tensor<15x7xf32>) -> tensor<15x7xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : tensor<15x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<15x7xf32>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[5, 7, 0], [5, 7, 0], [0, 0, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//       CHECK: func.func @static_tensors_non_pow_two_sizes()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @static_tensors_1x1() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
    %6 = linalg.matmul ins(%3, %4 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
    return
  }
}

// TODO: FIXME - scalable "16" ([16]) for just 1 element
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, [16], 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: func.func @static_tensors_1x1()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
module {
  func.func @depthwise_conv() attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x1x4x4xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x4x4xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x1x1x4xf32>>
    %input = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 1, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x4x4xf32>> -> tensor<1x1x4x4xf32>
    %filter = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 4, 4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4x4xf32>> -> tensor<1x4x4xf32>
    %5 = tensor.empty() : tensor<1x1x1x4xf32>
    %output = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x4x4xf32>) outs(%output : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1] : tensor<1x1x1x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x1x4xf32>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 2, 0, 0], [1, 1, 1, 2, 0, 0], [0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert, {{\{}}enable_peeling = true}>
//      CHECK: func.func @depthwise_conv()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]
