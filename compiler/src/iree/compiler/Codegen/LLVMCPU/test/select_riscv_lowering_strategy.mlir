// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_riscv_32_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_32", {cpu_features = "+m,+f", data_layout = "e-m:e-p:32:32-i64:64-n32-S128", native_vector_size = 16 : index, target_triple = "riscv32-none-elf"}>
module {
  func.func @matmul_riscv() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_32_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
    %5 = tensor.empty() : tensor<384x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, 32], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 32, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_riscv()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----
#executable_target_embedded_elf_riscv_32_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_32", {cpu_features = "+m,+f", data_layout = "e-m:e-p:32:32-i64:64-n32-S128", native_vector_size = 16 : index, target_triple = "riscv32-none-elf"}>
module {
  func.func @thin_depthwise_conv_static() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_32_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
    %5 = tensor.empty() : tensor<1x28x28x72xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1] : tensor<1x28x28x72xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 7, 7, 72, 0, 0], [1, 1, 7, 4, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @thin_depthwise_conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

