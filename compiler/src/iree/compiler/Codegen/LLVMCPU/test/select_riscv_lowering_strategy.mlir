// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-llvmcpu-riscv-aggressive-distribution=true --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s -check-prefixes=CHECK-AGGRESSIVE

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_riscv_32_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_32", {cpu_features = "+m,+f", data_layout = "e-m:e-p:32:32-i64:64-n32-S128", native_vector_size = 16 : index, target_triple = "riscv32-none-elf"}>
func.func @matmul_riscv() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_32_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
  %5 = tensor.empty() : tensor<384x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[48, 64], [8, 32], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[48, 64, 0], [48, 64, 0], [0, 0, 0], [8, 32, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_riscv()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+zvl512b,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 128 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @matmul_gemm_riscv_vl512() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %cst = arith.constant 0.0 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
    %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
    %init = tensor.empty() : tensor<384x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
    %res = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>) outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
    iree_tensor_ext.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [7, 64], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [7, 64, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_gemm_riscv_vl512()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+zvl1024b,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 256 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @matmul_gemm_riscv_vl1024() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %cst = arith.constant 0.0 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32>>
    %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
    %init = tensor.empty() : tensor<384x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x256xf32>) -> tensor<384x256xf32>
    %res = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x256xf32>) outs(%fill : tensor<384x256xf32>) -> tensor<384x256xf32>
    iree_tensor_ext.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [384, 256], strides = [1, 1] : tensor<384x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128], [7, 128], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128, 0], [64, 128, 0], [0, 0, 0], [7, 128, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_gemm_riscv_vl1024()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

//  CHECK-AGGRESSIVE-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 256], [7, 128], [0, 0], [0, 0]]>
//  CHECK-AGGRESSIVE-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 256, 0], [32, 256, 0], [0, 0, 0], [7, 128, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-AGGRESSIVE-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK-AGGRESSIVE: func.func @matmul_gemm_riscv_vl1024()
// CHECK-AGGRESSIVE-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK-AGGRESSIVE: linalg.matmul
// CHECK-AGGRESSIVE-SAME:     lowering_config = #[[CONFIG2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+zvl512b,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 128 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @matmul_gemv_riscv_vl512() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %cst = arith.constant 0.0 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128xf32>>
    %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x512xf32>> -> tensor<1x512xf32>
    %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
    %init = tensor.empty() : tensor<1x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x128xf32>) -> tensor<1x128xf32>
    %res = linalg.matmul ins(%lhs, %rhs : tensor<1x512xf32>, tensor<512x128xf32>) outs(%fill : tensor<1x128xf32>) -> tensor<1x128xf32>
    iree_tensor_ext.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [1, 128], strides = [1, 1] : tensor<1x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 128], [1, 128], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 128, 0], [0, 128, 0], [0, 0, 0], [1, 128, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_gemv_riscv_vl512()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_riscv_32_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_32", {cpu_features = "+m,+f", data_layout = "e-m:e-p:32:32-i64:64-n32-S128", native_vector_size = 16 : index, target_triple = "riscv32-none-elf"}>
func.func @thin_depthwise_conv_static() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_32_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x72xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
  %5 = tensor.empty() : tensor<1x28x28x72xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1] : tensor<1x28x28x72xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 28, 28, 8, 0, 0], [1, 1, 4, 4, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @thin_depthwise_conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]
