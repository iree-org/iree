// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-lower-executable-target))' -split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling = true}>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64}>
func.func @no_peel_static_matmul() attributes {hal.executable.target = #executable_target_system_elf_x86_64_, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x64xf32>> -> tensor<128x64xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xf32>> -> tensor<64x512xf32>
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x64xf32>, tensor<64x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}

// CHECK-LABEL: func @no_peel_static_matmul()
// Vectorization:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               vector.fma
// CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_cpu.lowering_config<distribution = [65, 65, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling = true}>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64}>
func.func @peel_static_matmul() attributes {hal.executable.target = #executable_target_system_elf_x86_64_, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>> -> tensor<128x49xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [49, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>> -> tensor<49x512xf32>
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x49xf32>, tensor<49x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}

// CHECK-LABEL: func @peel_static_matmul()
// Vectorization:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               vector.fma

// 2nd dim peeling:
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

// 3rd dim peeling:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

// CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling = true}>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64}>
func.func @peel_dynamic_matmul() attributes {hal.executable.target = #executable_target_system_elf_x86_64_, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = arith.index_cast %0 : i32 to index
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.index_cast %2 : i32 to index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %3}
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %5}
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
  %9 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%4, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %3} -> tensor<?x?xf32>
  %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%3, %5], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %5} -> tensor<?x?xf32>
  %11 = tensor.empty(%4, %5) : tensor<?x?xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = linalg.matmul {lowering_config = #config} ins(%9, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %13, %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
  return
}

// CHECK-LABEL: func @peel_dynamic_matmul()
// Distribution:
// CHECK:         scf.forall

// Vectorization:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.fma

// 1nd dim peeling:
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// 2nd dim peeling:
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// 3nd dim peeling:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [8, [32], 0], vector_reduction = [0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling = true}>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @peel_scalable_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_, translation_info = #translation} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
    %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
    %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
    %3 = arith.index_cast %0 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.index_cast %2 : i32 to index
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %3}
    %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %5}
    %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
    %9 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%4, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %3} -> tensor<?x?xf32>
    %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%3, %5], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %5} -> tensor<?x?xf32>
    %11 = tensor.empty(%4, %5) : tensor<?x?xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %13 = linalg.matmul {lowering_config = #config} ins(%9, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
    iree_tensor_ext.dispatch.tensor.store %13, %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
    return
  }
}

// CHECK-LABEL: func @peel_scalable_matmul()

// Vectorization:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.fma

// 2nd dim peeling:
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.fma

// 3nd dim peeling:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.fma

// CHECK-NOT: scf.for
