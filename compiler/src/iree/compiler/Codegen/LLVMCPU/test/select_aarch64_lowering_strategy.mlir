// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s
// Test the same lowering strategy selection on generic convolution ops by first
// generalizing the named ops. This ensures convolution pipeline selection works
// on both named and generic convs.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops),iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s --check-prefix=GENERIC

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_default(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors_default(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @i4_i4_i32_matmul(%7: tensor<?x?xi4>, %8: tensor<?x?xi4>, %9: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xi4>, tensor<?x?xi4>) outs(%9 : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %10 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @i4_i4_i32_matmul(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @batch_matmul_tensors(%7: tensor<?x?x?xf32>, %8: tensor<?x?x?xf32>, %0: index, %1: index, %2: index) -> tensor<?x?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %9 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = linalg.batch_matmul ins(%7, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%10 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %11 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [1, 64, 64, 0], vector_common_parallel = [1, 8, 16, 0], vector_reduction = [0, 0, 0, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @batch_matmul_tensors(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_static(%3: tensor<196x240xf32>, %4: tensor<240x40xf32>) -> tensor<196x40xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<196x40xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<196x40xf32>) -> tensor<196x40xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<196x240xf32>, tensor<240x40xf32>) outs(%6 : tensor<196x40xf32>) -> tensor<196x40xf32>
  return %7 : tensor<196x40xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [14, 40, 0], distribution = [14, 40, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_static(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @conv_static(%3: tensor<1x51x41x512xf32>, %4: tensor<3x3x512x512xf32>) -> tensor<1x25x20x512xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x25x20x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x51x41x512xf32>, tensor<3x3x512x512xf32>) outs(%6 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
  return %7 : tensor<1x25x20x512xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 5, 20, 64, 0, 0, 0], vector_common_parallel = [1, 1, 2, 2, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//       GENERIC: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 5, 20, 64, 0, 0, 0], vector_common_parallel = [1, 1, 2, 2, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 16]>
//       GENERIC: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert
//       GENERIC: func.func @conv_static(
//  GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//       GENERIC:     linalg.generic
//       GENERIC:         lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @restrict_num_workgroups(%3: tensor<1x11x11x576xf32>, %4: tensor<5x5x576xf32>) -> tensor<1x7x7x576xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x7x7x576xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%6 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  return %7 : tensor<1x7x7x576xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0], vector_common_parallel = [1, 1, 1, 4, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//       CHECK: func.func @restrict_num_workgroups(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
//       GENERIC: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0], vector_common_parallel = [1, 1, 1, 4, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1]>
//       GENERIC: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert
//       GENERIC: func.func @restrict_num_workgroups(
//  GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//       GENERIC: linalg.generic
//       GENERIC:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_aarch_i8_i8_i32_static(%3: tensor<128x384xi8>, %4: tensor<384x1536xi8>) -> tensor<128x1536xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<128x1536xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x384xi8>, tensor<384x1536xi8>) outs(%6 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  return %7 : tensor<128x1536xi32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_aarch_i8_i8_i32_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @matmul_aarch_i8_i8_i32_dynamic(%6: tensor<?x?xi8>, %7: tensor<?x?xi8>, %8: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %9 = linalg.matmul ins(%6, %7 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%8 : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %9 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_aarch_i8_i8_i32_dynamic(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @pack(%2: tensor<20x48xf32>) -> tensor<3x48x8x1xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<3x48x8x1xf32>
  %pack = linalg.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<20x48xf32> -> tensor<3x48x8x1xf32>
  return %pack : tensor<3x48x8x1xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 16], vector_common_parallel = [1, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling, {enable_decomposition}>
//       CHECK: func.func @pack(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.pack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @unpack_outer_dynamic(%10: tensor<?x?x32x16xi32>, %6: index, %7: index) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
  %unpack = linalg.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [32, 16]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling, {enable_decomposition}>
//       CHECK: func.func @unpack_outer_dynamic(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @unpack_fully_dynamic(%14: tensor<?x?x?x?xi32>, %8: index, %9: index, %10: index, %11: index) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %15 = tensor.empty(%8, %9) : tensor<?x?xi32>
  %unpack = linalg.unpack %14 inner_dims_pos = [0, 1] inner_tiles = [%10, %11] into %15 : tensor<?x?x?x?xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [1, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//       CHECK: func.func @unpack_fully_dynamic(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @mmt4d_384x384x512_4x1x4_dispatch_0(%3: tensor<96x384x4x1xf32>, %4: tensor<128x384x4x1xf32>, %5: tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %6 = linalg.mmt4d ins(%3, %4 : tensor<96x384x4x1xf32>, tensor<128x384x4x1xf32>) outs(%5 : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
  return %6 : tensor<96x128x4x4xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [16, 16, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 4, 4, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
//       CHECK: func.func @mmt4d_384x384x512_4x1x4_dispatch_0(
//       CHECK:   linalg.mmt4d
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
