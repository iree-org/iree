// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-vector-pproc-strategy=peel \
// RUN:   --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Check tile sizes depend on output element type - altering output element type changes tile sizes.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_f16(%7: tensor<?x?xf16>, %8: tensor<?x?xf16>, %9: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %10 : tensor<?x?xf16>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, [16], 0], vector_reduction = [0, 0, 8]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors_f16(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Check tile sizes depend on output element type - keeping output element type keeps tile sizes.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_i8i8_i32(%7: tensor<?x?xi8>, %8: tensor<?x?xi8>, %9: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%9 : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %10 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors_i8i8_i32(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @static_tensors_non_pow_two_sizes(%3: tensor<15x14xf32>, %4: tensor<14x7xf32>, %5: tensor<15x7xf32>) -> tensor<15x7xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %6 = linalg.matmul ins(%3, %4 : tensor<15x14xf32>, tensor<14x7xf32>) outs(%5 : tensor<15x7xf32>) -> tensor<15x7xf32>
  return %6 : tensor<15x7xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [5, 7, 0], distribution = [5, 7, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @static_tensors_non_pow_two_sizes(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @static_tensors_1x1(%3: tensor<1x1xf32>, %4: tensor<1x1xf32>, %5: tensor<1x1xf32>) -> tensor<1x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %6 = linalg.matmul ins(%3, %4 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %6 : tensor<1x1xf32>
}
// TODO: FIXME - scalable "16" ([16]) for just 1 element
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, [8], 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @static_tensors_1x1(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android30"}>
func.func @depthwise_conv(%input: tensor<1x1x4x4xf32>, %filter: tensor<1x4x4xf32>) -> tensor<1x1x1x4xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x1x1x4xf32>
  %output = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
          strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x4x4xf32>) outs(%output : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
  return %7 : tensor<1x1x1x4xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0, 1, 0, 0], vector_common_parallel = [1, 1, 1, 1, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 4]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @depthwise_conv(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]
