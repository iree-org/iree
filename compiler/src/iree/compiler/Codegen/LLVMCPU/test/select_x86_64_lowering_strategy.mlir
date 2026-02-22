// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s
// Test the same lowering strategy selection on generic convolution ops by first
// generalizing the named ops. This ensures convolution pipeline selection works
// on both named and generic convs.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops),iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s --check-prefix=GENERIC

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @matvec_static(%3: tensor<128x384xf32>, %4: tensor<384xf32>) -> tensor<128xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%3, %4 : tensor<128x384xf32>, tensor<384xf32>) outs(%6 : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<128xf32>
  return %7 : tensor<128xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [16, 0], vector_reduction = [0, 16]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @matvec_static(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @matvec_dynamic(%11: tensor<?xf32>, %12: tensor<?x?xf32>, %13: tensor<?xf32>) -> tensor<?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %14 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
  %15 = linalg.matvec ins(%12, %13 : tensor<?x?xf32>, tensor<?xf32>) outs(%14 : tensor<?xf32>) -> tensor<?xf32>
  return %15 : tensor<?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 0], vector_common_parallel = [16, 0], vector_reduction = [0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @matvec_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matvec
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @dot_static(%3: tensor<384xf32>, %4: tensor<384xf32>) -> tensor<f32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<f32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
  %7 = linalg.dot ins(%3, %4 : tensor<384xf32>, tensor<384xf32>) outs(%6 : tensor<f32>) -> tensor<f32>
  return %7 : tensor<f32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0], vector_reduction = [16]
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @dot_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @dot_dynamic(%5: tensor<f32>, %8: tensor<?xf32>, %9: tensor<?xf32>) -> tensor<f32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %10 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
  %11 = linalg.dot ins(%8, %9 : tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<f32>) -> tensor<f32>
  return %11 : tensor<f32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0], vector_reduction = [16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @dot_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @dynamic_add(%0: index, %1: index, %5: tensor<?x?xf32>, %6: tensor<?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<?x?xf32>, tensor<?xf32>) outs(%7 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %9 = arith.addf %in, %in_0 : f32
    linalg.yield %9 : f32
  } -> tensor<?x?xf32>
  return %8 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @dynamic_add(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @add4D(%0: index, %1: index, %2: index, %3: index, %7: tensor<?x?x?x?xf32>, %8: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
  %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.addf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 64, 64], vector_common_parallel = [1, 1, 1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @add4D(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @add_static(%2: tensor<64x16x32x128xf32>) -> tensor<64x16x32x128xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<64x16x32x128xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64x16x32x128xf32>) outs(%3 : tensor<64x16x32x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.addf %in, %in : f32
    linalg.yield %5 : f32
  } -> tensor<64x16x32x128xf32>
  return %4 : tensor<64x16x32x128xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [8, 16, 32, 64], vector_common_parallel = [1, 1, 1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @add_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#config = #iree_cpu.lowering_config<distribution = [64, 64, 0], cache_parallel = [64, 64, 0], vector_common_parallel = [32, 32, 0], vector_reduction = [0, 0, 32]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling}>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @preset_config_matmul_tensors(%3: tensor<128x256xf32>, %4: tensor<256x512xf32>) -> tensor<128x512xf32> attributes {
    hal.executable.target = #executable_target_system_elf_x86_64_,
    translation_info = #translation
  } {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %7 : tensor<128x512xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [32, 32, 0], vector_reduction = [0, 0, 32]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @preset_config_matmul_tensors(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @matmul_partially_peel(%3: tensor<16641x16xf32>, %4: tensor<16x8xf32>) -> tensor<16641x8xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<16641x8xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16641x8xf32>) -> tensor<16641x8xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<16641x16xf32>, tensor<16x8xf32>) outs(%6 : tensor<16641x8xf32>) -> tensor<16641x8xf32>
  return %7 : tensor<16641x8xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [43, 8, 0], vector_common_parallel = [3, 8, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @matmul_partially_peel(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_op_dynamic(%6 : memref<?x?xi32>, %subview : memref<?x?xi32, strided<[?, 1], offset: ?>>) attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<?x?xi32>) outs(%subview : memref<?x?xi32, strided<[?, 1], offset: ?>>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUBufferOpsTileAndVectorize>
//      CHECK: func.func @copy_op_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @static_1d_fft_stage2(%2: tensor<32xf32>, %3: tensor<32xf32>) -> tensor<32xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
  %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
  %4:2 = iree_linalg_ext.fft ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  return %4 : tensor<32xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//       CHECK: func.func @static_1d_fft_stage2(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @static_3d_fft_stage3(%1: memref<4xf32>, %0: memref<4xf32>, %2: memref<64x128x32xf32>, %3: memref<64x128x32xf32>) attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %c3 = arith.constant 3 : index
  iree_linalg_ext.fft ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
  return
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [8, 64, 64]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//       CHECK: func.func @static_3d_fft_stage3(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @outs_fusion_fn(%0: index, %1: index, %2: index, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<?x?xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<?x?xf32>
  %10 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.mulf %in, %in_0 : f32
    %12 = arith.addf %11, %out : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 4]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 32, 0], vector_common_parallel = [1, 4, 0], vector_reduction = [0, 0, 4]>
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @outs_fusion_fn(
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//  CHECK-SAME:    lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @conv_dynamic(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %12: tensor<?x?x?x?xf32>, %13: tensor<?x?x?x?xf32>, %14: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%12, %13 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%14 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %15 : tensor<?x?x?x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<distribution = [64, 64, 64, 64, 0, 0, 0], vector_common_parallel = [1, 1, 1, 1, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//      CHECK:         lowering_config = #[[CONFIG]]
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 64, 64, 0, 0, 0], vector_common_parallel = [1, 1, 1, 1, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 1]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @conv_dynamic(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:         lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @conv_static(%3: tensor<1x225x225x3xf32>, %4: tensor<3x3x3x16xf32>) -> tensor<1x112x112x16xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x112x112x16xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  return %7 : tensor<1x112x112x16xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 16, 56, 16, 0, 0, 0], vector_common_parallel = [1, 1, 4, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 3]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 16, 56, 16, 0, 0, 0], vector_common_parallel = [1, 1, 4, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 3]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @conv_static(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:         lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @conv_nchw_static(%3: tensor<1x128x30x30xf32>, %4: tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x128x28x28xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%6 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  return %7 : tensor<1x128x28x28xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 64, 28, 4, 0, 0, 0], vector_common_parallel = [1, 4, 1, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 8, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_nchw_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nchw_fchw
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 64, 28, 4, 0, 0, 0], vector_common_parallel = [1, 4, 1, 4, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 8, 1, 1]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @conv_nchw_static(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:         lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @depthwise_conv_static(%3: tensor<1x161x161x240xf32>, %4: tensor<3x3x240xf32>) -> tensor<1x80x80x240xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x80x80x240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x161x161x240xf32>, tensor<3x3x240xf32>) outs(%6 : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
  return %7 : tensor<1x80x80x240xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 40, 40, 16, 0, 0], vector_common_parallel = [1, 1, 4, 16, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @depthwise_conv_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 40, 40, 16, 0, 0], vector_common_parallel = [1, 1, 4, 16, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @depthwise_conv_static(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @thin_depthwise_conv_static(%3: tensor<1x57x57x72xf32>, %4: tensor<3x3x72xf32>) -> tensor<1x28x28x72xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x28x28x72xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  return %7 : tensor<1x28x28x72xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 4, 28, 36, 0, 0], vector_common_parallel = [1, 1, 7, 12, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @thin_depthwise_conv_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 4, 28, 36, 0, 0], vector_common_parallel = [1, 1, 7, 12, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @thin_depthwise_conv_static(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vnni,+adx,+clflushopt,+clwb,+cx16,+cx8,+crc32,+f16c,+fsgsbase,+fxsr,+invpcid,+lzcnt,+movbe,+pku,+prfchw,+rdrnd,+rdseed,+sahf,+x87,+xsave,+xsavec,+xsaveopt,+xsaves", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf", ukernels = false}>
func.func @pooling_nchw_max(%2: tensor<1x64x114x114xf32>) -> tensor<1x64x56x56xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant -3.40282347E+38 : f32
  %3 = tensor.empty() : tensor<1x64x56x56xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %6 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%2, %4 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %6 : tensor<1x64x56x56xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 32, 56, 8, 0, 0], vector_common_parallel = [1, 8, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @pooling_nchw_max(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.pooling_nchw_max
// CHECK-SAME:       lowering_config  = #[[CONFIG]]
//  GENERIC-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 32, 56, 8, 0, 0], vector_common_parallel = [1, 8, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
//  GENERIC-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
//      GENERIC: func.func @pooling_nchw_max(
// GENERIC-SAME:     translation_info = #[[TRANSLATION]]
//      GENERIC:     linalg.generic
//      GENERIC:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-pc-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_static(%2: tensor<96x16xf32>) -> tensor<16x96xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %3 = tensor.empty() : tensor<16x96xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<96x16xf32>) outs(%3 : tensor<16x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<16x96xf32>
  return %4 : tensor<16x96xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [16, 96], vector_common_parallel = [16, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @generic_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @matmul_static(%3: tensor<384x512xf32>, %4: tensor<512x128xf32>) -> tensor<384x128xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<384x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
  return %7 : tensor<384x128xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [48, 64, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @matmul_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @predict_dispatch_86(%0: tensor<7x7x2048xf32>) -> tensor<7xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+01 : f32
    %1 = tensor.empty() : tensor<7xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<7xf32>) -> tensor<7xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%0 : tensor<7x7x2048xf32>) outs(%2 : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<7xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%3 : tensor<7xf32>) outs(%1 : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.divf %in, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<7xf32>
  return %4 : tensor<7xf32>
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 0, 0], vector_common_parallel = [1, 0, 0], vector_reduction = [0, 1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @predict_dispatch_86(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic {indexing_maps = [#{{.+}}, #{{.+}}], iterator_types = ["parallel", "reduction", "reduction"]}
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @matmul_i8_i8_i32_static(%3: tensor<128x384xi8>, %4: tensor<384x1536xi8>) -> tensor<128x1536xi32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<128x1536xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x384xi8>, tensor<384x1536xi8>) outs(%6 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
  return %7 : tensor<128x1536xi32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_i8_i8_i32_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @gemm_unit_N(%5: tensor<?x1xf32>, %6: tensor<?x?xf32>, %7: tensor<?x1xf32>) -> tensor<?x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %8 = linalg.matmul ins(%6, %5 : tensor<?x?xf32>, tensor<?x1xf32>) outs(%7 : tensor<?x1xf32>) -> tensor<?x1xf32>
  return %8 : tensor<?x1xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 0, 0], vector_common_parallel = [8, 1, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @gemm_unit_N(
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @gemm_unit_M_unit_N(%4: tensor<1x?xf32>, %5: tensor<?x1xf32>, %6: tensor<1x1xf32>) -> tensor<1x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %7 = linalg.matmul ins(%4, %5 : tensor<1x?xf32>, tensor<?x1xf32>) outs(%6 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %7 : tensor<1x1xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @gemm_unit_M_unit_N(
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @matmul_odd(%4: tensor<33x16xf32>, %5: tensor<16x49xf32>, %6: tensor<33x49xf32>) -> tensor<33x49xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<33x49xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<33x49xf32>) -> tensor<33x49xf32>
  %9 = linalg.matmul ins(%4, %5 : tensor<33x16xf32>, tensor<16x49xf32>) outs(%8 : tensor<33x49xf32>) -> tensor<33x49xf32>
  return %9 : tensor<33x49xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [8, 49, 0], vector_common_parallel = [3, 7, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @matmul_odd(
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @generic_unit_dims_dynamic(%0: index, %1: index, %2: index, %3: index, %6: tensor<1x?x1x1x?x?x1x?xf32>) -> tensor<1x?x1x1x?x?x1x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %7 = tensor.empty(%0, %1, %2, %3) : tensor<1x?x1x1x?x?x1x?xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x?x1x1x?x?x1x?xf32>) outs(%7 : tensor<1x?x1x1x?x?x1x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.addf %in, %in : f32
    linalg.yield %9 : f32
  } -> tensor<1x?x1x1x?x?x1x?xf32>
  return %8 : tensor<1x?x1x1x?x?x1x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 64, 0, 0, 64, 64, 0, 64], vector_common_parallel = [1, 1, 1, 1, 1, 1, 1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @generic_unit_dims_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @reduce_to_scalar_static(%2: tensor<128xf32>) -> tensor<f32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%2 : tensor<128xf32>) outs(%4 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<f32>
  return %5 : tensor<f32>
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<distribution = [0], vector_reduction = [4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @reduce_to_scalar_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @reduce_to_scalar_dynamic(%3: tensor<?xf32>, %4: tensor<f32>) -> tensor<f32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%3 : tensor<?xf32>) outs(%4 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<f32>
  return %5 : tensor<f32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0], vector_reduction = [4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @reduce_to_scalar_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<() -> ()>
func.func @scalar(%2: tensor<f32>, %3: tensor<f32>) -> tensor<f32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%2 : tensor<f32>) outs(%3 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.addf %in, %out : f32
    linalg.yield %5 : f32
  } -> tensor<f32>
  return %4 : tensor<f32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDefault>
//      CHECK: func.func @scalar(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_8x8(%2: tensor<512x1024xf32>, %3: tensor<1024x512xf32>) -> tensor<1024x512xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024x512xf32>
  return %4 : tensor<1024x512xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, 8]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_16x16(%2: tensor<512x1024xf32>, %3: tensor<1024x512xf32>) -> tensor<1024x512xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024x512xf32>
  return %4 : tensor<1024x512xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [16, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @multi_root(%3: tensor<12x128x128xf32>, %4: tensor<12x128xf32>) -> tensor<12x128xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<12x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3 : tensor<12x128x128xf32>) outs(%4 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.maximumf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<12x128xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%6 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %9 = arith.subf %in, %in_0 : f32
    %10 = math.exp %9 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  return %8 : tensor<12x128xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 4]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 4, 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_cpu.lowering_config<distribution = [4, 32, 0], vector_common_parallel = [1, 4, 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @multi_root(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG2]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG3]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @pack(%2: tensor<20x48xf32>) -> tensor<2x48x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<2x48x16x1xf32>
  %pack = linalg.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<20x48xf32> -> tensor<2x48x16x1xf32>
  return %pack : tensor<2x48x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 8], vector_common_parallel = [1, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//      CHECK: func.func @pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @pack_f16(%2: tensor<20x48xf16>) -> tensor<2x48x16x1xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f16
  %3 = tensor.empty() : tensor<2x48x16x1xf16>
  %pack = linalg.pack %2 padding_value(%cst : f16) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<20x48xf16> -> tensor<2x48x16x1xf16>
  return %pack : tensor<2x48x16x1xf16>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 8], vector_common_parallel = [1, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//      CHECK: func.func @pack_f16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @pack_many_elements(%2: tensor<1200x500000xf32>) -> tensor<31250x1200x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<31250x1200x16x1xf32>
  %pack = linalg.pack %2 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %3 : tensor<1200x500000xf32> -> tensor<31250x1200x16x1xf32>
  return %pack : tensor<31250x1200x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [50, 3], vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//      CHECK: func.func @pack_many_elements(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @unpack_generic_pack(%3: tensor<24x32x16x16xf32>, %4: tensor<512xf32>) -> tensor<24x512x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 3.40282347E+38 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<24x512x16x1xf32>
  %6 = tensor.empty() : tensor<384x512xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %6 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%6 : tensor<384x512xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.addf %in, %in_1 : f32
    %9 = arith.minimumf %8, %cst : f32
    %10 = arith.maximumf %9, %cst_0 : f32
    linalg.yield %10 : f32
  } -> tensor<384x512xf32>
  %pack = linalg.pack %7 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %5 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
  return %pack : tensor<24x512x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [16, 4]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [16, 4]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @unpack_generic_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG3]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack(%2: tensor<128x384xf32>) -> tensor<16x384x8x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<128x384xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x384xf32>) outs(%3 : tensor<128x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %in : f32
    linalg.yield %6 : f32
  } -> tensor<128x384xf32>
  %5 = tensor.empty() : tensor<16x384x8x1xf32>
  %pack = linalg.pack %4 inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %5 : tensor<128x384xf32> -> tensor<16x384x8x1xf32>
  return %pack : tensor<16x384x8x1xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, 1]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @elem_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_pack(%2: tensor<30522x768xf32>) -> tensor<1908x768x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<768x30522xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<30522x768xf32>) outs(%3 : tensor<768x30522xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<768x30522xf32>
  %5 = tensor.empty() : tensor<1908x768x16x1xf32>
  %pack = linalg.pack %4 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %5 : tensor<768x30522xf32> -> tensor<1908x768x16x1xf32>
  return %pack : tensor<1908x768x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [1, 16]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @transpose_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @reduction_broadcast_pack(%5: tensor<384x1024xf32>, %6: tensor<384xf32>, %7: tensor<1024xf32>, %8: tensor<1024xf32>) -> tensor<24x1024x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 1.024000e+03 : f32
  %cst_1 = arith.constant 9.99999996E-13 : f32
  %9 = tensor.empty() : tensor<24x1024x16x1xf32>
  %10 = tensor.empty() : tensor<384x1024xf32>
  %11 = tensor.empty() : tensor<384xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<384xf32>) -> tensor<384xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%5, %6 : tensor<384x1024xf32>, tensor<384xf32>) outs(%12 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %15 = arith.subf %in, %in_2 : f32
    %16 = arith.mulf %15, %15 : f32
    %17 = arith.addf %out, %16 : f32
    linalg.yield %17 : f32
  } -> tensor<384xf32>
  %14 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map2, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %13, %7, %8, %6 : tensor<384x1024xf32>, tensor<384xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<384xf32>) outs(%10 : tensor<384x1024xf32>) {
  ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
    %15 = arith.divf %in_2, %cst_0 : f32
    %16 = arith.addf %15, %cst_1 : f32
    %17 = math.rsqrt %16 : f32
    %18 = arith.mulf %17, %in_3 : f32
    %19 = arith.mulf %in_5, %18 : f32
    %20 = arith.subf %in_4, %19 : f32
    %21 = arith.mulf %in, %18 : f32
    %22 = arith.addf %21, %20 : f32
    linalg.yield %22 : f32
  } -> tensor<384x1024xf32>
  %pack = linalg.pack %14 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %9 : tensor<384x1024xf32> -> tensor<24x1024x16x1xf32>
  return %pack : tensor<24x1024x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [16]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [16, 0], vector_reduction = [0, 16]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [16, 0], vector_inner_parallel = [0, 1]>
//  CHECK-DAG: #[[CONFIG4:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @reduction_broadcast_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG4]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @reduction_pack(%3: tensor<384x1024x32xf32>, %4: tensor<384x1024xf32>) -> tensor<1024x24x16x1xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant -0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1024x24x16x1xf32>
  %6 = tensor.empty() : tensor<384x1024x32xf32>
  %7 = tensor.empty() : tensor<384x1024xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<384x1024xf32>) -> tensor<384x1024xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<384x1024x32xf32>, tensor<384x1024xf32>) outs(%8 : tensor<384x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %10 = arith.subf %in, %in_0 : f32
    %11 = arith.mulf %10, %10 : f32
    %12 = arith.addf %out, %11 : f32
    linalg.yield %12 : f32
  } -> tensor<384x1024xf32>
  %pack = linalg.pack %9 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %5 : tensor<384x1024xf32> -> tensor<1024x24x16x1xf32>
  return %pack : tensor<1024x24x16x1xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [16, 1]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 32, 0], vector_common_parallel = [16, 1, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @reduction_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG3]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @unpack_static(%2: tensor<64x256x16x16xf32>) -> tensor<1024x4096xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<1024x4096xf32>
  %unpack = linalg.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<64x256x16x16xf32> -> tensor<1024x4096xf32>
  return %unpack : tensor<1024x4096xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [16, 16]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//      CHECK: func.func @unpack_static(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @unpack_elem(%3: tensor<48x64x8x2xf32>, %4: tensor<128xf32>) -> tensor<128x384xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %5 = tensor.empty() : tensor<128x384xf32>
  %6 = tensor.empty() : tensor<384x128xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %6 : tensor<48x64x8x2xf32> -> tensor<384x128xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<128xf32>, tensor<384x128xf32>) outs(%5 : tensor<128x384xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.addf %in, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<128x384xf32>
  return %7 : tensor<128x384xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [2, 8]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @unpack_elem(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @quant_model(%4: tensor<2304x24xi8>, %5: tensor<24x144xi8>, %6: tensor<144xi32>) -> tensor<2304x144xi8> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c12_i32 = arith.constant 12 : i32
  %c-128_i32 = arith.constant -128 : i32
  %c0_i32 = arith.constant 0 : i32
  %7 = tensor.empty() : tensor<2304x144xi8>
  %8 = tensor.empty() : tensor<2304x144xi32>
  %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
  %10 = linalg.matmul ins(%4, %5 : tensor<2304x24xi8>, tensor<24x144xi8>) outs(%9 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
  %11 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %10 : tensor<144xi32>, tensor<2304x144xi32>) outs(%7 : tensor<2304x144xi8>) {
  ^bb0(%in: i32, %in_0: i32, %out: i8):
    %12 = arith.subi %in_0, %c12_i32 : i32
    %13 = arith.addi %in, %12 : i32
    %14 = arith.trunci %13 : i32 to i8
    linalg.yield %14 : i8
  } -> tensor<2304x144xi8>
  return %11 : tensor<2304x144xi8>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 16, 0], distribution = [64, 16, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @quant_model(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = false}>
func.func @test(%2: tensor<i64>) -> tensor<i64> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
  %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
  %extracted = tensor.extract %2[] : tensor<i64>
  %3 = arith.muli %extracted, %c6364136223846793005_i64 : i64
  %4 = arith.addi %3, %c1442695040888963407_i64 : i64
  %inserted = tensor.insert %4 into %2[] : tensor<i64>
  return %inserted : tensor<i64>
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDefault>
//      CHECK:   func.func @test(
// CHECK-SAME:       translation_info = #[[TRANSLATION]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu = "cascadelake", cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", link_embedded = false, native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu", ukernels = false}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @non_trivial_program(%3: tensor<128x1x128x1xf32>, %4: tensor<128x1xf32>) -> tensor<1x1xf32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x1xf32>
  %6 = tensor.empty() : tensor<128xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<128xf32>) -> tensor<128xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1], [2, 3]] : tensor<128x1x128x1xf32> into tensor<128x128xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<128x128xf32>) outs(%7 : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %out, %in : f32
    linalg.yield %11 : f32
  } -> tensor<128xf32>
  %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
  %10 = linalg.matmul ins(%expanded, %4 : tensor<1x128xf32>, tensor<128x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %10 : tensor<1x1xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 16]>
//  CHECK-NOT:   lowering_config
//      CHECK: func.func @non_trivial_program(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = true}>
func.func @batch_mmt4d(%17: tensor<128x10x32x8x1xf32>, %18: tensor<128x80x32x4x1xf32>) -> tensor<128x10x80x8x4xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %19 = tensor.empty() : tensor<128x10x80x8x4xf32>
  %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
  %21 = linalg.batch_mmt4d ins(%17, %18 : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>) outs(%20 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
  return %21 : tensor<128x10x80x8x4xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 10, 80, 0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0, 8, 4, 0], vector_reduction = [0, 0, 0, 1, 0, 0, 1]>
//      CHECK: func.func @batch_mmt4d(
//      CHECK:   linalg.batch_mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
func.func @mmt4d_with_large_reduction(%3: tensor<7x18176x16x1xf32>, %4: tensor<284x18176x16x1xf32>) -> tensor<7x284x16x16xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<7x284x16x16xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<7x284x16x16xf32>) -> tensor<7x284x16x16xf32>
  %7 = linalg.mmt4d ins(%3, %4 : tensor<7x18176x16x1xf32>, tensor<284x18176x16x1xf32>) outs(%6 : tensor<7x284x16x16xf32>) -> tensor<7x284x16x16xf32>
  return %7 : tensor<7x284x16x16xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 2, 16]>
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 2, 16, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
//      CHECK: func.func @mmt4d_with_large_reduction(
//      CHECK:   linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG]]
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG1]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @pad_only(%2: tensor<1x112x112x64xf32>) -> tensor<1x114x114x64xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %2 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x112x112x64xf32> to tensor<1x114x114x64xf32>
  return %padded : tensor<1x114x114x64xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 6, 57, 64], vector_common_parallel = [1, 1, 1, 4]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @pad_only(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pad {{.+}} {
//      CHECK:     tensor.yield
// CHECK-NEXT:   } {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @winograd_output_transform(%2: tensor<8x8x2x6x6x128xf16>) -> tensor<2x36x36x128xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<2x36x36x128xf16>
  %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
  return %4 : tensor<2x36x36x128xf16>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 2, 6, 64], vector_common_parallel = [1, 1, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_output_transform(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.output_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @winograd_input_transform(%2: tensor<2x34x34x128xf16>) -> tensor<8x8x2x6x6x128xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
  %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
  return %4 : tensor<8x8x2x6x6x128xf16>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 2, 6, 64], vector_common_parallel = [1, 1, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_input_transform(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.input_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @winograd_filter_transform(%2: tensor<3x3x64x128xf32>) -> tensor<8x8x64x128xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %3 = tensor.empty() : tensor<8x8x64x128xf32>
  %4 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%2 : tensor<3x3x64x128xf32>) outs(%3 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %4 : tensor<8x8x64x128xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [8, 64], vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_filter_transform(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @attention(%4: tensor<20x4096x64xf16>, %5: tensor<20x4096x64xf16>, %6: tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %scale = arith.constant 0.125 : f16
  %7 = tensor.empty() : tensor<20x4096x64xf16>
  %8 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
    affine_map<(d0, d1, d2, d3, d4) -> ()>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
    ins(%4, %5, %6, %scale : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16)
    outs(%7 : tensor<20x4096x64xf16>) {
     ^bb0(%score: f32):
       iree_linalg_ext.yield %score : f32
    } -> tensor<20x4096x64xf16>
  return %8 : tensor<20x4096x64xf16>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 64, 0, 0, 64], vector_common_parallel = [1, 1, 0, 0, 32], vector_reduction = [0, 0, 0, 2, 0]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @attention(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//     CHECK:   iree_linalg_ext.attention
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @attention_transpose_distribute_4d(%29: index, %37: tensor<4x4x?x128xf16>, %38: tensor<4x4x?x1x1x128xf16>, %39: tensor<4x4x?x1x1x128xf16>, %40: tensor<4x4x?x?x1x1xf16>) -> tensor<4x?x4x128xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 8.837890e-02 : f16
  %30 = util.assume.int %29<umin = 16, umax = 131056, udiv = 16> : index
  %31 = iree_tensor_ext.dispatch.workload.ordinal %30, 0 : index
  %41 = tensor.empty(%31) : tensor<4x?x4x128xf16>
  %42 = tensor.empty(%31) : tensor<4x4x?x128xf16>
  %43 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>]} ins(%37, %38, %39, %cst, %40 : tensor<4x4x?x128xf16>, tensor<4x4x?x1x1x128xf16>, tensor<4x4x?x1x1x128xf16>, f16, tensor<4x4x?x?x1x1xf16>) outs(%42 : tensor<4x4x?x128xf16>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<4x4x?x128xf16>
  %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<4x4x?x128xf16>) outs(%41 : tensor<4x?x4x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x4x128xf16>
  return %44 : tensor<4x?x4x128xf16>
}
// CHECK-DAG:  #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 64, 64, 0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 2, 0, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 0, 1, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @attention_transpose_distribute_4d
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//     CHECK:   iree_linalg_ext.attention
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @elementwise_output_transposed(%4: tensor<i64>, %5: tensor<768xi64>, %6: tensor<32xi64>) -> tensor<32x32x768xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %7 = tensor.empty() : tensor<32x32x768xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5, %6 : tensor<i64>, tensor<768xi64>, tensor<32xi64>) outs(%7 : tensor<32x32x768xf32>) {
  ^bb0(%in: i64, %in_0: i64, %in_1: i64, %out: f32):
    %9 = arith.addi %in, %in_0 : i64
    %10 = arith.addi %9, %in_1 : i64
    %11 = arith.uitofp %10 : i64 to f32
    linalg.yield %11 : f32
  } -> tensor<32x32x768xf32>
  return %8 : tensor<32x32x768xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 32, 32], vector_common_parallel = [1, 8, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @elementwise_output_transposed(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//     CHECK:    linalg.generic
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

// Test scenario: While doing vectorization with masking strategy, and when the vector size is not a multiple of the element size,
// the vectorization could result in holes in the vectorized load/store, which might result in undefined behavior such as divide by zero.
// To avoid this, do not use masking strategy if the vectorized operation may result in undefined behavior. In this case, `arith.remsi`
// could result in divide by zero exception with masking strategy when the loop size is not a multiple of the vector size.

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 16}>
module {
  func.func @test_mod_vectorizing_strategy_peeling(%3: tensor<6xi32>, %4: tensor<6xi32>) -> tensor<6xi32> attributes {hal.executable.target = #executable_target_system_elf_x86_64_}{
    %c0 = arith.constant 0 : index
    %5 = tensor.empty() : tensor<6xi32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<6xi32>, tensor<6xi32>) outs(%5 : tensor<6xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %7 = arith.remsi %in, %in_0 : i32
      linalg.yield %7 : i32
    } -> tensor<6xi32>
  return %6 : tensor<6xi32>
  }
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling}>
// CHECK-LABEL: @test_mod_vectorizing_strategy_peeling
// CHECK-SAME: attributes {hal.executable.target = #executable_target_system_elf_x86_64, translation_info = #translation}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
    "llvm-cpu", "embedded-elf-x86_64",
    {cpu_features = "+avx512f",
     data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
     native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @custom_op(%arg0 : tensor<384x512xf32>, %arg1 : tensor<512x128xf32>,
    %arg2 : tensor<128xf32>) -> tensor<384x128xf32>
    attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<384x128xf32>
  %1 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d1)>,
                       affine_map<(d0, d1)[s0] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1, %arg2 : tensor<384x512xf32>, tensor<512x128xf32>, tensor<128xf32>)
      outs(%0 : tensor<384x128xf32>) {
    ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?xf32>, %t3 : tensor<?x?xf32>):
      %2 = linalg.fill ins(%cst : f32) outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %3 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %t2 : tensor<?x?xf32>, tensor<?xf32>)
          outs(%t3 : tensor<?x?xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %5 = arith.addf %b0, %b1 : f32
          linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<384x128xf32>
  return %1 : tensor<384x128xf32>
}
//  CHECK-DAG: #[[CONFIG0:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[48, 64, 0]]>
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 16]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [48, 64, 0], vector_common_parallel = [8, 16, 0], vector_reduction = [0, 0, 16]>
//  CHECK-DAG: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func @custom_op(
// CHECK-SAME:     translation_info = #translation
//      CHECK:   iree_linalg_ext.custom_op
// CHECK-SAME:       attributes {lowering_config = #[[CONFIG0]]}
//      CHECK:   ^bb
//      CHECK:     linalg.fill
// CHECK-SAME:         {lowering_config = #[[CONFIG1]]}
//      CHECK:     linalg.matmul
// CHECK-SAME:         {lowering_config = #[[CONFIG2]]}
//      CHECK:     linalg.generic
// CHECK-SAME:         {lowering_config = #[[CONFIG1]]}
//      CHECK:   iree_linalg_ext.yield

// -----

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64",
    {cpu_features = "+avx512f",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @custom_op_preset_config(%arg0: tensor<384x512xf32>, %arg1: tensor<512x128xf32>,
      %arg2: tensor<128xf32>) -> tensor<384x128xf32>
      attributes {hal.executable.target = #executable_target, translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<384x128xf32>
    %1 = iree_linalg_ext.custom_op{
        indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                         affine_map<(d0, d1)[s0] -> (s0, d1)>,
                         affine_map<(d0, d1)[s0] -> (d1)>,
                         affine_map<(d0, d1)[s0] -> (d0, d1)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<parallel>]}
        attributes {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[24, 32]]>}
        ins(%arg0, %arg1, %arg2 : tensor<384x512xf32>, tensor<512x128xf32>, tensor<128xf32>) outs(%0 : tensor<384x128xf32>) {
    ^bb0(%arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: tensor<?xf32>, %arg6: tensor<?x?xf32>):
      %2 = linalg.fill ins(%cst : f32) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %3 = linalg.matmul ins(%arg3, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %arg5 : tensor<?x?xf32>, tensor<?xf32>) outs(%arg6 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
    } -> tensor<384x128xf32>
    return %1 : tensor<384x128xf32>
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[24, 32]]>
//  CHECK-DAG: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<pipeline = CPUDefault>
//      CHECK: func @custom_op_preset_config(
// CHECK-SAME:     translation_info = #[[TRANSLATION_INFO]]
//      CHECK:   iree_linalg_ext.custom_op
// CHECK-SAME:       lowering_config = #[[CONFIG]]
//  CHECK-NOT:   lowering_config

// -----

// Test additional level of tiling in the CPUDefault pipeline. linalg.quantized_matmul doesn't have specialized pipeline
// since, it gets decomposed to matmul that has specialized pipeline.
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @test_tiling_cpu_default(%arg0: tensor<256x256xi8>, %arg1: tensor<256x256xi8>, %arg2: i32, %arg3: i32, %arg4: tensor<256x256xi32>) -> tensor<256x256xi32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %0 = linalg.quantized_matmul ins(%arg0, %arg1, %arg2, %arg3 : tensor<256x256xi8>, tensor<256x256xi8>, i32, i32) outs(%arg4 : tensor<256x256xi32>) -> tensor<256x256xi32>
    return %0 : tensor<256x256xi32>
}
// CHECK-DAG:  #[[CONFIG0:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [4, 64, 0]>
// CHECK-DAG:  #[[TRANSLATION_INFO]] = #iree_codegen.translation_info<pipeline = CPUDefault>
//      CHECK: func @test_tiling_cpu_default(
// CHECK-SAME:     translation_info = #[[TRANSLATION_INFO]]
//      CHECK:    linalg.quantized_matmul {lowering_config = #[[CONFIG0]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @i1_type(%3: tensor<8xi1>, %4: tensor<8xi1>)  -> tensor<8xi1> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<8xi1>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<8xi1>, tensor<8xi1>) outs(%5 : tensor<8xi1>) {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %7 = arith.xori %in, %in_0 : i1
    linalg.yield %7 : i1
  } -> tensor<8xi1>
  return %6 : tensor<8xi1>
}
// CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [8], vector_common_parallel = [8]>
// CHECK: func @i1_type(
// CHECK: linalg.generic {
// CHECK-SAME: {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @complex_view_as_real(%4: tensor<1xi32>, %5: tensor<1x1x32x50x2xf32>, %8: index, %9: tensor<50xcomplex<f32>>) -> tensor<32x50x2xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %6 = tensor.empty() : tensor<32x50x2xf32>
  %extracted = tensor.extract %4[%c0] : tensor<1xi32>
  %7 = arith.extsi %extracted : i32 to i64
  %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<50xcomplex<f32>>) outs(%6 : tensor<32x50x2xf32>) {
  ^bb0(%in: complex<f32>, %out: f32):
    %11 = linalg.index 0 : index
    %12 = linalg.index 1 : index
    %extracted_0 = tensor.extract %5[%c0, %c0, %11, %12, %c0] : tensor<1x1x32x50x2xf32>
    %extracted_1 = tensor.extract %5[%c0, %c0, %11, %12, %c1] : tensor<1x1x32x50x2xf32>
    %13 = complex.create %extracted_0, %extracted_1 : complex<f32>
    %14 = complex.mul %13, %in : complex<f32>
    %15 = complex.re %14 : complex<f32>
    %16 = complex.im %14 : complex<f32>
    %17 = linalg.index 2 : index
    %18 = arith.cmpi eq, %17, %c0 : index
    %19 = arith.select %18, %15, %16 : f32
    linalg.yield %19 : f32
  } -> tensor<32x50x2xf32>
  return %10 : tensor<32x50x2xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 25, 2], vector_common_parallel = [1, 1, 2]>
//      CHECK: func.func @complex_view_as_real(
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
func.func @decode_reduction_f32(%arg0: tensor<32x262144xf16>, %arg1: tensor<32xf32>, %arg2: tensor<32x16x16384xf16>, %arg3: tensor<32x16xf16>, %arg4: tensor<32x16xf16>) -> tensor<16384x32x16xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.621440e+05 : f32
  %cst_1 = arith.constant 9.99999997E-7 : f32
  %0 = tensor.empty() : tensor<16384x32x16xf16>
  %1 = tensor.empty() : tensor<32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32xf32>) -> tensor<32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x262144xf16>, tensor<32xf32>) outs(%2 : tensor<32xf32>) {
  ^bb0(%in: f16, %in_2: f32, %out: f32):
    %5 = arith.extf %in : f16 to f32
    %6 = arith.subf %5, %in_2 : f32
    %7 = arith.mulf %6, %6 : f32
    %8 = arith.addf %7, %out : f32
    linalg.yield %8 : f32
  } -> tensor<32xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map4, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2, %arg1, %3, %arg3, %arg4 : tensor<32x16x16384xf16>, tensor<32xf32>, tensor<32xf32>, tensor<32x16xf16>, tensor<32x16xf16>) outs(%0 : tensor<16384x32x16xf16>) {
  ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f16, %in_5: f16, %out: f16):
    %5 = arith.divf %in_3, %cst_0 : f32
    %6 = arith.addf %5, %cst_1 : f32
    %7 = math.rsqrt %6 : f32
    %8 = arith.extf %in : f16 to f32
    %9 = arith.subf %8, %in_2 : f32
    %10 = arith.mulf %9, %7 : f32
    %11 = arith.extf %in_4 : f16 to f32
    %12 = arith.mulf %10, %11 : f32
    %13 = arith.extf %in_5 : f16 to f32
    %14 = arith.addf %12, %13 : f32
    %15 = arith.truncf %14 : f32 to f16
    linalg.yield %15 : f16
  } -> tensor<16384x32x16xf16>
  return %4 : tensor<16384x32x16xf16>
}
//  CHECK-DAG: #[[CONFIG0:.+]] = #iree_cpu.lowering_config<distribution = [4, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 8]>
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [4, 0, 0], vector_inner_parallel = [0, 1, 4]>
//      CHECK: func.func @decode_reduction_f32
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG0]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]

// -----

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
func.func @attention_reshape_pack(%arg0: index, %arg1: tensor<4x2x?x32xf16>, %arg2: tensor<?x4x32xf16>, %arg3: tensor<?x4x32xf16>, %arg4: tensor<4x2x?x?xf16>) -> tensor<?x256x1x1xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 1.767580e-01 : f16
  %0 = tensor.empty(%arg0) : tensor<?x4x2x32xf16>
  %1 = tensor.empty(%arg0) : tensor<4x2x?x32xf16>
  %2 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]} ins(%arg1, %arg2, %arg3, %cst_0, %arg4 : tensor<4x2x?x32xf16>, tensor<?x4x32xf16>, tensor<?x4x32xf16>, f16, tensor<4x2x?x?xf16>) outs(%1 : tensor<4x2x?x32xf16>) {
  ^bb0(%arg5: f32):
    iree_linalg_ext.yield %arg5 : f32
  } -> tensor<4x2x?x32xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<4x2x?x32xf16>) outs(%0 : tensor<?x4x2x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<?x4x2x32xf16>
  %collapsed = tensor.collapse_shape %3 [[0], [1, 2, 3]] : tensor<?x4x2x32xf16> into tensor<?x256xf16>
  %4 = tensor.empty(%arg0) : tensor<?x256x1x1xf16>
  %pack = linalg.pack %collapsed padding_value(%cst : f16) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %4 : tensor<?x256xf16> -> tensor<?x256x1x1xf16>
  return %pack : tensor<?x256x1x1xf16>
}
//  CHECK-DAG: #[[CONFIG0:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 64, 16, 0, 0], vector_common_parallel = [1, 1, 4, 16, 0, 0], vector_reduction = [0, 0, 0, 0, 0, 32]>
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 4, 16]>
//  CHECK-NOT: #iree_cpu.lowering_config
//      CHECK: func.func @attention_reshape_pack
//      CHECK:   iree_linalg_ext.attention
// CHECK-SAME:       lowering_config = #[[CONFIG0]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
func.func @attention_dynamic_3d(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %scale = arith.constant 0.125 : f32
  %out = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %result = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
    affine_map<(d0, d1, d2, d3, d4) -> ()>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
    ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32)
    outs(%out : tensor<?x?x?xf32>) {
     ^bb0(%score: f32):
       iree_linalg_ext.yield %score : f32
    } -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 64, 0, 0, 64], vector_common_parallel = [1, 16, 0, 0, 16], vector_reduction = [0, 0, 16, 16, 0]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//      CHECK: func.func @attention_dynamic_3d(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.attention
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @mmt4d_generic_unpack_pack(%arg0: tensor<5x4096x16x1xf16>, %arg1: tensor<640x4096x16x1xf16>) -> tensor<5x10240x16x1xf16> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x640x16x16xf16>
  %1 = tensor.empty() : tensor<5x640x16x16xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<5x640x16x16xf32>) -> tensor<5x640x16x16xf32>
  %3 = linalg.mmt4d ins(%arg0, %arg1 : tensor<5x4096x16x1xf16>, tensor<640x4096x16x1xf16>) outs(%2 : tensor<5x640x16x16xf32>) -> tensor<5x640x16x16xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<5x640x16x16xf32>) outs(%0 : tensor<5x640x16x16xf16>) {
  ^bb0(%in: f32, %out: f16):
    %7 = arith.truncf %in : f32 to f16
    linalg.yield %7 : f16
  } -> tensor<5x640x16x16xf16>
  %5 = tensor.empty() : tensor<77x10240xf16>
  %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %5 : tensor<5x640x16x16xf16> -> tensor<77x10240xf16>
  %6 = tensor.empty() : tensor<5x10240x16x1xf16>
  %pack = linalg.pack %unpack padding_value(%cst : f16) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %6 : tensor<77x10240xf16> -> tensor<5x10240x16x1xf16>
  return %pack : tensor<5x10240x16x1xf16>
}
// CHECK-DAG:   #[[$CONFIG0:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 16, 16]>
// CHECK-DAG:   #[[$CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 16, 16, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
// CHECK-DAG:   #[[$CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-LABEL: func.func @mmt4d_generic_unpack_pack(
// CHECK:         linalg.fill
// CHECK-SAME:      {lowering_config = #[[$CONFIG0]]}
// CHECK:         linalg.mmt4d
// CHECK-SAME:      {lowering_config = #[[$CONFIG1]]}
// CHECK:         linalg.generic
// CHECK-SAME:      {lowering_config = #[[$CONFIG0]]}
// CHECK:         linalg.unpack
// CHECK-SAME:      {lowering_config = #[[$CONFIG2]]}
// CHECK:         linalg.pack
// CHECK-NOT:      lowering_config

// -----

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>
func.func @batch_mmt4d_generic_form(%lhs: tensor<128x10x32x8x1xf32>, %rhs: tensor<128x80x32x4x1xf32>, %acc: tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32> attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>)
    outs(%acc : tensor<128x10x80x8x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<128x10x80x8x4xf32>
  return %0 : tensor<128x10x80x8x4xf32>
}
// CHECK:       #[[$CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [32, 10, 20, 0, 8, 4, 0], vector_common_parallel = [1, 1, 1, 0, 1, 4, 0], vector_reduction = [0, 0, 0, 1, 0, 0, 0]>
// CHECK-LABEL: func.func @batch_mmt4d_generic_form(
// CHECK:         linalg.generic
// CHECK-SAME:      {lowering_config = #[[$CONFIG]]}

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
func.func @gather(%source: tensor<16x8x32x128xf16>, %indices: tensor<4xi64>, %out: tensor<4x8x32x128xf16>) -> tensor<4x8x32x128xf16> attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %0 = iree_linalg_ext.gather dimension_map = [0] ins(%source, %indices : tensor<16x8x32x128xf16>, tensor<4xi64>) outs(%out : tensor<4x8x32x128xf16>) -> tensor<4x8x32x128xf16>
  return %0 : tensor<4x8x32x128xf16>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 4, 32, 64], vector_common_parallel = [1, 1, 1, 8]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>
//       CHECK: func.func @gather(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.gather
//  CHECK-SAME:       lowering_config = #[[CONFIG]]
