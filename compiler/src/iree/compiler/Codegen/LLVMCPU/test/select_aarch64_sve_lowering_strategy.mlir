// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file --iree-llvmcpu-disable-arm-sme-tiling %s | FileCheck %s --check-prefixes=CHECK,DISABLE-ARM-SME

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [4, [16], 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @matmul_tensors(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @static_tensors_non_pow_two_sizes(%3: tensor<15x14xf32>, %4: tensor<14x7xf32>, %5: tensor<15x7xf32>) -> tensor<15x7xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %6 = linalg.matmul ins(%3, %4 : tensor<15x14xf32>, tensor<14x7xf32>) outs(%5 : tensor<15x7xf32>) -> tensor<15x7xf32>
  return %6 : tensor<15x7xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [5, 7, 0], vector_common_parallel = [5, [8], 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
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
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @static_tensors_1x1(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [4, [16], 0], vector_reduction = [0, 0, 1]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      DISABLE-ARM-SME: func.func @matmul_tensors(
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = {{\[}}[8], [8], 0], vector_reduction = [0, 0, 1]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       WITH-SME: func.func @matmul_tensors(
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @matmul_with_fill(%15: tensor<1024x256xi8>, %16: tensor<256x256xi8>, %17: tensor<1024xf32>, %18: tensor<256xf32>) -> tensor<1024x256xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c0_i32 = arith.constant 0 : i32
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
   return %23 : tensor<1024x256xf32>
}
// CHECK-DAG:  #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [4, [16]]>
// CHECK-DAG:  #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [4, [16], 0], vector_reduction = [0, 0, 1]>
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK:      func.func @matmul_with_fill(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

// This case tests if the inner tile size of the mmt4d is inferred properly from the shape-aware HAL binding and set accordingly.

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<()[s0] -> (256 ceildiv s0)>
func.func @mmt4d_tensors(%arg0: tensor<32x128x8x1xf32>, %arg1 : tensor<?x128x?x1xf32>) -> tensor<32x?x8x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %n0 = affine.apply #map()[%c8_vscale]
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x?x8x?xf32>>{%n0, %c8_vscale}
  %init = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, %n0, 8, %c8_vscale], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x?x8x?xf32>>{%n0, %c8_vscale} -> tensor<32x?x8x?xf32>
  %mmt4d = linalg.mmt4d ins(%arg0, %arg1 : tensor<32x128x8x1xf32>, tensor<?x128x?x1xf32>) outs(%init : tensor<32x?x8x?xf32>) -> tensor<32x?x8x?xf32>
  return %mmt4d : tensor<32x?x8x?xf32>
}
// CHECK-DAG:  #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 8, [8], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = Mmt4dTilingExpert>
// CHECK:      func.func @mmt4d_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<()[s0] -> (256 ceildiv s0)>
func.func @mmtd4_with_fill(%arg0 : tensor<32x128x8x1xf32>, %arg1 : tensor<?x128x?x1xf32>) -> tensor<32x?x8x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %0 = affine.apply #map()[%c8_vscale]
  %init = tensor.empty(%0, %c8_vscale) : tensor<32x?x8x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x?x8x?xf32>) -> tensor<32x?x8x?xf32>
  %mmt4d = linalg.mmt4d ins(%arg0, %arg1 : tensor<32x128x8x1xf32>, tensor<?x128x?x1xf32>) outs(%fill : tensor<32x?x8x?xf32>) -> tensor<32x?x8x?xf32>
  return %mmt4d : tensor<32x?x8x?xf32>
}
// CHECK-DAG:  #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 8, [8]]>
// CHECK-DAG:  #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [4, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 8, [8], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = Mmt4dTilingExpert>
// CHECK:      func.func @mmtd4_with_fill
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<()[s0] -> (320 ceildiv s0)>
func.func @unpack(%arg0 : tensor<128x10x?x8x?xf32>) -> tensor<128x80x320xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %init = tensor.empty() : tensor<128x80x320xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, %c8_vscale] into %init : tensor<128x10x?x8x?xf32> -> tensor<128x80x320xf32>
  return %unpack : tensor<128x80x320xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 40, 64], vector_common_parallel = [1, 8, [8]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling
//
// For SVE, we do not decompose unpacks.
//   CHECK-NOT: enable_loop_peeling
//       CHECK: func.func @unpack
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
func.func @unpack_outer_dynamic(%arg0 : tensor<?x?x32x?xi32>, %dim0 : index, %dim1 : index) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, %c8_vscale] into %init : tensor<?x?x32x?xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [32, [8]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling
//
// For SVE, we do not decompose unpacks.
//   CHECK-NOT: enable_loop_peeling
//       CHECK: func.func @unpack_outer_dynamic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
func.func @unpack_fully_dynamic(%arg0 : tensor<?x?x?x?xi32>, %m0 : index, %n0 : index, %m : index, %n : index) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %init = tensor.empty(%m, %n) : tensor<?x?xi32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [%m0, %n0] into %init : tensor<?x?x?x?xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}
// If the inner tile sizes are fully dynamic and/or the scalable tile sizes cannot be inferred from the IR,
// we fallback to the default tile sizes.
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [1, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling
//       CHECK: func.func @unpack_fully_dynamic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
func.func @unpack_with_generic(%arg0 : tensor<128x10x?x8x?xf32>, %arg1 : tensor<128x320xf32>) -> tensor<128x320x80xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %init = tensor.empty() : tensor<128x80x320xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, %c8_vscale] into %init : tensor<128x10x?x8x?xf32> -> tensor<128x80x320xf32>
  %init2 = tensor.empty() : tensor<128x320x80xf32>
  %generic = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1, %unpack : tensor<128x320xf32>, tensor<128x80x320xf32>) outs(%init2 : tensor<128x320x80xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %10 = arith.addf %in, %in_0 : f32
    linalg.yield %10 : f32
  } -> tensor<128x320x80xf32>
  return %generic : tensor<128x320x80xf32>
}
//   CHECK-DAG: #[[CONFIG_UNPACK:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8, [8]]>
//   CHECK-DAG: #[[CONFIG_GENERIC:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 40], vector_common_parallel = [1, [8], 8]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert
//       CHECK: func.func @unpack_with_generic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG_UNPACK]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[CONFIG_GENERIC]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map_pack = affine_map<()[s0] -> (48 ceildiv s0)>
func.func @pack(%arg0: tensor<20x48xf32>) -> tensor<2x?x16x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer1 = affine.apply #map_pack()[%c16_vscale]
  %empty = tensor.empty(%outer1, %c16_vscale) : tensor<2x?x16x?xf32>
  %pack = linalg.pack %arg0 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, %c16_vscale] into %empty : tensor<20x48xf32> -> tensor<2x?x16x?xf32>
  return %pack : tensor<2x?x16x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 4], vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDataTiling>
//      CHECK: func.func @pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map_elem_pack = affine_map<()[s0] -> (384 ceildiv s0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack(%arg0: tensor<128x384xf32>) -> tensor<16x?x8x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %empty = tensor.empty() : tensor<128x384xf32>
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer1 = affine.apply #map_elem_pack()[%c16_vscale]
  %filled = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x384xf32>) outs(%empty : tensor<128x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %in : f32
    linalg.yield %sum : f32
  } -> tensor<128x384xf32>
  %dest = tensor.empty(%outer1, %c16_vscale) : tensor<16x?x8x?xf32>
  %pack = linalg.pack %filled inner_dims_pos = [0, 1] inner_tiles = [8, %c16_vscale] into %dest : tensor<128x384xf32> -> tensor<16x?x8x?xf32>
  return %pack : tensor<16x?x8x?xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, [16]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @elem_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map_rb0 = affine_map<(d0, d1) -> (d0, d1)>
#map_rb1 = affine_map<(d0, d1) -> (d0)>
#map_rb2 = affine_map<(d0, d1) -> (d1)>
#map_rb_outer = affine_map<()[s0] -> (1024 ceildiv s0)>
func.func @reduction_broadcast_pack(%arg0: tensor<384x1024xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>) -> tensor<48x?x8x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 1.024000e+03 : f32
  %cst_1 = arith.constant 9.99999996E-13 : f32
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer1 = affine.apply #map_rb_outer()[%c16_vscale]
  %empty0 = tensor.empty() : tensor<384xf32>
  %empty1 = tensor.empty() : tensor<384x1024xf32>
  %empty2 = tensor.empty(%outer1, %c16_vscale) : tensor<48x?x8x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty0 : tensor<384xf32>) -> tensor<384xf32>
  %generic0 = linalg.generic {indexing_maps = [#map_rb0, #map_rb1, #map_rb1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<384x1024xf32>, tensor<384xf32>) outs(%fill : tensor<384xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %diff = arith.subf %in, %in2 : f32
    %mul = arith.mulf %diff, %diff : f32
    %res = arith.addf %out, %mul : f32
    linalg.yield %res : f32
  } -> tensor<384xf32>
  %generic1 = linalg.generic {indexing_maps = [#map_rb0, #map_rb1, #map_rb2, #map_rb2, #map_rb1, #map_rb0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %generic0, %arg2, %arg3, %arg1 : tensor<384x1024xf32>, tensor<384xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<384xf32>) outs(%empty1 : tensor<384x1024xf32>) {
  ^bb0(%in: f32, %in1: f32, %in2: f32, %in3: f32, %in4: f32, %out: f32):
    %nrm = arith.divf %in1, %cst_0 : f32
    %add = arith.addf %nrm, %cst_1 : f32
    %rsqrt = math.rsqrt %add : f32
    %mul0 = arith.mulf %rsqrt, %in2 : f32
    %mul1 = arith.mulf %in4, %mul0 : f32
    %sub = arith.subf %in3, %mul1 : f32
    %mul2 = arith.mulf %in, %mul0 : f32
    %add2 = arith.addf %mul2, %sub : f32
    linalg.yield %add2 : f32
  } -> tensor<384x1024xf32>
  %pack = linalg.pack %generic1 inner_dims_pos = [0, 1] inner_tiles = [8, %c16_vscale] into %empty2 : tensor<384x1024xf32> -> tensor<48x?x8x?xf32>
  return %pack : tensor<48x?x8x?xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [8, 0], vector_reduction = [0, 4]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 0], vector_inner_parallel = [0, [16]]>
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

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "generic", cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf", ukernels = false}>
#map_tp0 = affine_map<(d0, d1) -> (d1, d0)>
#map_tp1 = affine_map<(d0, d1) -> (d0, d1)>
#map_tp_outer = affine_map<()[s0] -> (30522 ceildiv s0)>
func.func @transpose_pack(%arg0: tensor<30522x768xf32>) -> tensor<?x96x?x8xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer_dynamic = affine.apply #map_tp_outer()[%c16_vscale]
  %empty0 = tensor.empty() : tensor<768x30522xf32>
  %0 = linalg.generic {indexing_maps = [#map_tp0, #map_tp1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<30522x768xf32>) outs(%empty0 : tensor<768x30522xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<768x30522xf32>
  %empty1 = tensor.empty(%outer_dynamic, %c16_vscale) : tensor<?x96x?x8xf32>
  %pack = linalg.pack %0 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [%c16_vscale, 8] into %empty1 : tensor<768x30522xf32> -> tensor<?x96x?x8xf32>
  return %pack : tensor<?x96x?x8xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, [16]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//      CHECK: func.func @transpose_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "generic", cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf", ukernels = false}>
#map_rp0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_rp1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_rp_outer = affine_map<()[s0] -> (1024 ceildiv s0)>
func.func @reduction_pack(%arg0: tensor<384x1024x32xf32>, %arg1: tensor<384x1024xf32>) -> tensor<?x24x16x?xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant -0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer0 = affine.apply #map_rp_outer()[%c16_vscale]
  %empty0 = tensor.empty() : tensor<384x1024xf32>
  %empty2 = tensor.empty(%outer0, %c16_vscale) : tensor<?x24x16x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty0 : tensor<384x1024xf32>) -> tensor<384x1024xf32>
  %generic = linalg.generic {indexing_maps = [#map_rp0, #map_rp1, #map_rp1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<384x1024x32xf32>, tensor<384x1024xf32>) outs(%fill : tensor<384x1024xf32>) {
  ^bb0(%in: f32, %in0: f32, %out: f32):
    %sub = arith.subf %in, %in0 : f32
    %mul = arith.mulf %sub, %sub : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<384x1024xf32>
  %pack = linalg.pack %generic outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, %c16_vscale] into %empty2 : tensor<384x1024xf32> -> tensor<?x24x16x?xf32>
  return %pack : tensor<?x24x16x?xf32>
}
//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [16, [16]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 32, 0], vector_common_parallel = [16, [16], 0], vector_reduction = [0, 0, 4]>
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

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<()[s0] -> (10240 ceildiv s0)>
func.func @mmt4d_generic_unpack_pack(%arg0: tensor<5x4096x16x1xf16>, %arg1: tensor<?x4096x?x1xf16>) -> tensor<5x10240x16x1xf16> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32

  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %n0 = affine.apply #map2()[%c16_vscale]

  %0 = tensor.empty(%n0, %c16_vscale) : tensor<5x?x16x?xf16>
  %1 = tensor.empty(%n0, %c16_vscale) : tensor<5x?x16x?xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<5x?x16x?xf32>) -> tensor<5x?x16x?xf32>
  %3 = linalg.mmt4d ins(%arg0, %arg1 : tensor<5x4096x16x1xf16>, tensor<?x4096x?x1xf16>) outs(%2 : tensor<5x?x16x?xf32>) -> tensor<5x?x16x?xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<5x?x16x?xf32>) outs(%0 : tensor<5x?x16x?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %7 = arith.truncf %in : f32 to f16
    linalg.yield %7 : f16
  } -> tensor<5x?x16x?xf16>
  %5 = tensor.empty() : tensor<77x10240xf16>
  %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, %c16_vscale] into %5 : tensor<5x?x16x?xf16> -> tensor<77x10240xf16>
  %6 = tensor.empty() : tensor<5x10240x16x1xf16>
  %pack = linalg.pack %unpack padding_value(%cst : f16) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %6 : tensor<77x10240xf16> -> tensor<5x10240x16x1xf16>
  return %pack : tensor<5x10240x16x1xf16>
}
// CHECK-DAG:   #[[$CONFIG0:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 16, [16]]>
// CHECK-DAG:   #[[$CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 16, [16], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
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

#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {cpu = "", cpu_features = "+v9a,+sve", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", link_embedded = false, native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android34"}>
func.func @depthwise_conv(%3: tensor<1x57x57x72xf32>, %4: tensor<3x3x72xf32>) -> tensor<1x28x28x72xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x28x28x72xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
  return %7 : tensor<1x28x28x72xf32>
}
// CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 28, 28, 8, 0, 0], vector_common_parallel = [1, 1, 4, [4], 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
// CHECK:      func.func @depthwise_conv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK:      linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Regression test. SVE isn't used (scalable vectorizaton of this op is not yet
// supported), but used to fail to compile when SVE was enabled due to tile
// sizes leading to large vectors.

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @pooling_nchw_max(%2: tensor<1x64x114x114xf32>) -> tensor<1x64x56x56xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<1x64x56x56xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %6 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%2, %4 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %6 : tensor<1x64x56x56xf32>
}
// CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 32, 56, 8, 0, 0], vector_common_parallel = [1, 2, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
// CHECK:      func.func @pooling_nchw_max
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK:      linalg.pooling_nchw_max
// CHECK-SAME:     lowering_config = #[[CONFIG]]
