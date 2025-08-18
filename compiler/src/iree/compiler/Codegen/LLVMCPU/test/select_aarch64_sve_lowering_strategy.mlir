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
func.func @unpack_generic(%arg0 : tensor<128x10x?x8x?xf32>, %arg1 : tensor<128x320xf32>) -> tensor<128x320x80xf32> attributes {hal.executable.target = #executable_target_system_elf_arm_64_} {
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
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8, [8]]>
//   CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 40], vector_common_parallel = [1, [8], 8]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert
//       CHECK: func.func @unpack_generic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[CONFIG1]]

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
