// Tests for RISC-V RVV targets with scalable vectorization enabled.
// These tests verify that the lowering strategy correctly handles
// mmt4d, pack, and unpack ops with scalable inner tile sizes.
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s

// This case tests if the inner tile size of the mmt4d is inferred properly
// from the shape-aware HAL binding and set accordingly.

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map = affine_map<()[s0] -> (256 ceildiv s0)>
func.func @mmt4d_tensors(%arg0: tensor<32x128x7x1xf32>, %arg1 : tensor<?x128x?x1xf32>) -> tensor<32x?x7x?xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %n0 = affine.apply #map()[%c8_vscale]
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x?x7x?xf32>>{%n0, %c8_vscale}
  %init = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, %n0, 7, %c8_vscale], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x?x7x?xf32>>{%n0, %c8_vscale} -> tensor<32x?x7x?xf32>
  %mmt4d = linalg.mmt4d ins(%arg0, %arg1 : tensor<32x128x7x1xf32>, tensor<?x128x?x1xf32>) outs(%init : tensor<32x?x7x?xf32>) -> tensor<32x?x7x?xf32>
  return %mmt4d : tensor<32x?x7x?xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 7, [8], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
//       CHECK: func.func @mmt4d_tensors
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.mmt4d
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map = affine_map<()[s0] -> (256 ceildiv s0)>
func.func @mmt4d_with_fill(%arg0 : tensor<32x128x7x1xf32>, %arg1 : tensor<?x128x?x1xf32>) -> tensor<32x?x7x?xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %0 = affine.apply #map()[%c8_vscale]
  %init = tensor.empty(%0, %c8_vscale) : tensor<32x?x7x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x?x7x?xf32>) -> tensor<32x?x7x?xf32>
  %mmt4d = linalg.mmt4d ins(%arg0, %arg1 : tensor<32x128x7x1xf32>, tensor<?x128x?x1xf32>) outs(%fill : tensor<32x?x7x?xf32>) -> tensor<32x?x7x?xf32>
  return %mmt4d : tensor<32x?x7x?xf32>
}
//   CHECK-DAG: #[[CONFIG1:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 7, [8]]>
//   CHECK-DAG: #[[CONFIG2:.+]] = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 7, [8], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
//       CHECK: func.func @mmt4d_with_fill
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.fill
//  CHECK-SAME:     lowering_config = #[[CONFIG1]]
//       CHECK: linalg.mmt4d
//  CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map = affine_map<()[s0] -> (320 ceildiv s0)>
func.func @unpack(%arg0 : tensor<128x10x?x8x?xf32>) -> tensor<128x80x320xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %init = tensor.empty() : tensor<128x80x320xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, %c8_vscale] into %init : tensor<128x10x?x8x?xf32> -> tensor<128x80x320xf32>
  return %unpack : tensor<128x80x320xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 40, 64], vector_common_parallel = [1, 8, [8]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DataTiling>>
//
// For RVV, we do not decompose unpacks.
//   CHECK-NOT: enable_loop_peeling
//       CHECK: func.func @unpack
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @unpack_outer_dynamic(%arg0 : tensor<?x?x32x?xf32>, %dim0 : index, %dim1 : index) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, %c8_vscale] into %init : tensor<?x?x32x?xf32> -> tensor<?x?xf32>
  return %unpack : tensor<?x?xf32>
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [32, [8]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DataTiling>>
//
// For RVV, we do not decompose unpacks.
//   CHECK-NOT: enable_loop_peeling
//       CHECK: func.func @unpack_outer_dynamic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.unpack
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map_pack = affine_map<()[s0] -> (48 ceildiv s0)>
func.func @pack(%arg0: tensor<20x48xf32>) -> tensor<2x?x16x?xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c16_vscale = arith.muli %vscale, %c16 : index
  %outer1 = affine.apply #map_pack()[%c16_vscale]
  %empty = tensor.empty(%outer1, %c16_vscale) : tensor<2x?x16x?xf32>
  %pack = linalg.pack %arg0 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, %c16_vscale] into %empty : tensor<20x48xf32> -> tensor<2x?x16x?xf32>
  return %pack : tensor<2x?x16x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 64], vector_common_parallel = [1, 1]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DataTiling>>
//      CHECK: func.func @pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map_elem_pack = affine_map<()[s0] -> (384 ceildiv s0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack(%arg0: tensor<128x384xf32>) -> tensor<16x?x8x?xf32> attributes {hal.executable.target = #executable_target_riscv64} {
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
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//      CHECK: func.func @elem_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+zvfh,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<()[s0] -> (10240 ceildiv s0)>
func.func @mmt4d_generic_unpack_pack(%arg0: tensor<5x4096x7x1xf16>, %arg1: tensor<?x4096x?x1xf16>) -> tensor<5x10240x7x1xf16> attributes {hal.executable.target = #executable_target_riscv64} {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32

  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %n0 = affine.apply #map2()[%c8_vscale]

  %0 = tensor.empty(%n0, %c8_vscale) : tensor<5x?x7x?xf16>
  %1 = tensor.empty(%n0, %c8_vscale) : tensor<5x?x7x?xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<5x?x7x?xf32>) -> tensor<5x?x7x?xf32>
  %3 = linalg.mmt4d ins(%arg0, %arg1 : tensor<5x4096x7x1xf16>, tensor<?x4096x?x1xf16>) outs(%2 : tensor<5x?x7x?xf32>) -> tensor<5x?x7x?xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<5x?x7x?xf32>) outs(%0 : tensor<5x?x7x?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %7 = arith.truncf %in : f32 to f16
    linalg.yield %7 : f16
  } -> tensor<5x?x7x?xf16>
  %5 = tensor.empty() : tensor<33x10240xf16>
  %unpack = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [7, %c8_vscale] into %5 : tensor<5x?x7x?xf16> -> tensor<33x10240xf16>
  %6 = tensor.empty() : tensor<5x10240x7x1xf16>
  %pack = linalg.pack %unpack padding_value(%cst : f16) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [7, 1] into %6 : tensor<33x10240xf16> -> tensor<5x10240x7x1xf16>
  return %pack : tensor<5x10240x7x1xf16>
}
// CHECK-DAG:   #[[$CONFIG0:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 7, [8]]>
// CHECK-DAG:   #[[$CONFIG1:.+]] = #iree_cpu.lowering_config<distribution = [5, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 7, [8], 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
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
