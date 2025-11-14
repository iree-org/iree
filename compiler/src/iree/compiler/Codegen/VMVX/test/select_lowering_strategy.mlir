// RUN: iree-opt -pass-pipeline='builtin.module(iree-vmvx-select-lowering-strategy)' -split-input-file %s | FileCheck %s

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
func.func @matmul_static(%3: tensor<384x512xf32>, %4: tensor<512x128xf32>) -> tensor<384x128xf32> attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<384x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
  return %7 : tensor<384x128xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<distribution = [64, 64, 0]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//      CHECK: func.func @matmul_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_op_dynamic(%0: index, %1: index, %4: index, %5: index, %6: memref<?x?xi32>, %7: memref<?x?xi32>) attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %subview = memref.subview %7[%4, %5] [%0, %1] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<?x?xi32>) outs(%subview : memref<?x?xi32, strided<[?, 1], offset: ?>>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//      CHECK: func.func @copy_op_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
func.func @static_1d_fft_stage2(%2: tensor<32xf32>, %3: tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>) attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
  %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
  %4:2 = iree_linalg_ext.fft ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  return %4#0, %4#1 : tensor<32xf32>, tensor<32xf32>
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//       CHECK: func.func @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fusion_quant_matmul_generic(%1: index, %7: tensor<?x3360xi8>, %8: tensor<3360x32xi8>, %9: tensor<32xi32>, %10: tensor<32xi32>) -> tensor<?x32xi8> attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c0_i32 = arith.constant 0 : i32
  %c-128_i32 = arith.constant -128 : i32
  %c1101627623_i32 = arith.constant 1101627623 : i32
  %c36_i8 = arith.constant 36 : i8
  %c127_i32 = arith.constant 127 : i32
  %11 = tensor.empty(%1) : tensor<?x32xi8>
  %12 = tensor.empty(%1) : tensor<?x32xi32>
  %13 = linalg.fill ins(%c0_i32 : i32) outs(%12 : tensor<?x32xi32>) -> tensor<?x32xi32>
  %14 = linalg.matmul ins(%7, %8 : tensor<?x3360xi8>, tensor<3360x32xi8>) outs(%13 : tensor<?x32xi32>) -> tensor<?x32xi32>
  %15 = linalg.generic {indexing_maps = [#map, #map1, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %14, %10 : tensor<32xi32>, tensor<?x32xi32>, tensor<32xi32>) outs(%11 : tensor<?x32xi8>) {
  ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i8):
    %16 = arith.muli %in_1, %c-128_i32 : i32
    %17 = arith.subi %in_0, %16 : i32
    %18 = arith.addi %in, %17 : i32
    %19 = tosa.apply_scale %18, %c1101627623_i32, %c36_i8 {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
    %20 = arith.addi %19, %c-128_i32 : i32
    %21 = arith.cmpi slt, %20, %c-128_i32 : i32
    %22 = arith.select %21, %c-128_i32, %20 : i32
    %23 = arith.cmpi sgt, %20, %c127_i32 : i32
    %24 = arith.select %23, %c127_i32, %22 : i32
    %25 = arith.trunci %24 : i32 to i8
    linalg.yield %25 : i8
  } -> tensor<?x32xi8>
  return %15 : tensor<?x32xi8>
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//       CHECK: func.func @fusion_quant_matmul_generic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
func.func @unpack_outer_dynamic(%6: index, %7: index, %10: tensor<?x?x32x16xi32>) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
  %unpack = linalg.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//      CHECK: func.func @unpack_outer_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = true}>
#map = affine_map<()[s0] -> (1024 ceildiv s0)>
#map1 = affine_map<()[s0] -> (2048 ceildiv s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_ukernels(%5: tensor<1024x2048xf32>) -> tensor<?x?x?x?xf32> attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %6 = tensor.empty() : tensor<1024x2048xf32>
  %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1024x2048xf32>) outs(%6 : tensor<1024x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.addf %in, %in : f32
    linalg.yield %15 : f32
  } -> tensor<1024x2048xf32>
  %8:2 = iree_codegen.query_tile_sizes tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> -> index, index
  %9 = affine.apply #map()[%8#0]
  %10 = affine.apply #map1()[%8#1]
  %11 = tensor.empty(%9, %10, %8#0, %8#1) : tensor<?x?x?x?xf32>
  %pack = linalg.pack %7 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [%8#0, %8#1] into %11 : tensor<1024x2048xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//      CHECK: func.func @elem_pack_ukernels
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @copy_cst() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %cst = arith.constant dense<4.200000e-01> : tensor<5x19x8x4xf32>
  %c32_i64 = arith.constant 32 : i64
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.shli %3, %c32_i64 : i64
  %5 = arith.ori %2, %4 : i64
  %6 = arith.index_castui %5 : i64 to index
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%6) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x19x8x4xf32>>
  iree_tensor_ext.dispatch.tensor.store %cst, %7, offsets = [0, 0, 0, 0], sizes = [5, 19, 8, 4], strides = [1, 1, 1, 1] : tensor<5x19x8x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x19x8x4xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = VMVXDefault>
//      CHECK: func.func @copy_cst
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
