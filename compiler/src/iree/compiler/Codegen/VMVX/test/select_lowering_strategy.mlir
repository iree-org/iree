// RUN: iree-opt -pass-pipeline='builtin.module(iree-vmvx-select-lowering-strategy)' -split-input-file %s | FileCheck %s

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_static() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
  %5 = tensor.empty() : tensor<384x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: func.func @matmul_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_op_dynamic() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
  %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?xi32>{%0, %1}
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<?x?xi32>{%2, %3}
  %subview = memref.subview %7[%4, %5] [%0, %1] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<?x?xi32>) outs(%subview : memref<?x?xi32, strided<[?, 1], offset: ?>>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: func.func @copy_op_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @static_1d_fft_stage2() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
  %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
  %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
  %4:2 = iree_linalg_ext.fft ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//       CHECK: func.func @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fusion_quant_matmul_generic() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c0_i32 = arith.constant 0 : i32
  %c-128_i32 = arith.constant -128 : i32
  %c1101627623_i32 = arith.constant 1101627623 : i32
  %c36_i8 = arith.constant 36 : i8
  %c127_i32 = arith.constant 127 : i32
  %c107520 = arith.constant 107520 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3360x32xi8>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32xi32>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c107520) : !flow.dispatch.tensor<readonly:tensor<32xi32>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x3360xi8>>{%1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x32xi8>>{%1}
  %7 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, 3360], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x3360xi8>>{%1} -> tensor<?x3360xi8>
  %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [3360, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3360x32xi8>> -> tensor<3360x32xi8>
  %9 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xi32>> -> tensor<32xi32>
  %10 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xi32>> -> tensor<32xi32>
  %11 = tensor.empty(%1) : tensor<?x32xi8>
  %12 = tensor.empty(%1) : tensor<?x32xi32>
  %13 = linalg.fill ins(%c0_i32 : i32) outs(%12 : tensor<?x32xi32>) -> tensor<?x32xi32>
  %14 = linalg.matmul ins(%7, %8 : tensor<?x3360xi8>, tensor<3360x32xi8>) outs(%13 : tensor<?x32xi32>) -> tensor<?x32xi32>
  %15 = linalg.generic {indexing_maps = [#map, #map1, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %14, %10 : tensor<32xi32>, tensor<?x32xi32>, tensor<32xi32>) outs(%11 : tensor<?x32xi8>) {
  ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i8):
    %16 = arith.muli %in_1, %c-128_i32 : i32
    %17 = arith.subi %in_0, %16 : i32
    %18 = arith.addi %in, %17 : i32
    %19 = tosa.apply_scale %18, %c1101627623_i32, %c36_i8 {double_round = true} : (i32, i32, i8) -> i32
    %20 = arith.addi %19, %c-128_i32 : i32
    %21 = arith.cmpi slt, %20, %c-128_i32 : i32
    %22 = arith.select %21, %c-128_i32, %20 : i32
    %23 = arith.cmpi sgt, %20, %c127_i32 : i32
    %24 = arith.select %23, %c127_i32, %22 : i32
    %25 = arith.trunci %24 : i32 to i8
    linalg.yield %25 : i8
  } -> tensor<?x32xi8>
  flow.dispatch.tensor.store %15, %6, offsets = [0, 0], sizes = [%1, 32], strides = [1, 1] : tensor<?x32xi8> -> !flow.dispatch.tensor<writeonly:tensor<?x32xi8>>{%1}
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//       CHECK: func.func @fusion_quant_matmul_generic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unpack_outer_dynamic() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
  %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
  %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
  %unpack = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
  flow.dispatch.tensor.store %unpack, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: func.func @unpack_outer_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = true}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0] -> (1024 ceildiv s0)>
#map1 = affine_map<()[s0] -> (2048 ceildiv s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_ukernels() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x2048xf32>>
  %1:2 = iree_codegen.query_tile_sizes tensor<1024x2048xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
  %2 = affine.apply #map()[%1#0]
  %3 = affine.apply #map1()[%1#1]
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%2, %3, %1#0, %1#1}
  %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x2048xf32>> -> tensor<1024x2048xf32>
  %6 = tensor.empty() : tensor<1024x2048xf32>
  %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1024x2048xf32>) outs(%6 : tensor<1024x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.addf %in, %in : f32
    linalg.yield %15 : f32
  } -> tensor<1024x2048xf32>
  %8:2 = iree_codegen.query_tile_sizes tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
  %9 = affine.apply #map()[%8#0]
  %10 = affine.apply #map1()[%8#1]
  %11 = tensor.empty(%9, %10, %8#0, %8#1) : tensor<?x?x?x?xf32>
  %pack = tensor.pack %7 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [%8#0, %8#1] into %11 : tensor<1024x2048xf32> -> tensor<?x?x?x?xf32>
  %12:2 = iree_codegen.query_tile_sizes tensor<1024x2048xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
  %13 = affine.apply #map()[%12#0]
  %14 = affine.apply #map1()[%12#1]
  flow.dispatch.tensor.store %pack, %4, offsets = [0, 0, 0, 0], sizes = [%13, %14, %12#0, %12#1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%13, %14, %12#0, %12#1}
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
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
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%6) : !flow.dispatch.tensor<writeonly:tensor<5x19x8x4xf32>>
  flow.dispatch.tensor.store %cst, %7, offsets = [0, 0, 0, 0], sizes = [5, 19, 8, 4], strides = [1, 1, 1, 1] : tensor<5x19x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<5x19x8x4xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: func.func @copy_cst
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
