// RUN: iree-opt  --pass-pipeline="builtin.module(iree-vmvx-select-lowering-strategy, func.func(iree-vmvx-lower-executable-target))" --split-input-file %s | FileCheck %s

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0] -> (16 ceildiv s0)>

// CHECK: func @mmt4d_i8()
func.func @mmt4d_i8() attributes {hal.executable.target = #executable_target_vmvx_bytecode_fb} {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c16 = arith.constant 16 : index
  %0:2 = iree_codegen.query_tile_sizes tensor<16x16xi8, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %1 = affine.apply #map3()[%0#0]
  %2 = affine.apply #map3()[%0#1]
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%1, %2, %0#0, %0#1}
  %4:2 = iree_codegen.query_tile_sizes tensor<16x16xi8, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %5 = affine.apply #map3()[%4#0]
  %6 = affine.apply #map3()[%4#1]
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c256) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%5, %6, %4#0, %4#1}
  %8:2 = iree_codegen.query_tile_sizes tensor<16x16xi32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %9 = affine.apply #map3()[%8#0]
  %10 = affine.apply #map3()[%8#1]
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c512) : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1}
  %12 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%1, %2, %0#0, %0#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%1, %2, %0#0, %0#1} -> tensor<?x?x?x?xi8>
  %13 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [%5, %6, %4#0, %4#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%5, %6, %4#0, %4#1} -> tensor<?x?x?x?xi8>
  %14 = flow.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%9, %10, %8#0, %8#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1} -> tensor<?x?x?x?xi32>
  // CHECK: iree_codegen.ukernel.generic "vmvx.mmt4d"
  %15 = linalg.mmt4d ins(%12, %13 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>) outs(%14 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  flow.dispatch.tensor.store %15, %11, offsets = [0, 0, 0, 0], sizes = [%9, %10, %8#0, %8#1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1}
  return
}
