// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding),canonicalize,cse)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#batch_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#batch_map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#input_encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul,
    element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2],
    iteration_sizes = [2, 6, 3]>
#weights_encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul,
    element_types = [f32, f32, f32],
    user_indexing_maps = [#batch_map, #batch_map1, #batch_map2],
    iteration_sizes = [4, 1, 6, 3]>
#output_encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul,
    element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2],
    iteration_sizes = [2, 6, 3]>
#narrow_input_encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul,
    element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2],
    iteration_sizes = [2, 1, 3]>
#narrow_weights_encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul,
    element_types = [f32, f32, f32],
    user_indexing_maps = [#batch_map, #batch_map1, #batch_map2],
    iteration_sizes = [4, 2, 1, 3]>
#narrow_output_encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul,
    element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2],
    iteration_sizes = [2, 1, 3]>

func.func @group_matmul(%input: tensor<2x3xf32, #input_encoding>,
                        %weights: tensor<4x3x6xf32, #weights_encoding>,
                        %offsets: tensor<4xi64>, %row_offset: index,
                        %output: tensor<2x6xf32, #output_encoding>)
    -> tensor<2x6xf32, #output_encoding> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %result = iree_linalg_ext.group_matmul ins(
      %input, %weights, %offsets, %row_offset : tensor<2x3xf32, #input_encoding>,
      tensor<4x3x6xf32, #weights_encoding>, tensor<4xi64>, index)
    outs(%output : tensor<2x6xf32, #output_encoding>)
    -> tensor<2x6xf32, #output_encoding>
  return %result : tensor<2x6xf32, #output_encoding>
}

// CHECK-LABEL: func.func @group_matmul
// CHECK: iree_linalg_ext.group_mmt4d
// CHECK-SAME: tensor<1x1x2x4xf32>
// CHECK-SAME: tensor<4x1x1x8x4xf32>
// CHECK-SAME: tensor<1x1x2x8xf32>
// CHECK-NOT: transposed = true

// -----

func.func @group_matmul_narrow_n(%input: tensor<2x3xf32, #narrow_input_encoding>,
                                 %weights: tensor<4x3x1xf32, #narrow_weights_encoding>,
                                 %offsets: tensor<4xi64>, %row_offset: index,
                                 %output: tensor<2x1xf32, #narrow_output_encoding>)
    -> tensor<2x1xf32, #narrow_output_encoding> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %result = iree_linalg_ext.group_matmul ins(
      %input, %weights, %offsets, %row_offset : tensor<2x3xf32, #narrow_input_encoding>,
      tensor<4x3x1xf32, #narrow_weights_encoding>, tensor<4xi64>, index)
    outs(%output : tensor<2x1xf32, #narrow_output_encoding>)
    -> tensor<2x1xf32, #narrow_output_encoding>
  return %result : tensor<2x1xf32, #narrow_output_encoding>
}

// CHECK-LABEL: func.func @group_matmul_narrow_n
// CHECK: iree_linalg_ext.group_mmt4d {transposed = true}
// CHECK-SAME: tensor<1x1x8x4xf32>
// CHECK-SAME: tensor<4x1x1x1x4xf32>
// CHECK-SAME: tensor<1x1x1x8xf32>
