// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error @+2 {{expected valid keyword}}
// expected-error @+1 {{expected a parameter name in struct}}
#encoding = #iree_encoding.encoding<>
func.func @illegal_encoding_attr_missing_operand_index(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----


// expected-error @+1 {{struct is missing required parameter: op_type}}
#encoding = #iree_encoding.encoding<operand_index = 0>
func.func @illegal_encoding_attr_missing_operand_index(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

// expected-error @+1 {{struct is missing required parameter: element_types}}
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul>
func.func @illegal_encoding_attr_missing_operand_index(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// expected-error @+1 {{`operandIndex` exceeds the size of `user_indexing_maps`}}
#encoding = #iree_encoding.encoding<operand_index = 2 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1]>
func.func @illegal_encoding_attr_with_operand_index_exceeding_indexing_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// expected-error @+1 {{found a non-composable attribute in `user_indexing_maps` at index: 2}}
#encoding = #iree_encoding.encoding<operand_index = 2 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, 1]>
func.func @illegal_encoding_attr_with_invalid_attr(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// expected-error @+1 {{found a non-composable attribute in `user_indexing_maps` at index: 0}}
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[[#map, #map1]]]>
func.func @illegal_encoding_attr_with_too_many_nested_levels(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// expected-error @+1 {{found `iteration_sizes` without any corresponding `user_indexing_maps`}}
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], iteration_sizes = [?, ?]>
func.func @illegal_encoding_attr_with_iteration_sizes_but_no_user_indexing_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// expected-error @+1 {{found an encoding with 2 iteration sizes, but expected 4 based on the user indexing maps}}
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?]>
func.func @illegal_encoding_attr_with_invalid_iteration_sizes(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

func.func @illegal_set_encoding_op_with_no_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_set_encoding_op_with_source_encoding(%arg0 : tensor<?x?xf32, #encoding>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of set_encoding op cannot have a tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32, #encoding> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_set_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, "gemm_lhs"> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, "gemm_lhs">
  return %0 : tensor<?x?xf32, "gemm_lhs">
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_set_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32>) -> tensor<?xf32, #encoding> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?xf32, #encoding>
  return %0 : tensor<?xf32, #encoding>
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_set_encoding_op_with_shape_change(%arg0 : tensor<10x20xf32>) -> tensor<20x30xf32, #encoding> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_encoding.set_encoding %arg0: tensor<10x20xf32> -> tensor<20x30xf32, #encoding>
  return %0 : tensor<20x30xf32, #encoding>
}

// -----

func.func @illegal_unset_encoding_op_with_no_source_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_unset_encoding_op_with_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  // expected-error @+1 {{result of unset_encoding op cannot have a tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// -----

func.func @illegal_unset_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32, "gemm_lhs">) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32, "gemm_lhs"> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_unset_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32, #encoding>) -> tensor<?xf32> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32, #encoding> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

#encoding = #iree_encoding.testing_encoding<>
func.func @illegal_unset_encoding_op_with_shape_change(%arg0 : tensor<20x30xf32, #encoding>) -> tensor<10x20xf32> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<20x30xf32, #encoding> -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}

// -----

// expected-error @+1 {{expected non-empty layouts}}
#encoding = #iree_encoding.layout<[]>
func.func @illegal_layout_encoding_without_any_layout(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}

// -----

// expected-error @+1 {{expected all the layout attributes to implement SerializableAttr}}
#encoding = #iree_encoding.layout<[#iree_encoding.unknown_encoding]>
func.func @illegal_layout_encoding_with_invalid_layouts(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}

// -----

// expected-error @+1 {{expected all padding values need to be non-negative or dynamic}}
#encoding = #iree_encoding.pad_encoding_layout<[0, ?, -1]>
func.func @negative_layout_encoding(%arg0: tensor<?x?x?xf32, #encoding>) -> tensor<?x?x?xf32, #encoding> {
  return %arg0 : tensor<?x?x?xf32, #encoding>
}
