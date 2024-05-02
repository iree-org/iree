// RUN: iree-opt --split-input-file --verify-diagnostics %s

func.func @illegal_set_encoding_op_with_no_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_set_encoding_op_with_source_encoding(%arg0 : tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of set_encoding op cannot have a tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_set_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, "gemm_lhs"> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, "gemm_lhs">
  return %0 : tensor<?x?xf32, "gemm_lhs">
}

// -----

func.func @illegal_set_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32>) -> tensor<?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_encoding.set_encoding %arg0: tensor<?x?xf32> -> tensor<?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
  return %0 : tensor<?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
}

// -----

func.func @illegal_set_encoding_op_with_shape_change(%arg0 : tensor<10x20xf32>) -> tensor<20x30xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_encoding.set_encoding %arg0: tensor<10x20xf32> -> tensor<20x30xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
  return %0 : tensor<20x30xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
}

// -----

func.func @illegal_unset_encoding_op_with_no_source_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  // expected-error @+1 {{result of unset_encoding op cannot have a tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
  return %0 : tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>
}

// -----

func.func @illegal_unset_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32, "gemm_lhs">) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32, "gemm_lhs"> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>) -> tensor<?xf32> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<?x?xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_shape_change(%arg0 : tensor<20x30xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>>) -> tensor<10x20xf32> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_encoding.unset_encoding %arg0: tensor<20x30xf32, #iree_encoding.encoding<role = LHS, element_types = [f32, f32, f32]>> -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}
