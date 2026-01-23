// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS
#define IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS

include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MatchCastCompatibleDagFromRootOp : Op<Transform_Dialect, "iree.match.cast_compatible_dag_from_root",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleOpMatcher,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
     MemoryEffectsOpInterface]> {
  let summary = [{
      Checks if the body of the target op matches an operation dag starting
      at the given root.
  }];
  let description = [{
    Checks whether the given root op matches an operation dag specified in the
    body of this op. Enforces cast compatibilty between types rather than a
    strict equality, similar to `iree.match.cast_compatible_type`.

    Note: This operation is experimental and subject to change. General subgraph
    matching is difficult and can spawn various DSLs and a slew of transforms.
    This op tries to keep it relatively simple an inflexible, reflecting the
    expected use case of splicing in hand written kernels that can be equally
    inflexible.

    #### Return modes

    Succeeds if the root operation matches the dag given by this op, and
    produces a silenceable failure otherwise. Produces a definite failure
    if the operand is not associated with a single payload value.

    On success, this operation produces a handle to the inputs and outputs
    of the operation dag based on the outputs of the root op and the block
    arguments of this operations body.
  }];

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformValueHandleTypeInterface:$inputs,
                      TransformValueHandleTypeInterface:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$operand_handle attr-dict-with-keyword regions `:` functional-type(operands, results)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let hasVerifier = 1;
}

def MatchCastCompatibleTypesOp : Op<Transform_Dialect, "iree.match.cast_compatible_type",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleValueMatcher,
     MemoryEffectsOpInterface]> {
  let summary = [{Checks if the payload value is cast compatible with a target type.}];
  let description = [{
    Checks whether the given value is cast-compatible with the given target
    type attribute.

    Currently this operation only allows casting of tensor types. Other types
    must match exactly.

    #### Return modes

    Succeeds if the value's type is compatible with the target type, and
    produces a silenceable failure otherwise. Produces a definite failure
    if the operand is not associated with a single payload value.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$operand_handle,
                       TypeAttr:$target_type);
  let assemblyFormat = "$operand_handle `=` $target_type attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleValueMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchDimBoundsOp : Op<Transform_Dialect, "iree.match.dim_bounds",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleValueMatcher,
     MemoryEffectsOpInterface]> {
  let summary = [{Checks whether the size of a dim is within given bounds.}];
  let description = [{
    Checks whether a dim is within a specified lower and upper bound.

    #### Return modes

    Succeeds if the value's type is compatible with the target type, and
    produces a silenceable failure otherwise. Produces a definite failure
    if the operand is not associated with a single payload value.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$operand_handle,
                       I64Attr:$dim,
                       OptionalAttr<I64Attr>:$lower_bound,
                       OptionalAttr<I64Attr>:$upper_bound);

  // `<=` isn't a valid literal so use `le` instead.
  let assemblyFormat = [{
    $operand_handle `[` $dim `]` `,` (`umin` `=` $lower_bound^):(`none`)?
    `,` (`umax` `=` $upper_bound^):(`none`)?  attr-dict `:` type($operand_handle)
  }];
  let extraClassDeclaration = SingleValueMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

// TODO: Combine with MatchDimBounds.
def MatchDimIsMultipleOfOp : Op<Transform_Dialect, "iree.match.dim_is_multiple_of",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleValueMatcher,
     MemoryEffectsOpInterface]> {
  let summary = [{Checks the static size of a dim is divisible by a given value.}];
  let description = [{
    Checks whether the given dimension given shaped value is a multiple of the
    given size.

    #### Return modes

    Succeeds if the value's type is compatible with the target type, and
    produces a silenceable failure otherwise. Produces a definite failure
    if the operand is not associated with a single payload value.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$operand_handle,
                       I64Attr:$dim,
                       I64Attr:$size);
  let assemblyFormat = "$operand_handle `[` $dim `]` `,` $size "
                       "attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleValueMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchRegionsOp : Op<Transform_Dialect, "iree.match.regions",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleOpMatcher,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
     MemoryEffectsOpInterface]> {
  let summary = [{Checks if the body of the target op matches the body of the single contained op.}];
  let description = [{
    Does a structural comparison of the regions of the single op contained
    within the region of this op against the regions of the target operation.

    #### Return modes

    Succeeds if the operation body satisfies the specified criteria, produces a
    silenceable failure otherwise. Produces a definite failure if the operand is
    not associated with a single payload op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$operand_handle attr-dict `:` type($operand_handle) regions";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let hasVerifier = 1;
}

def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
    [MatchOpInterface,
     SingleOpMatcher,
     MemoryEffectsOpInterface,
     AllTypesMatch<["batch_dims", "m_dims", "n_dims", "k_dims"]>
     ]> {
  let summary = [{Check whether the op is a contraction operation.}];
  let description = [{
    Matches operations that implement the ContractionOpInterface.
    This includes operations like linalg.matmul, linalg.batch_matmul, etc.

    Optionally matches specific indexing maps patterns.

    ### Example

    ```mlir
    #map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
    #map_rhs = affine_map<(d0, d1, d2) -> (d1, d2)>
    #map_output = affine_map<(d0, d1, d2) -> (d0, d1)>

    %batch_dims, %m_dims, %n_dims, %k_dims =
      transform.iree.match.contraction %matmul_op,
        lhs_type = f32, rhs_type = f32, output_type = f32,
        indexing_maps = [#map_lhs, #map_rhs, #map_output] :
        !transform.any_op -> !transform.param<i64>
    ```

    This succeeds when `%matmul_op` is a contraction operation with f32 input
    types and f32 output type, and matches the specified indexing maps pattern.

    #### Return modes

    Succeeds if the operation is a contraction operation, and
    produces a silenceable failure otherwise.

    #### Results

    Returns arrays of dimension sizes for each contraction dimension:
    - batch_dims: Array of batch dimension sizes.
    - m_dims: Array of M dimension sizes.
    - n_dims: Array of N dimension sizes.
    - k_dims: Array of K dimension sizes.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$operand_handle,
    TypeAttr:$lhs_type,                    // LHS input type.
    TypeAttr:$rhs_type,                    // RHS input type.
    TypeAttr:$output_type,                 // Output type.
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps
  );

  let results = (outs
    TransformParamTypeInterface:$batch_dims,
    TransformParamTypeInterface:$m_dims,
    TransformParamTypeInterface:$n_dims,
    TransformParamTypeInterface:$k_dims
  );

  let assemblyFormat = [{
    $operand_handle
    `,` `lhs_type` `=` $lhs_type
    `,` `rhs_type` `=` $rhs_type
    `,` `output_type` `=` $output_type
    (`,` `indexing_maps` `=` $indexing_maps^)?
    attr-dict `:` type($operand_handle) `->` type($batch_dims)
  }];
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchConvolutionOp : Op<Transform_Dialect, "iree.match.convolution",
    [MatchOpInterface,
     SingleOpMatcher,
     MemoryEffectsOpInterface,
     AllTypesMatch<["batch_dims", "output_image_dims", "output_channel_dims",
                    "filter_dims", "input_channel_dims", "depth_dims",
                    "strides", "dilations"]>
     ]> {
  let summary = [{Check whether the op is a convolution operation.}];
  let description = [{
    Matches operations that implement the ConvolutionOpInterface.
    This includes operations like linalg.conv_2d_nhwc_hwcf,
    linalg.conv_2d_nchw_fchw, linalg.depthwise_conv_2d_nhwc_hwc, etc.

    Optionally matches specific indexing maps patterns.

    ### Example

    ```mlir
    #map_input = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
    #map_filter = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
    #map_output = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

    %batch_dims, %output_image_dims, %output_channel_dims, %filter_dims,
    %input_channel_dims, %depth_dims, %strides, %dilations =
      transform.iree.match.convolution %conv_op,
        lhs_type = f32, rhs_type = f32, output_type = f32,
        indexing_maps = [#map_input, #map_filter, #map_output] :
        !transform.any_op -> !transform.param<i64>
    ```

    This succeeds when `%conv_op` is a convolution operation with f32 element
    types for input, filter, and output tensors.

    #### Return modes

    Succeeds if the operation is a convolution operation, and
    produces a silenceable failure otherwise.

    #### Results

    Returns arrays of dimension sizes for each convolution dimension:
    - batch_dims: Array of batch dimension sizes.
    - output_image_dims: Array of output spatial dimension sizes.
    - output_channel_dims: Array of output channel dimension sizes.
    - filter_dims: Array of filter spatial dimension sizes.
    - input_channel_dims: Array of input channel dimension sizes.
    - depth_dims: Array of depth dimension sizes (for depthwise convolutions).
    - strides: Array of stride values.
    - dilations: Array of dilation values.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$operand_handle,
    TypeAttr:$lhs_type,
    TypeAttr:$rhs_type,
    TypeAttr:$output_type,
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps
  );

  let results = (outs
    TransformParamTypeInterface:$batch_dims,
    TransformParamTypeInterface:$output_image_dims,
    TransformParamTypeInterface:$output_channel_dims,
    TransformParamTypeInterface:$filter_dims,
    TransformParamTypeInterface:$input_channel_dims,
    TransformParamTypeInterface:$depth_dims,
    TransformParamTypeInterface:$strides,
    TransformParamTypeInterface:$dilations
  );

  let assemblyFormat = [{
    $operand_handle
    `,` `lhs_type` `=` $lhs_type
    `,` `rhs_type` `=` $rhs_type
    `,` `output_type` `=` $output_type
    (`,` `indexing_maps` `=` $indexing_maps^)?
    attr-dict `:` type($operand_handle) `->` type($batch_dims)
  }];
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchAttentionOp : Op<Transform_Dialect, "iree.match.attention",
    [MatchOpInterface,
     SingleOpMatcher,
     MemoryEffectsOpInterface,
     AllTypesMatch<["batch_dims", "m_dims", "n_dims",
                    "k1_dims", "k2_dims"]>
     ]> {
  let summary = [{Check whether the op is an attention operation.}];
  let description = [{
    Matches operations from the IREELinalgExt dialect that implement
    attention: iree_linalg_ext.attention.

    ### Example

    ```mlir
    #map_query = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
    #map_key = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
    #map_value = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
    #map_scale = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
    #map_output = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

    %batch_dims, %m_dims, %n_dims, %k1_dims, %k2_dims =
      transform.iree.match.attention %attn_op,
        query_type = f32, key_type = f32, value_type = f32, output_type = f32,
        indexing_maps = [#map_query, #map_key, #map_value, #map_scale, #map_output] :
        !transform.any_op -> !transform.param<i64>
    ```

    This succeeds when `%attn_op` is an attention operation with f32 element
    types for query, key, value, and output tensors, and matches the specified
    indexing maps pattern.

    #### Return modes

    Succeeds if the operation is an attention operation, and
    produces a silenceable failure otherwise.

    #### Results

    Returns arrays of dimension sizes extracted from the iteration domain:
    - batch_dims: Array of batch dimension sizes.
    - m_dims: Array of query sequence length dimension sizes.
    - n_dims: Array of number of heads dimension sizes.
    - k1_dims: Array of key/value sequence length dimension sizes.
    - k2_dims: Array of key embedding dimension sizes.

    The exact interpretation depends on the indexing maps of the attention op.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$operand_handle,
    TypeAttr:$query_type,
    TypeAttr:$key_type,
    TypeAttr:$value_type,
    TypeAttr:$output_type,
    AffineMapArrayAttr:$indexing_maps
  );

  let results = (outs
    TransformParamTypeInterface:$batch_dims,
    TransformParamTypeInterface:$m_dims,
    TransformParamTypeInterface:$n_dims,
    TransformParamTypeInterface:$k1_dims,
    TransformParamTypeInterface:$k2_dims
  );

  let assemblyFormat = [{
    $operand_handle
    `,` `query_type` `=` $query_type
    `,` `key_type` `=` $key_type
    `,` `value_type` `=` $value_type
    `,` `output_type` `=` $output_type
    `,` `indexing_maps` `=` $indexing_maps
    attr-dict `:` type($operand_handle) `->` type($batch_dims)
  }];
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchDimsEqualOp : Op<Transform_Dialect, "iree.match.dims_equal",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{Check whether transform parameters match expected size values exactly.}];
  let description = [{
    Matches dimension sizes against expected values.
    Each position in the dimension sizes array must match the corresponding
    expected value exactly.

    ### Example

    ```mlir
    transform.iree.match.size_equals %m, [512, 256] : !transform.param<i64>
    ```

    This succeeds when `%m` has exactly two dimensions, 512 and 256.

    #### Return modes

    Succeeds if all parameters match their corresponding expected values, and
    produces a silenceable failure otherwise.
  }];

  let arguments = (ins
    TransformParamTypeInterface:$dimension_sizes, // Array of dimension parameters.
    DenseI64ArrayAttr:$expected_values            // Array of expected i64 values.
  );

  let assemblyFormat = [{
    $dimension_sizes `,` $expected_values
    attr-dict `:` type($dimension_sizes)
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif // IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS
