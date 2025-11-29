// RUN: iree-opt --split-input-file --verify-diagnostics %s

util.func public @barrier_shape_mismatch(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier' op value and result types must match}}
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<8x4xf32>
  util.return %0 : tensor<8x4xf32>
}

// -----

util.func public @barrier_missing_dynamic_dims(%arg0: tensor<?x?xf32>, %dim0: index) -> tensor<?x?xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier' op value set has 2 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<?x?xf32>{%dim0} -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

util.func public @barrier_static_with_dims(%arg0: tensor<4x8xf32>, %dim0: index) -> tensor<4x8xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier' op value set has 0 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32>{%dim0} -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}

// -----

//===----------------------------------------------------------------------===//
// iree_tensor_ext.ragged_shape
//===----------------------------------------------------------------------===//

// Check that the sparse dimensions are in-bounds of the rank.
// expected-error @+1{{sparse dimensions specified are greater than the rank of the shaped type}}
util.func public @test(memref<?xf32, #iree_tensor_ext.ragged_shape<0>>) {
  return
}

// -----

//===----------------------------------------------------------------------===//
// iree_tensor_ext.cast_to_ragged_shape
//===----------------------------------------------------------------------===//

// Error if the ragged_dim value is greater than the rank of the source.
util.func public @raggedDimError(%source : tensor<10x20x30xf32>,
    %columnLengths : tensor<4xindex>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected `ragged_dim` to be less than 3, i.e the rank of source}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(3) column_lengths(%columnLengths)
      : (tensor<10x20x30xf32>, tensor<4xindex>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error if the rank of the `column_lengths` is not 1.
util.func public @zeroDColumnLengths(%source : tensor<10x20x30xf32>,
    %columnLengths : tensor<index>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected `column_lengths` to be of rank 1}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<10x20x30xf32>, tensor<index>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error if the rank of the `column_lengths` is greater than 1.
util.func public @twoDcolumnLengths(%source : tensor<10x20x30xf32>,
    %columnLengths : tensor<4x2xindex>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected `column_lengths` to be of rank 1}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<10x20x30xf32>, tensor<4x2xindex>) -> tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<10x3x8x30xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error if the result rank is not one more than the source rank.
util.func public @resultRankSameAsSourceRank(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected result rank to be 4, i.e. one more than the source rank, but got 3}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error if the result rank is not one more than the source rank.
util.func public @resultRankTwoMoreThanSourceRank(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected result rank to be 4, i.e. one more than the source rank, but got 5}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error if the result does not have a `iree_tensor_ext.ragged_shape` attribute.
util.func public @missingRaggedShapeAttr(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{expected result type to have an encoding attribute that implements the `SparseShapeAttrInterface`}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?xf32>
}

// -----

// Error if the result does not have a `iree_tensor_ext.ragged_shape` attribute.
util.func public @wrongEncodingAttr(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?xf32, "foo"> {
  // expected-error @+1 {{expected result type to have an encoding attribute that implements the `SparseShapeAttrInterface`}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?x?xf32, "foo">
  util.return %0 : tensor<?x?x?x?xf32, "foo">
}

// -----

// Error if the result does not have a `iree_tensor_ext.ragged_shape` attribute.
util.func public @wrongEncodingAttr(%source : memref<?x?x?xf32>,
    %columnLengths : memref<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> memref<?x?x?x?xf32> {
  // expected-error @+1 {{expected result type to have a layout attribute that implements the `SparseShapeAttrInterface`}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (memref<?x?x?xf32>{%d0, %d1, %d2}, memref<?xindex>)
      -> memref<?x?x?x?xf32>
  util.return %0 : memref<?x?x?x?xf32>
}

// -----

// Mismatched between `ragged_shape` attr used and `ragged_dim` specified.
util.func public @mismatchedRaggedShapeAttr(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<0>> {
  // expected-error @+1 {{mismatch in specified `ragged_dim` value of 1 and `raggedRow` value in the sparse encoding 0}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  util.return %0 : tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
}

// -----

// Error when `column_lengths` is static shaped when `num_ragged_rows` is dynamic.
util.func public @staticColumnLenghtsWithDynamicNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{invalid to have static dimensions for `column_lengths` when `num_ragged_rows` is dynamic}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>)
      -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `result` has static shape for `ragged_dim` when `num_ragged_rows` is dynamic.
util.func public @staticResultDimWithDynamicNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>, %numRaggedRows : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{invalid to have static value for dimension 1 of result when `num_ragged_rows` is specified as dynamic value}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      num_ragged_rows(%numRaggedRows) : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>)
      -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `num_ragged_rows` is unspecified (i.e. it is static) but `column_lengths` is
// static. When `num_ragged_rows` is not specified, the size of `column_lengths` gives
// the number of rows.
util.func public @dynamicColumnLengthWithStaticNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<?xindex>,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of `column_lengths` to be static and greater than 1 when `num_ragged_rows` is unspecified, i.e. number of ragged rows is statically known}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?xindex>) -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `num_ragged_rows` is unspecified (i.e. it is static), the size of
// `column_lengths` is less then or equal to 1.
util.func public @unitColumnLengthWithStaticNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<1xindex>,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of `column_lengths` to be static and greater than 1 when `num_ragged_rows` is unspecified, i.e. number of ragged rows is statically known}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<1xindex>) -> tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x2x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `ragged_dim` dimension of result is dynamic with static `column_lengths`.
util.func public @dynamicResultDimWithStaticNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of dimension 1 of result, i.e. the `ragged_dim`, to be static and equal to 3 when `num_ragged_rows` is unspecified, i.e number of ragged rows is statically known}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `ragged_dim` dimension of result is mismatched with static `column_lengths` shape.
util.func public @dynamicResultDimWithStaticNumRaggedRows(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x4x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of dimension 1 of result, i.e. the `ragged_dim`, to be static and equal to 3 when `num_ragged_rows` is unspecified, i.e number of ragged rows is statically known}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x4x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x4x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when `ragged_dim` + 1 dimension is static.
util.func public @staticRaggedColumnLengths(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %avgColumnLength : index,
    %d0 : index, %d1 : index, %d2 : index) -> tensor<?x3x4x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected dimension 2 of result, i.e. `ragged_dim` + 1, to be dynamic}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      avg_ragged_column_length(%avgColumnLength)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x3x4x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x4x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result.
util.func public @mismatchSourceAndResultDims0(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index, %d2 : index) -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected dimension 0 of result to be dynamic since the corresponding dimension 0 in the source is dynamic}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result

util.func public @mismatchSourceAndResultDims1(%source : tensor<4x?x?xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index)
    -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of dimension 0 of result to match the shape of dimension 0 of source, but got 2 and 4 respectively}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<4x?x?xf32>{%d0, %d1}, tensor<4xindex>) -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result

util.func public @mismatchSourceAndResultDims2(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index, %d2 : index)
    -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected dimension 0 of result to be dynamic since the corresponding dimension 0 in the source is dynamic}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<2x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result

util.func public @mismatchSourceAndResultDims3(%source : tensor<?x?x4xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index)
    -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of dimension 3 of result to match the shape of dimension 2 of source, but got ? and 4 respectively}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x4xf32>{%d0, %d1}, tensor<4xindex>) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result

util.func public @mismatchSourceAndResultDims4(%source : tensor<?x?x4xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index)
    -> tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected shape of dimension 3 of result to match the shape of dimension 2 of source, but got 5 and 4 respectively}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x4xf32>{%d0, %d1}, tensor<4xindex>) -> tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Error when mismatch between dimensions of source and result

util.func public @mismatchSourceAndResultDims4(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index, %d2 : index)
    -> tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{expected dimension 3 of result to be dynamic since the corresponding dimension 2 in the source is dynamic}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<4xindex>) -> tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Unspecified dynamic dimensions of source

util.func public @insufficientSourceDynamicDims(%source : tensor<?x?x?xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index)
    -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{mismatch in number of dynamic dimensions specified for source, expected 3 values, got 2}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<?x?x?xf32>{%d0, %d1}, tensor<4xindex>) -> tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<?x3x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
}

// -----

// Overspecified dynamic dimensions of source

util.func public @extraSourceDynamicDims(%source : tensor<2x?x5xf32>,
    %columnLengths : tensor<4xindex>, %d0 : index, %d1 : index, %d2 : index)
    -> tensor<2x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>> {
  // expected-error @+1 {{mismatch in number of dynamic dimensions specified for source, expected 1 values, got 2}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1) column_lengths(%columnLengths)
      : (tensor<2x?x5xf32>{%d0, %d1}, tensor<4xindex>) -> tensor<2x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
  util.return %0 : tensor<2x3x?x5xf32, #iree_tensor_ext.ragged_shape<1>>
}
