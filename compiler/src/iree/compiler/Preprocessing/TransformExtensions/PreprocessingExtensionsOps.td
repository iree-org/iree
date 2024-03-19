// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS
#define IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS

include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MatchCastCompatibleDagFromRootOp : Op<Transform_Dialect, "iree.match.cast_compatible_dag_from_root",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleOpMatcher,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
     MemoryEffectsOpInterface]> {
  let summary =
      "Checks if the body of the target op matches an operation dag starting "
      "at the given root";
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
  let summary =
      "Checks if the body of the target op matches the body of the single contained op";
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

def MatchDimIsMultipleOfOp : Op<Transform_Dialect, "iree.match.dim_is_multiple_of",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleValueMatcher,
     MemoryEffectsOpInterface]> {
  let summary =
      "Checks if the body of the target op matches the body of the single contained op";
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
  let summary =
      "Checks if the body of the target op matches the body of the single contained op";
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

#endif // IREE_COMPILER_DIALECT_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS
