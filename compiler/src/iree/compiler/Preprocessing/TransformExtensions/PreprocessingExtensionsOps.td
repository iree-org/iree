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

def GetNearestSymbolTableOp : Op<Transform_Dialect, "iree.get_nearest_symbol_table",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Returns the nearest symbol table op for each op in the payload, inclusive.

    #### Return modes

    This operation reads the `target` handle and produces the `result`
    handle. This operation emits a definite failure if the nearest symbol table
    is unknown.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ImportSymbolOp : Op<Transform_Dialect, "iree.import_symbol",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Clones the op defined by the given symbol into the given symbol table and
    returns the cloned symbol. If `force_import` is set, this will (unsafely)
    overwrite any pre-existing definitions of the same symbol. If
    `if_undefined` is set, this will return a handle to the pre-existing symbol
    in the payload if found instead of failing.

    #### Return modes

    This operation reads the `symbol_table` handle and produces the
    `cloned_symbol` handle. This operation emits a definite failure if the if
    the `symbol_table` op does not define a symbol table.

    This will emit a definite failure if the symbol already exists in the
    symbol table and neither `force_import` and `if_undefined` are set.
  }];

  let arguments = (ins SymbolRefAttr:$symbol,
                       UnitAttr:$if_undefined,
                       UnitAttr:$force_import,
                       TransformHandleTypeInterface:$symbol_table);
  let results = (outs TransformHandleTypeInterface:$cloned_symbol);

  let assemblyFormat = [{
    (`force` $force_import^)? $symbol `into` $symbol_table
    (`if` `undefined` $if_undefined^)? attr-dict 
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let hasVerifier = 1;
}

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
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
     MemoryEffectsOpInterface]> {
  let summary =
      "Checks if the body of the target op matches the body of the single contained op";
  let description = [{
    Checks whether the given value is cast-compatible with the given target
    type attribute. If the target type attribute is a tensor type, an additional
    encoding can be specified to indicate whether the given static size is an
    alignment requirement.

    Currently this operation only supports tensor types.

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
