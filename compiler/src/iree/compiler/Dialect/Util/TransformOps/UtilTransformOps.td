// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORM_OPS
#define IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpBase.td"

def GetNearestSymbolTableOp : Op<Transform_Dialect, "util.get_nearest_symbol_table", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface,
  TransformEachOpTrait,
  ReportTrackingListenerFailuresOpTrait,
]> {
  let cppNamespace = "mlir::iree_compiler::IREE::Util::transform_dialect";

  let description = [{
    Returns the nearest symbol table op for each op in the payload, inclusive.

    This operation reads the `target` handle and produces the `result`
    handle. This operation emits a definite failure if the nearest symbol table
    is unknown.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target
  );
  let results = (outs
    TransformHandleTypeInterface:$result
  );

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation* target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ImportSymbolOp : Op<Transform_Dialect, "util.import_symbol", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  ReportTrackingListenerFailuresOpTrait,
]> {
  let cppNamespace = "mlir::iree_compiler::IREE::Util::transform_dialect";

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

  let arguments = (ins
    SymbolRefAttr:$symbol,
    UnitAttr:$if_undefined,
    UnitAttr:$force_import,
    TransformHandleTypeInterface:$symbol_table
  );
  let results = (outs
    TransformHandleTypeInterface:$cloned_symbol
  );

  let assemblyFormat = [{
    (`force` $force_import^)? $symbol `into` $symbol_table
    (`if` `undefined` $if_undefined^)? attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def CastAndCallOp : Op<Transform_Dialect, "util.cast_and_call", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AttrSizedOperandSegments,
  ReportTrackingListenerFailuresOpTrait,
] # GraphRegionNoTerminator.traits> {
  let cppNamespace = "mlir::iree_compiler::IREE::Util::transform_dialect";

  let summary = "Casts values to the signature of a function and replaces them "
                "with a call";
  let description = [{
    This transform takes value handles to a set of `inputs` and `outputs` and
    attempts to cast them to the function signature of the attached function
    op, then builds a call to the function and replaces the users of the
    outputs. It is the responsibility of the user to ensure that the slice of
    the program replaced by this operation makes sense, i.e. there is no
    verification that the inputs to this operation have any relation to the
    outputs outside of basic dominance requirements needed for the call.

    The casting materialization functions are specified in the graph region of
    this op. They must implement the `TypeConverterBuilderOpInterface`. The
    order of ops within the region is irrelevant.

    The target function can be specified by a symbol name or by a handle to the
    operation.

    This transform only reads the operand handles and only replaces the users of
    the outputs with the results of the call. No handles are consumed and no
    operations are removed. Users are expected to run cleanup separately if
    desired.

    Warning: The replacement of the uses of the outputs could invalidate certain
    restricted value handle types (e.g. `transform.block_arg` if it existed, by
    replacing the use with something not coming from a block argument). The
    value will still exist in such cases but wouldn't verify against the type.
    See the discussion here for more information:
    https://github.com/llvm/llvm-project/pull/78398#discussion_r1455070087

    This transform will emit a silenceable failure if:
     - The set of outputs isn't unique
     - The handle for the insertion point does not include exactly one operation
     - The insertion point op does not dominate any of the output users
     - The insertion point op is not dominated by any of the inputs
     - The function signature does not match the number of inputs/outputs

    This transform will emit a definite failure if it fails to resolve the
    target function, or if it fails to materialize the conversion casts of
    either the inputs to the function argument types, or the call results to
    the output types.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$insertion_point,
    UnitAttr:$insert_after,
    Optional<TransformValueHandleTypeInterface>:$inputs,
    Optional<TransformValueHandleTypeInterface>:$outputs,
    OptionalAttr<SymbolRefAttr>:$function_name,
    Optional<TransformHandleTypeInterface>:$function
  );
  let results = (outs
    TransformHandleTypeInterface:$result
  );
  let regions = (region
    MaxSizedRegion<1>:$conversions
  );

  let assemblyFormat = [{
    ($function_name^)? ($function^)?
    ( `(` $inputs^ `)` )?
    ( `->` $outputs^ )?
    (`after` $insert_after^):(`before`)? $insertion_point
    ($conversions^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

#endif // IREE_COMPILER_DIALECT_UTIL_TRANSFORMOPS_UTILTRANSFORM_OPS
