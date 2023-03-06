// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STRUCTURED_TRANSFORM_OPS_EXT
#define STRUCTURED_TRANSFORM_OPS_EXT

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def LowerToLLVMOp : Op<Transform_Dialect, "lower_to_llvm",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Indicates that the entire module should be converted
  to the LLVM dialect. This is expected to be the last transformation in
  a sequence.}];

  let arguments =
    (ins DefaultValuedAttr<BoolAttr, "false">:$reassociate_fp_reductions,
     DefaultValuedAttr<BoolAttr, "false">:$enable_index_optimizations,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_neon,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_sve,
     DefaultValuedAttr<BoolAttr, "false">:$enable_amx,
     DefaultValuedAttr<BoolAttr, "false">:$enable_x86vector,
     DefaultValuedAttr<BoolAttr, "false">:$enable_async);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::transform_ext";
}


def RegisterMatchCallbacksOp :
    Op<Transform_Dialect, "iree.register_match_callbacks",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Registers named structured op matcher callbacks specific for IREE to use
    with `transform.iree.match_callback`. This should be called before first
    `match_callback` may be executed following the transform dialect control
    flow.

    The callbacks must have a unique name and a signature compatible with
    `MatchCallbacksRegistry::MatchCallbackFn`, which currently means
    `DiagnosedSilenceableFailure(MatchCallbackResult &, Location,
     const TransformState &, ValueRange)`. The callback receives a "result",
     followed by a location at which errors should be reported, a transform
     state at the moment of the _match_ (not registration) and a list of
     handle values passed as operands to the `match_callback` operation.
     It is expected to populate the "result" object with lists of payload
     operations that will be bound to the handles produced by the
     `match_callback` operation. The callback may fail, at which point
     it should produce a silenceable error. The callback currently is not
     allowed to modify the payload IR (though this may be revised in the
     future for the purpose of communicating the properties of the IR
     captured by the match). Therefore, it should not have a reason to
     produce a definite error.
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::transform_ext";
}

def MatchCallbackOp :
    Op<Transform_Dialect, "iree.match_callback",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Performs payload IR matching using a C++ callback registered beforehand.
    The callback is identified by name and is passed the current transform
    state and the list of handle operands, along with information necessary
    for error propagation. See `register_match_callbacks` for the description
    of the callback contract.

    If `failure_propagation_mode` is set to `suppress`, any silenceable errors
    in the callback (typically, "failure to match") will be ignored and the
    resulting handles will be associated with empty lists of payload
    operations. Otherwise, silenceable failures are propagated.
  }];

  let arguments = (ins StrAttr:$callback_name,
                       FailurePropagationMode:$failure_propagation_mode,
                       Variadic<TransformHandleTypeInterface>:$inputs);
  let results = (outs Variadic<TransformHandleTypeInterface>:$outputs);
  let assemblyFormat = "`failures` `(` $failure_propagation_mode `)` "
                       "$callback_name `(` $inputs `)` attr-dict "
                       "`:` functional-type($inputs, $outputs)";
  let cppNamespace = "mlir::transform_ext";
}

def TakeFirstOp :
    Op<Transform_Dialect, "iree.take_first",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given an arbitrary list of handles associated with potentially empty lists
    of payload operations, produces two new handles:

      - a handle pointing to the same payload operations as the first operand
        handle with a non-empty list of payload operations;
      - a handle pointing to the concatenated list of payload operations
        associated with any other handle.

    Note that this does not perform any deduplication.

    This operation is useful to select a single target after some potentially
    unsuccessful matches.
  }];

  let arguments = (ins Variadic<TransformHandleTypeInterface>:$inputs);
  let results = (outs TransformHandleTypeInterface:$first,
                      TransformHandleTypeInterface:$rest);
  let assemblyFormat =
      "$inputs attr-dict `:` functional-type($inputs, results)";
  let cppNamespace = "mlir::transform_ext";
}

def EmitRemarkOp :
    Op<Transform_Dialect, "iree.emit_remark",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Emits a diagnostic remark with the given message located at payload ops
    associated with the given handle. This can be used, e.g., for debugging.
  }];

  let arguments = (ins TransformHandleTypeInterface:$handle,
                       StrAttr:$message);
  let assemblyFormat = "$message `at` $handle attr-dict `:` type($handle)";
  let cppNamespace = "mlir::transform_ext";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

#endif // STRUCTURED_TRANSFORM_OPS_EXT
