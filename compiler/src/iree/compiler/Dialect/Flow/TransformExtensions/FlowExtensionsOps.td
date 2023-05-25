// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.forall_to_flow",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Rewrite an scf.forall to Flow::DispatchWorkgroups.

    #### Return modes

    This operation ignores non-scf::ForeachThread ops and drops them in the
    return.

    If any rewrite fails, the transform definitely fails.

    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully produced
    equivalent flow::DispatchWorkgroups operations, which can be empty.

    This operation consumes the `target` handle and produces the `transformed`
    handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def RegionToWorkgroupsOp : Op<Transform_Dialect, "iree.region_to_workgroups",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Convert a flow.dispatch.region op into a flow.dispatch.workgroups op.

    Region ops can capture any values that are in scope in their body, whereas
    workgroup ops are isolated from above and all values must be explicitly
    captured via operands.

    #### Return modes

    This operation consumes the `target` handle and produces the `transformed`
    handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::iree_compiler::IREE::Flow::DispatchRegionOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def WrapInDispatchRegionOp : Op<
    Transform_Dialect, "iree.wrap_in_dispatch_region",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Wrap the `target` op in a new `dispatch.region` op. All uses of target op
    are replaces with the results of the newly generated `dispach.region` op.

    The argument `generateWorkload` can be set to generate workload information
    from the given `target` op.

    #### Return modes

    This transform reads the `target` handle and produces the `transformed`
    handle (i.e., the `dispatch.region` op).
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DefaultValuedAttr<BoolAttr, "true">:$generateWorkload);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ClonePrecedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.clone_preceding_op_into_dispatch_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformOpInterface]> {
  let description = [{
    Clone the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    All uses of the target inside of the dispatch region are replaced with the
    results of the cloned op. Uses of the target outside of the dispatch region
    remain unchanged.

    #### Return modes

    This transform reads the `target` handle and the `dispatch_region`
    handle. It produces a new handle to the cloned ops.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$dispatch_region);
  let results = (outs TransformHandleTypeInterface:$cloned);
  let assemblyFormat = [{
    $target `into` $dispatch_region attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def MovePrecedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.move_preceding_op_into_dispatch_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformOpInterface]> {
  let description = [{
    Move the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    An extra result is added to the dispatch region for every result of the
    target op. All uses of the target op are replaced with the newly added
    results of the dispatch region.

    Note: This transform generates invalid IR if there are uses of the target op
    that appear before (i.e., dominate) the dispatch region.

    #### Return modes

    This transform reads the `target` handle and consumes the `dispatch_region`
    handle. It produces a new handle to the extended dispatch region.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$dispatch_region);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = [{
    $target `into` $dispatch_region attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def CloneSucceedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.clone_succeeding_op_into_dispatch_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformOpInterface]> {
  let description = [{
    Clone the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    All operands of the target are replaced with values that are defined inside
    of the dispatch region when possible. No new results are added to the
    dispatch region.

    #### Return modes

    `dispatch_region` must be mapped to exactly one op. `target` may be mapped
    to multiple ops.

    This transform reads both the `target` handle and the `dispatch_region`
    handle. It produces a new handle to the cloned ops.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$dispatch_region);
  let results = (outs TransformHandleTypeInterface:$cloned);
  let assemblyFormat = [{
    $target `into` $dispatch_region attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def MoveSucceedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.move_succeeding_op_into_dispatch_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformOpInterface]> {
  let description = [{
    Move the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    All operands of the target are replaced with values that are defined inside
    of the dispatch region when possible. All uses of the target op are
    replaced with a newly ammended results of the dispatch region op.

    #### Return modes

    `dispatch_region` must be mapped to exactly one op. `target` may be mapped
    to multiple ops.

    This transform reads the `target` handle and consumes the `dispatch_region`
    handle. It produces a new handle to the new dispatch region.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$dispatch_region);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = [{
    $target `into` $dispatch_region attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
