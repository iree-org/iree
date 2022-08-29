// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.foreach_thread_to_flow",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Rewrite an scf.foreach_thread to Flow::DispatchWorkgroups.


    Return modes:
    =============
    This operation ignores non-scf::ForeachThread ops and drops them in the
    return.

    If any rewrite fails, the transform definitely fails.

    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully produced
    equivalent flow::DispatchWorkgroups operations, which can be empty.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForeachThreadOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def RegionToWorkgroupsOp : Op<Transform_Dialect, "iree.region_to_workgroups",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Convert a flow.dispatch.region op into a flow.dispatch.workgroups op.

    Region ops can capture any values that are in scope in their body, whereas
    workgroup ops are isolated from above and all values must be explicitly
    captured via operands.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::iree_compiler::IREE::Flow::DispatchRegionOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def WrapInDispatchRegionOp : Op<
    Transform_Dialect, "iree.wrap_in_dispatch_region",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Wrap the `target` op in a new `dispatch.region` op. All uses of target op
    are replaces with the results of the newly generated `dispach.region` op.

    #### Return modes

    This transform consumes the `target` handle and produces the `transformed`
    handle (i.e., the `dispatch.region` op).
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ClonePrecedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.clone_preceding_op_into_dispatch_region",
    [TransformOpInterface]> {
  let description = [{
    Clone the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    All uses of the target inside of the dispatch region are replaced with the
    results of the cloned op.

    If `update_uses_outside_of_region` is set (default value: `false`), all
    uses outside of the dispatch region are also replaced: The results of the
    cloned target op are yielded from the dispatch region and used in all uses
    outside of the dispatch region. The transform fails if there are uses that
    appear before the dispatch region.

    #### Return modes

    This transform consumes both the `target` handle and the `dispatch_region`
    handle. It produces a new handle to the extended dispatch region.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$target,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$dispatch_region,
                       DefaultValuedAttr<BoolAttr, "false">:$update_uses_outside_of_region);
  let results = (outs Res<PDL_Operation, "",
                           [TransformMappingAlloc,
                            TransformMappingWrite]>:$transformed);
  let assemblyFormat = "$target `into` $dispatch_region attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def CloneSucceedingOpIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.clone_succeeding_op_into_dispatch_region",
    [TransformOpInterface]> {
  let description = [{
    Clone the `target` op into the given dispatch region op. The dispatch region
    handle must be mapped to exactly one payload op.

    All operands of the target are replaced with values that are defined inside
    of the dispatch region when possible. 

    If `update_uses_outside_of_region` is set (default value: `true`), all uses
    of the original target op are replaced: The results of the cloned target op
    are yielded from the dispatch region and used instead of results of the
    original target op.

    #### Return modes

    This transform consumes both the `target` handle and the `dispatch_region`
    handle. It produces a new handle to the extended dispatch region.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$target,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$dispatch_region,
                       DefaultValuedAttr<BoolAttr, "true">:$update_uses_outside_of_region);
  let results = (outs Res<PDL_Operation, "",
                           [TransformMappingAlloc,
                            TransformMappingWrite]>:$transformed);
  let assemblyFormat = "$target `into` $dispatch_region attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
