// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def ApplyPatternsOp : Op<Transform_Dialect, "iree.apply_patterns",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Greedily applies patterns as specified by its attributes.

    Must be applied to an op with trait IsolatedFromAbove since the
    GreedyPatternRewriter asserts those.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    The following additive attributes can be set, they add patterns in an
    unspecified order:
      - additional_iree_patterns: fancy patterns we shortcut into the system,
      will need to be sliced out better in the future.
      - canonicalization: adds all the canonicalization patterns of all
      registered dialects and ops.
      - promote_foreach_thread_capture_to_shared: adds patterns that rewrite
      uses of values captured by scf.foreach_thread with the matching
      shared_outs bbarg. This checks that the values captured are
      tensor.extract_slice with a matching tensor.parallel_insert_slice to
      approximate the lack of cross-thread dependences. However this can still
      be unsafe wrt parallelism so use carefully!
      - rank_reducing: adds patterns that results in rank-reducing behavior on
      subset-based operations.
      - expand_memref_strided_metadata: adds patterns that expand memref
      operations into extract_strided_metadata operations and a materialization
      of their effect on the metadata (sizes, offset, strides).
      - swapping_patterns: adds patterns that swap operations for a better outcome.
      This is a catch all that can be refined further if/when needed.
      - swap_padding_elide_conditional: refines the tensor.pad +
      tensor.extract_slice swapping pattern. This injects static information
      that guarantees padding is smaller than the window size which guarantees
      we never see a tile comprised of padding-only.
      This allows dropping the generation or an annoying internal scf.if but may
      yield incorrect code in pathological cases.

    Return modes:
    =============
    This operation applies a number of patterns to rewrite vector IR into
    distributed warp form. To apply these patterns, this operation must target
    an operation that is isolated from above, otherwise the transform definitely
    fails.

    If the pattern application fails, or if the underlying listener fails to
    capture op handles, the transformation definitely fails.

    Otherwise the transformation is successful and no result is returned.
  }];

  let arguments = (ins PDL_Operation:$target,
                       UnitAttr:$additional_iree_patterns,
                       UnitAttr:$canonicalization,
                       UnitAttr:$promote_foreach_thread_capture_to_shared,
                       UnitAttr:$rank_reducing,
                       UnitAttr:$expand_memref_strided_metadata,
                       UnitAttr:$swap_padding_elide_conditional,
                       UnitAttr:$swapping_patterns);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    // TODO: Some bitvector to scale better than n-bools.
    OpBuilder<(ins "Value":$target, "bool":$rankReducing)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def IREEBufferizeOp : Op<Transform_Dialect, "iree.bufferize",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Target the whole hal.executable_variant op and call upstream comprehensive
    bufferize with extra IREE hooks.

    By default, CPU allocations are emitted. This behavior can be modified by
    using the following attributes:
      - target_gpu: if set, GPU allocations are emitted.

    Return modes:
    =============
    This operation calls the upstream one-shot bufferization pass with extra
    registered patterns for IREE.

    The pass is ran on all the ModuleOp nested under the top-level op on which
    the transform dialect interpreter pass is applied.

    If any of the pass on any of the ModuleOp fails, the transformation
    definitely fails. Otherwise the transformation succeeds.

    No handles are consumed or produced.
  }];

  let arguments = (
      ins PDL_Operation:$target,
          UnitAttr:$target_gpu,
          DefaultValuedAttr<BoolAttr, "false">:$test_analysis_only,
          DefaultValuedAttr<BoolAttr, "false">:$print_conflicts
  );
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, CArg<"bool", "false">:$targetGpu)>
  ];
}

def IREEEraseHALDescriptorTypeFromMemRefOp : Op<Transform_Dialect,
    "iree.erase_hal_descriptor_type_from_memref",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Erase #hal.descriptor_type from MemRef memory space to ignore all IREE
    memory space planning. This is meant to ease transitioning given that
    various LLVM conversion upstream patterns assumes numeric memory space,
    especially the default 0.

    Return modes:
    =============
    The pass is ran on all FuncOp nested under the top-level op on which
    the transform dialect interpreter pass is applied.

    If any of the pass on any of the FuncOp fails, the transformation
    definitely fails. Otherwise the transformation succeeds.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [OpBuilder<(ins "Value":$target)>];
}

def ForeachThreadToWorkgroupOp : Op<Transform_Dialect,
    "iree.foreach_thread_to_workgroup",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Target the whole hal.executable_variant op and rewrite the unique topLevel
    scf.foreach_thread to distributed workgroup_id and workgroup_count.

    The mapping of threads to workgroup_id is currently one-to-one and in order.
    Only **bufferized** scf.foreach_thread are currently supported.
    Only scf.foreach_thread distributed to **at most 3 dimensions** are currently
    supported.

    Return modes:
    =============
    This operation ignores non-Func ops and drops them in the return.

    If no unique scf.foreach_thread topLevel operation is found, then the
    transform definitely fails.
    If the unique topLevel scf.foreach_thread has results (i.e. tensors), then
    the transform definitely fails.

    If the unique topLevel scf.foreach_thread maps to a dynamic number of
    threads, then the transform definitely fails. This is a temporary
    limitation until the backward slice computing scf.foreach_thread.num_threads
    can be extracted into the hal::executable_export workgroup_count region.
    This region may require arbitrary computations and cannot magically match
    what the `stream.cmd.dispatch` has already imposed on us at a distance.
    For now we must specify the number of values properly when applying the
    topLevel tile_to_foreach_thread_op.

    If the unique topLevel scf.foreach_thread operation contained within the
    FuncOp referred to by the `target` PDLOperation lowers to workgroup properly,
    the transform succeeds. Otherwise the transform definitely fails.

    The returned handle points to the same FuncOp operand, consuming it and
    producing a new SSA value to satisfy chaining and linearity of the IR
    properties.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::func::FuncOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TileToForeachThreadAndWorkgroupCountRegionOp :
    Op<Transform_Dialect, "iree.tile_to_foreach_thread_and_workgroup_count_region",
      [AttrSizedOperandSegments,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let description = [{
    Wrapper around `structured.tile_to_foreach_thread_op` for use within IREE.

    In addition to tile and distribute using `scf.foreach_thread`, lowers the
    the `workgroup_count` region of the export op corresponding to the parent
    `func.func` of the target to return the number of workgroups.
    Please see the doc of `structured.tile_to_foreach_thread_op` for full
    description of op semantics.
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$num_threads,
                   Variadic<PDL_Operation>:$tile_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_num_threads,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs PDL_Operation:$foreach_thread_op,
                      PDL_Operation:$tiled_op);
  let assemblyFormat = [{
    $target oilist(
        `num_threads` custom<DynamicIndexList>($num_threads,
                                               $static_num_threads) |
         `tile_sizes` custom<DynamicIndexList>($tile_sizes,
                                               $static_tile_sizes))
    (`(` `mapping` `=` $mapping^ `)`)? attr-dict
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"::mlir::transform::TileSizesSpec",
                        "::mlir::transform::TileSizesSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"::mlir::transform::TileSizesSpec",
                        "::mlir::transform::TileSizesSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticNumThreads,
                   CArg<"::mlir::transform::NumThreadsSpec",
                        "::mlir::transform::NumThreadsSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedNumThreads,
                   CArg<"::mlir::transform::NumThreadsSpec",
                        "::mlir::transform::NumThreadsSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedNumThreads();
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTileSizes();
  }];
}

def ConfigExtractPart :
    Op<Transform_Dialect, "iree.config.extract_part",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let description = [{
  }];
  let arguments = (ins PDL_Operation:$target,
                       StrAttr:$attr_name,
                       OptionalAttr<I64Attr>:$level);
  // TODO: allow return attributes ?
  let results = (outs PDL_Operation:$resultConfigPart);
  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  // TODO: impl me.
  // let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$target, "StringRef":$attrName,
               CArg<"Optional<int64_t>", "llvm::None">:$level)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
