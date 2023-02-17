// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def ApplyPatternsOp : Op<Transform_Dialect, "iree.apply_patterns",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Greedily applies patterns as specified by its attributes.

    Must be applied to an op with trait IsolatedFromAbove since the
    GreedyPatternRewriter asserts those. Internally, uses the tracking rewriter
    to preserve handles to payload operations nested within operations
    associated with `target`. Fails if tracking cannot find replacement for a
    payload operation. This may become controllable with an attribute in the
    future.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    The following additive attributes can be set, they add patterns in an
    unspecified order:
      - additional_iree_patterns: fancy patterns we shortcut into the system,
      will need to be sliced out better in the future.
      - bubble_collapse_expand: bubble `expand_shape` up and `collapse_shape`
      down across Linalg ops.
      - canonicalization: adds all the canonicalization patterns of all
      registered dialects and ops.
      - erase_unnecessary_tensor_operands: add patterns that erase unnecessary
      tensor operands.
      - expand_memref_strided_metadata: adds patterns that expand memref
      operations into extract_strided_metadata operations and a materialization
      of their effect on the metadata (sizes, offset, strides).
      - fold_memref_aliases: adds patterns for folding ops such as
      memref.subview.
      - fold_reassociative_reshapes: adds patterns that fold insert_slice/
      extract_slice ops with reassociative reshape ops.
      - fold_tensor_empty_extract: Fold tensor.empty used by extract_slice in
      case it is the only use of extract.
      - lower_transfer_op_permutations: Lower transfer ops to transfer ops
      with minor identity permutations.
      - rank_reducing_linalg: adds patterns that results in rank-reducing
      behavior on subset-based linalg operations.
      - rank_reducing_vector: adds patterns that results in rank-reducing 
      behavior on subset-based vector operations.
      - rewrite_pack_ops: rewrite tensor.pack/unpack to linalg_ext.pack/unpack.
      This is a temporary pattern that is needed to connect to IREE until it 
      adopts the upstream version.
      - swapping_patterns: adds patterns that swap operations for a better outcome.
      This is a catch all that can be refined further if/when needed.
      - swap_padding_elide_conditional: refines the tensor.pad +
      tensor.extract_slice swapping pattern. This injects static information
      that guarantees padding is smaller than the window size which guarantees
      we never see a tile comprised of padding-only.
      - unroll_vectors_gpu_mma_sync: adds patterns that unroll vectors to a native tile
      size for GPUs with mma operations. The size is currently hardcoded but 
      should be refactored upstream and made pluggable.
      - unroll_vectors_gpu_wmma: adds patterns that unroll vectors to a native tile
      size for GPUs with wmma operations. The size is currently hardcoded but 
      should be refactored upstream and made pluggable.


    #### Return modes:

    This operation applies a set of patterns specified by attributes. To apply
    these patterns, this operation must target an operation that is isolated
    from above, otherwise the transform definitely fails.

    If the pattern application fails, or if the underlying listener fails to
    capture op handles, the transformation definitely fails.

    Otherwise the transformation is successful and returns the handle to the
    same payload as its operand to allow for simpler composition.
  }];

  let arguments = (ins PDL_Operation:$target,
                       UnitAttr:$additional_iree_patterns,
                       UnitAttr:$bubble_collapse_expand,
                       UnitAttr:$canonicalization,
                       UnitAttr:$erase_unnecessary_tensor_operands,
                       UnitAttr:$expand_memref_strided_metadata,
                       UnitAttr:$fold_memref_aliases,
                       UnitAttr:$fold_reassociative_reshapes,
                       UnitAttr:$fold_tensor_empty_extract,
                       UnitAttr:$lower_transfer_op_permutations,
                       UnitAttr:$rank_reducing_linalg,
                       UnitAttr:$rank_reducing_vector,
                       UnitAttr:$rewrite_pack_ops,
                       UnitAttr:$swap_padding_elide_conditional,
                       UnitAttr:$swapping_patterns,
                       UnitAttr:$unroll_vectors_gpu_mma_sync,
                       UnitAttr:$unroll_vectors_gpu_wmma);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    // TODO: Some bitvector to scale better than n-bools.
    OpBuilder<(ins "Value":$target,
                   "const ApplyPatternsOpPatterns &":$patterns)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
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

    #### Return modes

    This operation calls the upstream one-shot bufferization pass with extra
    registered patterns for IREE.

    The pass is ran on all the ModuleOp nested under the top-level op on which
    the transform dialect interpreter pass is applied.

    If any of the pass on any of the ModuleOp fails, the transformation
    definitely fails. Otherwise the transformation succeeds.

    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (
      ins PDL_Operation:$target,
          UnitAttr:$target_gpu,
          UnitAttr:$test_analysis_only,
          UnitAttr:$print_conflicts
  );
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target,
                   CArg<"bool", "false">:$targetGpu,
                   CArg<"bool", "false">:$testAnalysisOnly,
                   CArg<"bool", "false">:$printConflicts)>
  ];
}

def IREEEliminateEmptyTensorsOp : Op<
    Transform_Dialect, "iree.eliminate_empty_tensors",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    This is a pre-processing pass for iree.bufferize. It tries to remove
    tensor.empty ops by replacing them with a suitable destination tensors,
    which can reduce the number of allocations when bufferizing.

    This transform is not part of iree.bufferize because additional
    canonicalization are sometimes possible after eliminate_empty_tensors but
    before iree.bufferize.

    #### Return modes

    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
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

def ForallToWorkgroupOp : Op<Transform_Dialect,
    "iree.forall_to_workgroup",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Target the whole hal.executable_variant op and rewrite the unique topLevel
    scf.forall to distributed workgroup_id and workgroup_count.

    The mapping of threads to workgroup_id is currently one-to-one and in order.
    Only **bufferized** scf.forall are currently supported.
    Only scf.forall distributed to **at most 3 dimensions** are currently
    supported.

    Return modes:
    =============
    This operation ignores non-Func ops and drops them in the return.

    If no unique scf.forall topLevel operation is found, then the
    transform definitely fails.
    If the unique topLevel scf.forall has results (i.e. tensors), then
    the transform definitely fails.

    If the unique topLevel scf.forall maps to a dynamic number of
    threads, then the transform definitely fails. This is a temporary
    limitation until the backward slice computing scf.forall.num_threads
    can be extracted into the hal::executable_export workgroup_count region.
    This region may require arbitrary computations and cannot magically match
    what the `stream.cmd.dispatch` has already imposed on us at a distance.
    For now we must specify the number of values properly when applying the
    topLevel tile_to_forall_op.

    If the unique topLevel scf.forall operation contained within the
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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ShareForallOperandsOp : Op<
    Transform_Dialect, "iree.share_forall_operands", [
      FunctionalStyleTransformOpTrait,
      MemoryEffectsOpInterface,
      TransformEachOpTrait,
      TransformOpInterface]> {
  let description = [{
    Target a single scf.forall op and shares all uses of the specified 
    `share_operands` operand indices.

    Sharing can be thought of as the inverse of traditional privatization.
    Privatization consists in determining that a part of memory is only accessed
    by a single thread to and subsequently slicing out that part into a 
    thread_private storage that has smaller footprint, better locality and better
    alignment properties.
    In the case of scf.forall on tensors, tensor values are immutable 
    and the same tensor value may be passed as `shared_outs` and also captured
    for internal uses.
    Due to the immutability property, the whole tensor values are private by 
    construction and result in alloc + copy of the whole tensor on every thread
    to maintain the original SSA value after bufferizing. 

    An analysis similar to privatization is needed to ensure that only a private
    slice is needed and that the whole tensor can be shared.
    This transformation amounts to injecting the result of such an analysis as 
    static information in the program.
    The transformation checks that the values captured are `tensor.extract_slice`
    with a matching `tensor.parallel_insert_slice`, to approximate the lack of
    a cross-thread dependence analysis. 
    However this can still be unsafe wrt parallelism so use carefully!

    Sharing consists in rewriting all uses of the operands passed as 
    `shared_outs` that are also captured wihtin the `scf.forall` region
    into the matching `shared_outs` bbarg.

    Only those operands whose indices are specified in `share_operands` are
    shared. An empty `share_operands` specification considers all operands to
    be shared.

    #### Return modes

    If any of the `share_operands` indices overflow, a definite error is produced.
    
    If a `share_operands` fails a sharing precondition, it is ignored.
    In the future, we should emit a notification.

    This transform consumes the target handle and produces a result handle to
    the modified `scf.forall` op.
  }];

  let arguments = (
      ins Transform_ConcreteOpType<"scf.forall">:$forall_op,
          DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$share_operands
  );
  let results = (outs Transform_ConcreteOpType<"scf.forall">:$result);

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let assemblyFormat = [{
    $forall_op (`share_operands` `=` $share_operands^ )? attr-dict
      `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForallOp forallOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TileToForallAndWorkgroupCountRegionOp :
    Op<Transform_Dialect, "iree.tile_to_forall_and_workgroup_count_region",
      [AttrSizedOperandSegments,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let description = [{
    Wrapper around `structured.tile_to_forall_op` for use within IREE.

    In addition to tile and distribute using `scf.forall`, lowers the
    the `workgroup_count` region of the export op corresponding to the parent
    `func.func` of the target to return the number of workgroups.
    Please see the doc of `structured.tile_to_forall_op` for full
    description of op semantics.
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$num_threads,
                   Variadic<PDL_Operation>:$tile_sizes,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_num_threads,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs PDL_Operation:$forall_op,
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
               CArg<"Optional<int64_t>", "std::nullopt">:$level)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def ApplyBufferOptimizationsOp :
  Op<Transform_Dialect, "iree.apply_buffer_optimizations",
    [TransformEachOpTrait,
     TransformOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    This applies memory optimization on memref. In particular it does store to
    load forwarding, dead store elimination and dead alloc elimination.

    #### Return modes

    This operation applies a set of memory optimization on the whole region of
    the operand.

    If the transformation is successful it returns the handle to the
    same payload as its operand to allow for simpler composition.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
