// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MapNestedForallToGpuThreadsOp :
  Op<Transform_Dialect, "iree.map_nested_forall_to_gpu_threads",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Target the whole hal.executable_variant op and rewrite all scf.forall
    to distributed gpu.thread_id and translation_info attribute.

    This op will handle all the scf.forall using gpu.thread or gpu.warp
    mapping.

    The mapping of threads to gpu.thread_id is currently one-to-one and in order.
    Only **bufferized** scf.forall are currently supported.
    Only scf.forall distributed to **at most 3 dimensions** are currently
    supported.

    Multiple scf.forall are supported per function in which case, the
    max of all the threads is computed and taken for the global gpu.thread_id.
    If necessary, scf.forall that do not use the whole thread range
    result in predicated computations.

    Barriers are inserted after each scf.forall op for now.

    Return modes:
    =============
    This operation ignores non-Func ops and drops them in the return.

    If any scf.forall with tensors is found, the transform definitely
    fails.

    If all the scf.forall operations contained within the FuncOp
    referred to by the `target` operand lower to GPU properly, the
    transform succeeds. Otherwise the transform definitely fails.

    The returned handle points to the same FuncOp operand, consuming it and
    producing a new SSA value to satisfy chaining and linearity of the IR
    properties.

    Example:
    ========

    ```
    hal.executable {
      hal.executable.variant {
        hal.executable.export {
          func @foo() {
            scf.forall (%i, %j) in (7, 9) {
              ... // body 1
            }
            scf.forall (%i) in (12) {
              ... // body 2
            }
          }
    ```

    is translated to:

    ```
    hal.executable {
      hal.executable.variant {
        hal.executable.export ... workgroup_size = [12 : index, 9 : index, 1 : index] {
          func @foo() {
            if (threadIdx.x < 7) {
              ... // body 1
            }
            if (threadIdx.y < 1) {
              ... // body 2
            }
          }
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$workgroup_dims,
                   DefaultValuedOptionalAttr<I64Attr, "32">:$subgroup_size);
  let results = (outs);

  let assemblyFormat = [{
    $target
    `workgroup_dims` `=` $workgroup_dims
    (`subgroup_size` `=` $subgroup_size^)?
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def VectorToWarpExecuteOnLane0Op : Op<Transform_Dialect, "iree.vector.to_warp_execute_on_lane_0",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Given an scf.if target predicated by `if (threadIdx.x == 0)`, rewrite its
    body to vector.execute_on_lane_0 running ***on a single warp***.

    The warp size is determined by the `warp_size` attribute (it is generally
    32 but we do not want to hardcode it).

    This rewrite only applies if it can be determined from the IR (i.e. from
    the surrounding IREE::HAL::ExecutableExportOp) that the number of threads
    along the warp dimension is a multiple of the warp size. The transformation
    bails on non-perfect multiples of the warp size that would not properly
    distribute.

    This is the first of two step towards apply vector distribution to a single
    warp.


    Return modes:
    =============
    This operation ignores non-scf::IfOp ops and drops them in the return.

    If all the operations referred to by the `target` operand are properly
    properly, the transform succeeds. Otherwise the transform silently fails.

    If the transform is anchored at a top-level that is not isolated from above,
    the transform definitely fails.

    If the transform cannot find a proper HAL::ExecutableExportOp with a
    well-formed workgroup_size 3-entry attribute such that the threadIdx.x
    component is a multiple of warp_size, the transform silently fails.
    If the scf::ForOp predicate does not predicate on threadIdx.x == 0, the
    transform silently fails.

    Otherwise the transformation succeeds and the returned handle points to the
    produced vector::WarpExecuteOnThread0Op.


    Example:
    ========

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c1 : index
        scf.if %2 {
          %3 = arith.constant dense<1.0> : vector<128xf32>
          vector.transfer_write %3, %0[%c0] : vector<128xf32>, memref<128xf32>
        }
      }
    }
    ```

    rewrites to:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          vector.warp_execute_on_lane_0(%1)[32] {
            %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
            vector.transfer_write %cst, %0[%c0] : vector<128xf32>, memref<128xf32>
          }
        }
      }
    }
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedAttr<I64Attr, "{}">:$warp_size);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, $result)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$warpSize)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::IfOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def VectorWarpDistributionOp : Op<Transform_Dialect, "iree.vector.warp_distribute",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Given a vector.warp_execute_on_lane_0, apply the patterns to rewrite into
    distributed form with warp synchronization. This produces IR that runs
    ***on a single warp***.

    IR that cannot be distributed will be predicated by `if (threadIdx.x == 0)`.

    This is the second step of two for applying vector distribution to a single
    warp.


    Return modes:
    =============
    This operation applies a number of patterns to rewrite vector IR into
    distributed warp form. To apply these patterns, this operation must target
    an operation that is isolated from above, otherwise the transform definitely
    fails.

    Patterns sets are applied in the following order:
      - applyMultiReductionLoweringPatterns
      - applyVectorTransferWriteDistribution
      - applyPropagateVectorDistribution
      - applyWarpExecuteOnLane0ToScf

    If any of the pattern sets fail to apply, the transformation definitely
    fails.

    Otherwise the transformation is successful and no result is returned.


    Example:
    ========

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          vector.warp_execute_on_lane_0(%1)[32] {
            %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
            vector.transfer_write %cst, %0[%c0] : vector<128xf32>, memref<128xf32>
          }
      }
      }
    }
    ```

    distributes to:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          %3 = arith.cmpi eq, %1, %c0 : index
          %4 = memref.alloc() : memref<128xf32, 3>
          // Single-thread guard runs on thread 0 only.
          scf.if %3 {
            vector.store %cst, %4[%c0] : memref<128xf32, 3>, vector<128xf32>
          }
          %5 = arith.muli %1, %c4 : index
          %6 = vector.load %4[%5] : memref<128xf32, 3>, vector<4xf32>
          %7 = affine.apply #map()[%1]
          vector.transfer_write %6, %0[%7] {in_bounds = [true]} : vector<4xf32>, memref<128xf32>
        }
      }
    }
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target 
    attr-dict 
    `:` functional-type($target, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def VectorToMMAConversionOp : Op<Transform_Dialect, "iree.vector.vector_to_mma_conversion",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This converts slices of operations containing vector.contract op into
    mma operations, targetting warp level tensorcore operations. If the vector
    operations are bigger than the native mma size it will first split up those
    vector operations.

    Exactly one of use_wmma or use_mma_sync must be specified.

    #### Return modes

    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       UnitAttr:$use_mma_sync,
                       UnitAttr:$use_wmma);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def PromoteOperandsOp :
  Op<Transform_Dialect, "iree.promote_operands",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This op promotes the specified operands of the provided target handle.

    #### Return modes
    This op consume its target handle and returns a new handle to its target handle
    as well as an allocTensorOp for each of the provided valid indices.

    If the promotion of any specified operand fails to occur, the op definitely
    fails.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$indices);
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = [{ $target $indices attr-dict `:` functional-type(operands, results)}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def PipelineSharedMemoryCopiesOp : Op<
    Transform_Dialect, "iree.pipeline_shared_memory_copies", [
      FunctionalStyleTransformOpTrait,
      MemoryEffectsOpInterface,
      TransformEachOpTrait,
      TransformOpInterface,
      ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This applies software pipelining to a given scf.for loop. The pipelining
    strategy will look for a copy to shared memory and pipeline it to overlap
    it with the rest of the loop.
    It is user responsability to ensure that there are no dependency between
    `depth` iterations of the loop by using multi-buffering.

    `depth` will indicate how many stages the software pipeline should have.
    `peel_epilogue` allows to force the epilogue to be peeled out instead of
    potentially using predicated operations for the epilogue phase.

    #### Return modes
    This transform consumes the scf.for handle and produces a result handle
    which points to a) the new scf.for loop generated (success case) or b) the
    existing scf.for loop (failure case).
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$for_op,
          I64Attr:$depth,
          UnitAttr:$peel_epilogue,
          UnitAttr:$use_mma_sync);
  let results = (outs TransformHandleTypeInterface:$result);

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let assemblyFormat = [{ 
    $for_op 
    attr-dict 
    `:` functional-type(operands, results)}];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ForOp forOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def SynchronizeLoopOp : Op<
    Transform_Dialect, "iree.synchronize_loop", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformEachOpTrait,
      TransformOpInterface,
      ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    This inserts a gpu.barrier after a given scf.for loop.

    #### Return modes
    This transform consumes the scf.for handle and produces a result handle
    which points to the new scf.for loop generated. It will fail if the loop
    cannot be pipelined or if there are no shared memory copies.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$for_op);
  let results = (outs);

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let assemblyFormat = [{ 
    $for_op 
    attr-dict 
    `:` functional-type(operands, results)}];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ForOp forOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CreateAsyncGroupsOp :
  Op<Transform_Dialect, "iree.create_async_groups",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Convert copies to shared memory to async copies. This creates groups
    of consecutive copies and emit wait operation right after.
    The input operation is a `func.func`.

    `use_mma_sync` specifies whether `bypassL1` attributes should be added to the
    async copies.

    #### Return modes
    This op returns a handle to the transformed function, even if nothing
    changed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   UnitAttr:$use_mma_sync);
  let results = (outs);

  let assemblyFormat = [{ 
    $target 
    attr-dict 
    `:` functional-type(operands, results)}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LayoutAnalysisAndDistributionOp :
  Op<Transform_Dialect, "iree.layout_analysis_and_distribution",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Targets the whole func op and does the following:

    1. First, computes the layouts of the of any vector.contract ops
       that could lower to mma.syncs. Then, propagates these layouts to
       neighboring ops.

    2. Next, it lowers each of these ops based on their layout.

    The layout is represented as an 8D vector with the following shape:
    batch0, batch1, laneidz, laneidy, laneidx, vecidz, vecidy, vecidx

  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = [{ $target attr-dict `:` functional-type(operands, results)}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

}

def ReorderTransposeOp :
  Op<Transform_Dialect, "iree.reorder_transpose",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Targets the whole func op and finds transpose ops whose source
    comes from an elementwise op. For each of those transpose ops,
    it moves the transpose before the elementwise op by first
    transposing the operands of the elementwise op and then redoing
    the elementwise op using the transposed operands. It then
    replaces all uses of the original transpose op with the result
    of the new elementwise op.

    More specifically, given IR that looks like below,
      %0 = arith.subf %a, %b : vector<16x8xf16>
      %1 = vector.transpose %0 [1, 0] : vector<16x8xf16> to vector<8x16xf16>

    It moves the transpose before the elementwise op to produce
      %transposed_a = vector.transpose %a [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %transposed_b = vector.transpose %b [1, 0] : vector<16x8xf16> to vector<8x16xf16>
      %0 = arith.subf %transposed_a, %transposed_b : vector<8x16xf16>
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = [{ $target attr-dict `:` functional-type(operands, results)}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

}

def EliminateGpuBarriersOp :
  Op<Transform_Dialect, "iree.eliminate_gpu_barriers",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Removes unnecessary GPU barriers from the function. If a barrier does not
    enforce any conflicting pair of memory effects, including a pair that is
    enforced by another barrier, it is unnecessary and can be removed.

    #### Return modes

    Consumes the operand handle and produces a new handle to the function after
    rewriting.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{ $target attr-dict `:` functional-type(operands, results)}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def PackSharedMemoryAllocOp :  Op<Transform_Dialect, "iree.pack_shared_memory_alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Pack shared memory allocation to reduce memory usage";
  let description = [{
    Looks for allocs in shared memory space with overlapping liveness and
    groups them, then packs all the allocations in each group into one i8
    alloc. Also adds barriers to make sure we are done writing/reading
    from the previous alias group before starting a new one.

    #### Return modes

    It does not consume the target handle and always return success.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$target
  );
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def OptimizeSharedMemoryReadsAndWritesOp :
  Op<Transform_Dialect, "iree.optimize_shared_memory_reads_and_writes",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Reduce shared memory bank conflicts";
  let description = [{
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS
