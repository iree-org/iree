// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- ConvertToGPUPass.cpp -----------------------------------------------===//
//
// Partition computation within dispatch function to workgroups/workitems.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Loop utilities
//===----------------------------------------------------------------------===//

/// Builds an empty loop.for operation. The default builder adds an entry basic
/// block which needs to be avoided here.
static loop::ForOp buildEmptyForOp(Location loc, OpBuilder &builder, Value lb,
                                   Value ub, Value step) {
  OperationState state(loc, loop::ForOp::getOperationName());
  state.addOperands({lb, ub, step});
  state.addRegion();
  return cast<loop::ForOp>(builder.createOperation(state));
}

namespace {
struct LoopBounds {
  Value lb;
  Value ub;
  Value step;
};
}  // namespace

/// Replaces a loop.parallelOp with an optional loop.parallel op and nested
/// loop.for operations. To create the loop.parallel op as the outermost loop,
/// pass the lower bound, upper bound and steps in `newPLoopLbs`, `newPLoopUbs`,
/// and `newPLoopStep` respectively. The bounds of the inner loop.for operations
/// to be created are passed in `forLbs`, `forUbs`, and `forStep`. The
/// `permutation` vector contains a mapping from the original loop order, to the
/// loop order to be generated.
static Operation *replacePLoopOp(ConversionPatternRewriter &rewriter,
                                 loop::ParallelOp pLoopOp,
                                 ArrayRef<LoopBounds> newPLoopBounds,
                                 ArrayRef<LoopBounds> forBounds,
                                 ArrayRef<unsigned> permutation) {
  assert(!forBounds.empty() && "unhandled case of no loop.for created");
  unsigned numLoops = pLoopOp.getNumLoops();
  Location loc = pLoopOp.getLoc();
  assert(forBounds.size() + newPLoopBounds.size() == numLoops &&
         "cannot drop loops when splitting loop.parallel operation");
  assert(permutation.size() == numLoops);
  OpBuilder::InsertionGuard guard(rewriter);

  // Need a signature conversion for the body of the loop.parallel operation,
  // before can it can be used as the body of the innermost loop created here.
  TypeConverter::SignatureConversion signatureConverter(numLoops);
  Operation *outermostLoop = nullptr;
  auto permuteIt = permutation.begin();

  // Create the loop.parallel operation as the outermost loop, if specified.
  if (!newPLoopBounds.empty()) {
    auto lbs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) -> Value { return bounds.lb; }));
    auto ubs = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.ub; }));
    auto steps = llvm::to_vector<2>(llvm::map_range(
        newPLoopBounds, [](LoopBounds bounds) { return bounds.step; }));
    auto newPLoop = rewriter.create<loop::ParallelOp>(loc, lbs, ubs, steps);
    for (auto iv : newPLoop.getInductionVars()) {
      signatureConverter.remapInput(*permuteIt, iv);
      permuteIt++;
    }
    rewriter.setInsertionPointToStart(newPLoop.getBody());
    outermostLoop = newPLoop.getOperation();
  }

  // Generate the nested loop.for operations with the bounds passed.
  for (auto it : enumerate(forBounds)) {
    Value lb = it.value().lb, ub = it.value().ub, step = it.value().step;
    if (it.index() != forBounds.size() - 1) {
      auto forOp = rewriter.create<loop::ForOp>(loc, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.remapInput(*permuteIt, forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    } else {
      // For the last loop, move the body of the loop.parallel op as the body of
      // the loop after signature conversion.
      auto forOp = buildEmptyForOp(loc, rewriter, lb, ub, step);
      if (!outermostLoop) outermostLoop = forOp.getOperation();
      signatureConverter.addInputs(*permuteIt, rewriter.getIndexType());
      Region &pLoopOpRegion = pLoopOp.getLoopBody();
      rewriter.applySignatureConversion(&pLoopOpRegion, signatureConverter);
      Region &forOpRegion = forOp.getLoopBody();
      rewriter.inlineRegionBefore(pLoopOpRegion, forOpRegion,
                                  forOpRegion.begin());
    }
    permuteIt++;
  }
  rewriter.eraseOp(pLoopOp);
  return outermostLoop;
}

/// Serializes the dimensions of the loop.parallel specified in
/// `serializedDimensions`, by creating an nested loop.for operation for each
/// dimension.
// TODO(ravishankarm): Move this into LoopUtils.h in MLIR.
static Operation *serializeDimensions(ConversionPatternRewriter &rewriter,
                                      loop::ParallelOp pLoopOp,
                                      ArrayRef<unsigned> serializedDimensions) {
  assert(!serializedDimensions.empty() &&
         "unhandled corner case of no serializing dims");
  OpBuilder::InsertionGuard guard(rewriter);
  DenseSet<unsigned> serializedDimSet;
  serializedDimSet.insert(serializedDimensions.begin(),
                          serializedDimensions.end());
  assert(serializedDimSet.size() == serializedDimensions.size() &&
         "cannot repeat dimensions during serialization of loop.parallel");
  SmallVector<LoopBounds, 2> newPLoopBounds, forBounds;
  SmallVector<unsigned, 2> permutation;
  auto lbs = pLoopOp.lowerBound();
  auto ubs = pLoopOp.upperBound();
  auto steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, pLoopOp.getNumLoops())) {
    if (serializedDimSet.count(i)) {
      forBounds.push_back({lbs[i], ubs[i], steps[i]});
    } else {
      newPLoopBounds.push_back({lbs[i], ubs[i], steps[i]});
      permutation.push_back(i);
    }
  }
  permutation.append(serializedDimensions.begin(), serializedDimensions.end());
  return replacePLoopOp(rewriter, pLoopOp, newPLoopBounds, forBounds,
                        permutation);
}

/// Serialize all inner dimensions of a `pLoopOp` starting from `serializeFrom`.
static Operation *serializeDimensionsFrom(ConversionPatternRewriter &rewriter,
                                          loop::ParallelOp pLoopOp,
                                          unsigned serializeFrom) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(serializeFrom > 0 && "unhandled serializaing all dimensions");
  assert(serializeFrom < numLoops &&
         "unhandled corner case of no serialization");
  SmallVector<unsigned, 2> serializedDimensions;
  for (unsigned dim : llvm::seq(serializeFrom, numLoops))
    serializedDimensions.push_back(dim);
  return serializeDimensions(rewriter, pLoopOp, serializedDimensions);
}

//===----------------------------------------------------------------------===//
// GPU processor ID mapping utilities
//===----------------------------------------------------------------------===//

/// Distribute loop.parallel to processors with the processors logically
/// arranged with same dimensionality as the number of loops, i.e. a
/// loop.parallel with 2 loops to a 2D grid of processors. `processorIDs` and
/// `numProcessors` must be of same size as the number of loops and are the
/// values to use for process ID and number of processors along each dimension
/// in the distributed code.
static LogicalResult mapToProcessors(ConversionPatternRewriter &rewriter,
                                     loop::ParallelOp pLoopOp,
                                     ArrayRef<Value> processorIDs,
                                     ArrayRef<Value> numProcessors) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(numLoops == processorIDs.size() &&
         "expected as many pids as number of loops");
  assert(numLoops == numProcessors.size() &&
         "expected as many nprocs as number of loops");
  SmallVector<LoopBounds, 2> forBounds;
  SmallVector<unsigned, 2> permutation;
  forBounds.reserve(numLoops);
  permutation.reserve(numLoops);
  Location loc = pLoopOp.getLoc();
  auto lbs = pLoopOp.lowerBound(), ubs = pLoopOp.upperBound(),
       steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, processorIDs.size())) {
    Value mappedLb = rewriter.create<AddIOp>(
        loc, lbs[i], rewriter.create<MulIOp>(loc, steps[i], processorIDs[i]));
    Value mappedStep = rewriter.create<MulIOp>(loc, steps[i], numProcessors[i]);
    forBounds.push_back({mappedLb, ubs[i], mappedStep});
    permutation.push_back(i);
  }
  replacePLoopOp(rewriter, pLoopOp, /*newPLoopBounds=*/{}, forBounds,
                 permutation);
  return success();
}

/// Distribute loop.parallel to processors where `IdOp` is used to get the
/// processor ID and `DimOp` is used to get the number of processors along a
/// dimension.
template <typename IdOp, typename DimOp>
static LogicalResult mapToProcessor(ConversionPatternRewriter &rewriter,
                                    loop::ParallelOp pLoopOp) {
  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops > 3)
    pLoopOp =
        cast<loop::ParallelOp>(serializeDimensionsFrom(rewriter, pLoopOp, 3));
  SmallVector<Value, 2> pid, nprocs;
  pid.reserve(numLoops);
  nprocs.reserve(numLoops);
  Location loc = pLoopOp.getLoc();
  auto createIdAndNprocs = [&](StringRef dim) {
    pid.insert(pid.begin(), rewriter.create<IdOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getStringAttr(dim)));
    nprocs.insert(nprocs.begin(),
                  rewriter.create<DimOp>(loc, rewriter.getIndexType(),
                                         rewriter.getStringAttr(dim)));
  };
  createIdAndNprocs("x");
  if (numLoops > 1) createIdAndNprocs("y");
  if (numLoops > 2) createIdAndNprocs("z");
  return mapToProcessors(rewriter, pLoopOp, pid, nprocs);
}

/// Distribute the loop.parallel to workgroups.
static LogicalResult mapToWorkgroups(ConversionPatternRewriter &rewriter,
                                     loop::ParallelOp pLoopOp) {
  return mapToProcessor<gpu::BlockIdOp, gpu::GridDimOp>(rewriter, pLoopOp);
}

/// Distribute loop.parallel to workitems.
static LogicalResult mapToWorkitems(ConversionPatternRewriter &rewriter,
                                    loop::ParallelOp pLoopOp) {
  return mapToProcessor<gpu::ThreadIdOp, gpu::BlockDimOp>(rewriter, pLoopOp);
}

/// Check if the linalg operation has the specified attribute (set during
/// tiling to denote what processor heirarchy executes this operation).
template <typename LinalgOpTy>
bool checkMarkerValue(LinalgOpTy linalgOp, StringRef marker = "") {
  auto attr = linalgOp.template getAttrOfType<StringAttr>(
      linalg::LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker == "" || attr.getValue() == marker);
}

//===----------------------------------------------------------------------===//
// Pass and patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Pass to convert from tiled and fused linalg ops into gpu.func.
struct ConvertToGPUPass : public PassWrapper<ConvertToGPUPass, FunctionPass> {
  void runOnFunction() override;
};

/// Pattern to map loop.parallel to workgroups.
struct PartitionPLoopToWorkgroups
    : public OpConversionPattern<loop::ParallelOp> {
  using OpConversionPattern<loop::ParallelOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      loop::ParallelOp pLoopOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    return mapToWorkgroups(rewriter, pLoopOp);
  }
};

/// Map tiled linalg op to workitems by lowering it to loop.parallel and
/// partitioning it to workitems.
template <typename LinalgOpTy>
struct MapLinalgOpToWorkitems : public OpConversionPattern<LinalgOpTy> {
  using OpConversionPattern<LinalgOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LinalgOpTy linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Check for marker that specifies that the linalg op is to be partitioned
    // across threads within a workgroup.
    if (!checkMarkerValue(linalgOp, getWorkItemMarker())) return failure();
    Optional<linalg::LinalgLoops> loops =
        linalg::linalgLowerOpToLoops<loop::ParallelOp, LinalgOpTy>(rewriter,
                                                                   linalgOp);
    if (!loops) return failure();
    if (!loops.getValue().empty()) {
      loop::ParallelOp pLoopOp =
          dyn_cast<loop::ParallelOp>(loops.getValue()[0]);
      if (!pLoopOp || failed(mapToWorkitems(rewriter, pLoopOp)))
        return failure();
    }
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

/// Map linalg operation to loops to be executed by a single thread on the GPU.
template <typename LinalgOpTy>
struct ExecuteLinalgOpSequentially : public OpConversionPattern<LinalgOpTy> {
  using OpConversionPattern<LinalgOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LinalgOpTy linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If marker exists, then dont execute the op sequentially.
    if (checkMarkerValue(linalgOp) ||
        failed(linalg::linalgOpToLoops<LinalgOpTy>(rewriter, linalgOp)))
      return failure();
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

/// Remove the linalg.range operation created when lowering to loops.
struct RemoveLinalgRange : public OpConversionPattern<linalg::RangeOp> {
  using OpConversionPattern<linalg::RangeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      linalg::RangeOp rangeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!rangeOp.getResult().use_empty()) return failure();
    rewriter.eraseOp(rangeOp);
    return success();
  }
};
}  // namespace

void ConvertToGPUPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body)) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  // Ater this pass Linalg and loop.parallel ops should be gone.
  target.addIllegalOp<loop::ParallelOp>();
  target.addIllegalDialect<linalg::LinalgDialect>();
  // Reshape ops are treated legal since they just change the way the underlying
  // buffer is viewed. These are legalized downstream. They become no ops when
  // lowering to SPIR-V since the SPIR-V code uses linearized arrays.
  target.addLegalOp<linalg::ReshapeOp>();
  // Let the rest fall through.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  OwningRewritePatternList patterns;
  patterns.insert<
#define ADD_ALL_LINALG_PATTERNS(OP_NAME) \
  ExecuteLinalgOpSequentially<OP_NAME>, MapLinalgOpToWorkitems<OP_NAME>
      ADD_ALL_LINALG_PATTERNS(linalg::ConvOp),
      ADD_ALL_LINALG_PATTERNS(linalg::CopyOp),
      ADD_ALL_LINALG_PATTERNS(linalg::FillOp),
      ADD_ALL_LINALG_PATTERNS(linalg::GenericOp),
      ADD_ALL_LINALG_PATTERNS(linalg::IndexedGenericOp),
      ADD_ALL_LINALG_PATTERNS(linalg::MatmulOp),
      ADD_ALL_LINALG_PATTERNS(linalg::PoolingMaxOp),
      ADD_ALL_LINALG_PATTERNS(linalg::PoolingMinOp),
      ADD_ALL_LINALG_PATTERNS(linalg::PoolingSumOp),
#undef ADD_ALL_LINALG_PATTERNS
      PartitionPLoopToWorkgroups, RemoveLinalgRange>(context);

  if (failed(applyFullConversion(funcOp, target, patterns)))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> createConvertToGPUPass() {
  return std::make_unique<ConvertToGPUPass>();
}

static PassRegistration<ConvertToGPUPass> pass(
    "iree-convert-to-gpu", "Map tiled linalg and loop ops to GPU",
    [] { return std::make_unique<ConvertToGPUPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
