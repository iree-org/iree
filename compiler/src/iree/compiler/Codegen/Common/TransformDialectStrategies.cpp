// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"

#include <numeric>
#include <type_traits>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
namespace mlir {
namespace iree_compiler {

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

// TODO: significantly better namespacing.
using ::mlir::iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using ::mlir::iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using ::mlir::iree_compiler::IREE::transform_dialect::
    ForeachThreadToWorkgroupOp;
using ::mlir::iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using ::mlir::iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
using ::mlir::iree_compiler::IREE::transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp;
using ::mlir::iree_compiler::IREE::transform_dialect::
    VectorToWarpExecuteOnLane0Op;
using ::mlir::iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using ::mlir::transform::FuseIntoContainingOp;
using ::mlir::transform::MatchOp;
using ::mlir::transform::MergeHandlesOp;
using ::mlir::transform::PrintOp;
using ::mlir::transform::SplitHandlesOp;
using ::mlir::transform::SplitReductionOp;
using ::mlir::transform::TileToForeachThreadOp;
using ::mlir::transform::VectorizeOp;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandlesOp>(matchedH,
                                          /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

/// Prints `handles` in order. Prints the whole IR if `handles` is empty.
static void print(ImplicitLocOpBuilder &b, ValueRange handles = {}) {
  if (handles.empty()) b.create<PrintOp>();
  for (auto h : handles) b.create<PrintOp>(h);
}

namespace {
struct BatchFusionSpec {};
struct IterativeFusionSpec {};

struct TileOrNumThreadHandle {};
struct TileOrNumThreadList {};
}  // namespace

/// Performs the following transformations:
///   1. Tiles `rootH` to scf.foreach_thread to with `tileSizesOrNumThreads`
///      according to whether spec is a TileSizesSpec or a NumThreadsSpec.
///   2. Maps the resulting scf.foreach_thread to threads according to
///      `threadDimMapping`.
///   3. Iterates over `opsHToFuse` in order and fuses into the containing op.
/// Returns a handle to the resulting scf.foreach_thread.
///
/// Fusion is controlled by the BatchOrIterativeFusion templatetype:
///   1. In batch mode, a single fusion command is issued and a topological sort
///      is automatically computed by the fusion. Since this applies a single
///      fusion, no interleaved canonicalization/cse/enabling occur and the
///      resulting fusion may not be as good.
///   2. In iterative mode, the user is responsible for providing the fusion
///      order but interleaved canonicalization/cse/enabling occur which may
///      result in better fusion results.
///      TODO: Transform dialect op to apply topological sort and avoid the
///      manual intervention.
/// If `resultingFusedOpsHandles` is a non-null pointer, the fused operation are
/// appended in order.
// TODO: apply forwarding pattern.
template <typename TilingTransformOp, typename TileOrNumThreadSpec,
          typename BatchOrIterativeFusion, typename TileOrNumThreadKind>
static Value tileAndFuseAndDistributeImpl(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    std::conditional_t<std::is_same_v<TileOrNumThreadKind, TileOrNumThreadList>,
                       ArrayRef<int64_t>, ArrayRef<OpFoldResult>>
        tileSizesOrNumThreads,
    ArrayAttr threadDimMapping, SmallVector<Value> *resultingFusedOpsHandles) {
  auto tileToForeachOp = b.create<TilingTransformOp>(
      rootH, tileSizesOrNumThreads, TileOrNumThreadSpec(), threadDimMapping);
  Value foreachThreadH = tileToForeachOp.getForeachThreadOp();
  if (std::is_same<BatchOrIterativeFusion, BatchFusionSpec>::value) {
    Value mergedOpsH =
        b.create<MergeHandlesOp>(opsHToFuse, /*deduplicate=*/true);
    Value fusedH = b.create<FuseIntoContainingOp>(mergedOpsH, foreachThreadH);
    (void)fusedH;
    assert(!resultingFusedOpsHandles && "Handle needs unpacking");
  } else {
    for (Value h : opsHToFuse) {
      Value fusedH = b.create<FuseIntoContainingOp>(h, foreachThreadH);
      if (resultingFusedOpsHandles) resultingFusedOpsHandles->push_back(fusedH);
    }
  }
  return foreachThreadH;
}

/// Call tileAndFuseAndDistributeImpl with ArrayRef<int64_t> tilesSizes.
template <typename TilingTransformOp = TileToForeachThreadOp,
          typename BatchOrIterativeFusion = BatchFusionSpec>
static Value buildTFDWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<int64_t> tileSizes, ArrayAttr threadDimMapping = {},
    SmallVector<Value> *resultingFusedOpsHandles = nullptr) {
  return tileAndFuseAndDistributeImpl<
      TilingTransformOp, transform::TileSizesSpec, BatchOrIterativeFusion,
      TileOrNumThreadList>(b, rootH, opsHToFuse, tileSizes, threadDimMapping,
                           resultingFusedOpsHandles);
}

/// Call tileAndFuseAndDistributeImpl with a handle to multiple tilesSizes.
template <typename TilingTransformOp = TileToForeachThreadOp,
          typename BatchOrIterativeFusion = BatchFusionSpec>
static Value buildTFDWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    Value tileSizeHandle, ArrayAttr threadDimMapping = {},
    SmallVector<Value> *resultingFusedOpsHandles = nullptr) {
  return tileAndFuseAndDistributeImpl<
      TilingTransformOp, transform::TileSizesSpec, BatchOrIterativeFusion,
      TileOrNumThreadHandle>(b, rootH, opsHToFuse,
                             ArrayRef<OpFoldResult>{tileSizeHandle},
                             threadDimMapping, resultingFusedOpsHandles);
}

/// Call tileAndFuseAndDistributeImpl with ArrayRef<int64_t> numThreads.
template <typename TilingTransformOp = TileToForeachThreadOp,
          typename BatchOrIterativeFusion = BatchFusionSpec>
static Value buildTFDWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<int64_t> numThreads, ArrayAttr threadDimMapping = {},
    SmallVector<Value> *resultingFusedOpsHandles = nullptr) {
  return tileAndFuseAndDistributeImpl<
      TilingTransformOp, transform::NumThreadsSpec, BatchOrIterativeFusion>(
      b, rootH, opsHToFuse, numThreads, threadDimMapping,
      resultingFusedOpsHandles);
}

/// Call tileAndFuseAndDistributeImpl with a handle to multiple numThreads.
template <typename TilingTransformOp = TileToForeachThreadOp,
          typename BatchOrIterativeFusion = BatchFusionSpec>
static Value buildTFDWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    Value numThreads, ArrayAttr threadDimMapping = {},
    SmallVector<Value> *resultingFusedOpsHandles = nullptr) {
  return tileAndFuseAndDistributeImpl<
      TilingTransformOp, transform::NumThreadsSpec, BatchOrIterativeFusion>(
      b, rootH, opsHToFuse, ArrayRef<OpFoldResult>{numThreads},
      threadDimMapping, resultingFusedOpsHandles);
}

/// Apply patterns and vectorize (for now always applies rank-reduction).
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
// TODO: configure patterns.
static Value buildVectorizeStrategy(ImplicitLocOpBuilder &b, Value funcH) {
  funcH = b.create<ApplyPatternsOp>(funcH, /*rankReducing=*/true);
  return b.create<VectorizeOp>(funcH);
}

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
static Value mapToBlockAndThreads(ImplicitLocOpBuilder &b, Value funcH,
                                  ArrayRef<int64_t> blockSize) {
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  return b.create<MapNestedForeachThreadToGpuThreadsOp>(funcH, blockSize);
}

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
static Value distributeVectors(ImplicitLocOpBuilder &b, Value funcH,
                               int64_t warpSize = 32) {
  funcH = b.create<ApplyPatternsOp>(funcH, /*rankReducing=*/true);
  Value ifH = b.create<MatchOp>(funcH, scf::IfOp::getOperationName());
  ifH = b.create<VectorToWarpExecuteOnLane0Op>(ifH, warpSize);
  b.create<VectorWarpDistributionOp>(funcH);
  return funcH;
}

/// Distribute to blocks using the current IREE lowering config.
///
/// The tiling and distributing to blocks is done within a transform SequenceOp.
/// It runs without interleaved canonicalize, CSE or enabling transforms which
/// allows the transform dialect to build payload IR and not risk seeing it
/// being DCE'd away immediately.
template <typename TileSizesType>
static Value buildReductionStrategyBlockDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, Value originalFillH,
    Value originalGenericH, TileSizesType tileSizes0Generic) {
  // Step 1. Split the reduction to get meatier parallelism.
  // TODO: use a scf.foreach_thread for this.
  auto splitReductionTransformOp =
      b.create<SplitReductionOp>(originalGenericH,
                                 /*splitFactor=*/2,
                                 /*insertSplitDimension=*/1);
  Value splitFillH = splitReductionTransformOp.getFillOp();
  Value splitLinalgH = splitReductionTransformOp.getSplitLinalgOp();
  Value combinerH = splitReductionTransformOp.getCombiningLinalgOp();
  // TODO: IREE needs own workgroup mapping attribute.
  // TODO: num of GPU block mapping attr is statically known here which is
  // brittle. In the future, the builder of scf.foreach_thread can trim the
  // number of mapping dims to the number of sizes.
  auto x = mlir::gpu::GPUBlockMappingAttr::get(b.getContext(),
                                               ::mlir::gpu::Blocks::DimX);
  // Step 2. First level of tiling + fusion parallelizes to blocks using
  // `tileSizes`.
  buildTFDWithTileSizes<TileToForeachThreadAndWorkgroupCountRegionOp>(
      b,
      /*rootH=*/combinerH,
      /*opsHToFuse=*/{originalFillH, splitFillH, splitLinalgH},
      tileSizes0Generic, b.getArrayAttr({x}));

  return variantH;
}

static Value buildReductionStrategyThreadDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, ArrayRef<int64_t> tileSizes1Fill,
    ArrayRef<int64_t> tileSizes1Generic) {
  // TODO: Relying on ordering is brittle, harden this.
  auto [fill2dH, parParRedH, fill1dH, parRedH] = matchAndUnpack<4>(
      b, variantH,
      ArrayRef<StringRef>{linalg::GenericOp::getOperationName(),
                          linalg::FillOp::getOperationName()});
  auto z = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimZ);
  auto y = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimY);
  (void)z;
  (void)y;

  // clang-format off
  buildTFDWithTileSizes<TileToForeachThreadOp>(b,
                   /*rootH=*/parRedH,
                   /*opsHToFuse=*/{fill1dH},
                  // TODO: activate this, we have the information but it does
                  // not generate the IR we want atm.
                  //  /*tileSizes=*/tileSizes1Fill,
                  //  /*threadDimMapping=*/b.getArrayAttr({y}));
                   /*tileSizes=*/tileSizes1Fill,
                   /*threadDimMapping=*/b.getArrayAttr({z}));
  buildTFDWithTileSizes<TileToForeachThreadOp>(b,
                   /*rootH=*/parParRedH,
                   /*opsHToFuse=*/{fill2dH},
                  // TODO: activate this, we have the information but it does
                  // not generate the IR we want atm.
                  //  /*tileSizes=*/tileSizes1Generic,
                  //  /*threadDimMapping=*/b.getArrayAttr({y}));
                   /*tileSizes=*/tileSizes1Generic,
                   /*threadDimMapping=*/b.getArrayAttr({z,y}));
  // clang-format on
  return variantH;
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
static void buildReductionCudaStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                       TileSizesListType &tileSizes,
                                       std::array<int64_t, 3> &workgroupSize) {
  // Step 0. Match the ops.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = buildReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, originalGenericH, tileSizes[0]);

  // Step 2. Second level of tiling + fusion parallelizes to threads.
  variantH = buildReductionStrategyThreadDistributionPart(
      b, variantH, tileSizes[1], tileSizes[2]);

  // Step 3. Rank-reduce and vectorize.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildVectorizeStrategy(b, funcH);

  // Step 4. Bufferize.
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may ned hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = mapToBlockAndThreads(b, funcH, workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  distributeVectors(b, funcH);
}

namespace {

/// A tag indicating the shape being static or dynamic, for use with the
/// structured op matcher.
enum class ShapeKind { Static, Dynamic };

/// A placeholder indicating the structured op matcher to check the predicate
/// for all dimensions.
struct AllDims {};

/// A placeholder indicating the structured op matcher to check the predicate
/// for all operands of the relevant kind.
struct AllOperands {};

/// Base class for predicate parameters that can be described with the single
/// value. Concrete predicate parameters should inherit this and forward the
/// constructor via `using Base::Base`.
template <typename T>
struct SingleValuePredicateParam {
  using Base = SingleValuePredicateParam<T>;
  explicit SingleValuePredicateParam(T value) : value(value) {}
  const T value;
};

/// Indicates that the dimension must be divisible by the given value.
struct DivisibleBy : public SingleValuePredicateParam<int64_t> {
  using Base::Base;
};

/// Indicates that the number of entities must be equal to the given value.
struct NumEqualsTo : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Indicates that the bit width of the elemental type must be equal to the give
/// value.
struct ElementTypeBitWidth : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Predicate tag indicating that the affine map is a permutation.
struct IsPermutation {};

/// Predicate tag indicating that the reduction is produced by a single combiner
/// operation.
struct SingleCombinerReduction {};

class StructuredOpMatcher;
StructuredOpMatcher m_StructuredOp();

/// Structured op matcher with additional predicates attachable through the
/// fluent, a.k.a. chainable, API. Note that public API must *not* accept
/// additional callbacks even; new predicates should be added instead when
/// necessary. Not only this decreases the depth of the callback stack and
/// increases readability, it also allows us to port the matcher to a
/// declarative format using PDL and/or Transform dialect in the future. The
/// latter will become impossible with arbitrary C++ callbacks.
class StructuredOpMatcher {
  friend StructuredOpMatcher m_StructuredOp();
  using PredicateFn = std::function<bool(linalg::LinalgOp)>;

 public:
  /// Matches the given operation, hook for `matchPattern`.
  bool match(Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) return false;

    return llvm::all_of(
        predicates, [linalgOp](const PredicateFn &fn) { return fn(linalgOp); });
  }

  /// Adds a predicate checking that the given iteration space dimension is
  /// static/dynamic. The dimension index may be negative, in which case
  /// dimensions are counted from the last one Python-style, or be an AllDims
  /// tag, in which case all dimensions are checked. This may be eventually
  /// extended to slices and/or lists of dimensions.
  StructuredOpMatcher &dim(int64_t dimension, ShapeKind kind) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
      int64_t transformedDimension =
          dimension >= 0 ? dimension : shape.size() + dimension;
      if (transformedDimension >= shape.size()) return false;
      return ShapedType::isDynamic(shape[transformedDimension]) ^
             (kind == ShapeKind::Static);
    });
    return *this;
  }
  StructuredOpMatcher &dim(AllDims tag, ShapeKind kind) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
      return llvm::all_of(shape, [=](int64_t dimension) {
        return ShapedType::isDynamic(dimension) ^ (kind == ShapeKind::Static);
      });
    });
    return *this;
  }

  /// Adds a predicate checking that the given iteration space dimension has the
  /// given iterator type, e.g., parallel or reduction. The dimension index may
  /// be negative, in which case dimensions are counted from the last one
  /// Python-style, or be an AllDims tag, in which case all dimensions are
  /// checked. This may be eventually extended to slices and/or lists of
  /// dimensions.
  StructuredOpMatcher &dim(int64_t dimension, utils::IteratorType kind) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      unsigned rank = linalgOp.getNumLoops();
      int64_t transformedDimension =
          dimension >= 0 ? dimension : rank + dimension;
      if (transformedDimension >= rank) return false;

      utils::IteratorType iteratorKind =
          linalgOp.getIteratorTypesArray()[transformedDimension];
      return iteratorKind == kind;
    });
    return *this;
  }
  StructuredOpMatcher &dim(AllDims tag, utils::IteratorType kind) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      return llvm::all_of(linalgOp.getIteratorTypesArray(),
                          [=](utils::IteratorType iteratorType) {
                            return iteratorType == kind;
                          });
    });
    return *this;
  }

  /// Adds a predicate checking that the given iteration space dimension is
  /// statically known to be divisible by the given value. The dimension index
  /// may be negative, in which case dimensions are counted from the last one
  /// Python-style.
  StructuredOpMatcher &dim(int64_t dimension, DivisibleBy divisibleBy) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      unsigned rank = linalgOp.getNumLoops();
      int64_t transformedDimension =
          dimension >= 0 ? dimension : rank + dimension;
      if (transformedDimension >= rank) return false;

      int64_t size = linalgOp.getStaticLoopRanges()[transformedDimension];
      return !ShapedType::isDynamic(size) && (size % divisibleBy.value == 0);
    });
    return *this;
  }

  /// Adds a predicate checking that all input operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &input(AllOperands tag, IsPermutation) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      // all_of with a lambda requires const-casting dance, so using a loop.
      for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
        if (!linalgOp.getMatchingIndexingMap(operand).isPermutation())
          return false;
      }
      return true;
    });
    return *this;
  }

  /// Adds a predicate checking that the structured op has the given number of
  /// outputs.
  StructuredOpMatcher &output(NumEqualsTo num) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      return linalgOp.getNumDpsInits() == num.value;
    });
    return *this;
  }

  /// Adds a predicate checking that the bit width of the elemental type of the
  /// structured op output at the given position is equal to the given value.
  StructuredOpMatcher &output(int64_t position, ElementTypeBitWidth width) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      int64_t updatedPosition =
          position >= 0 ? position : linalgOp.getNumDpsInits() + position;
      if (updatedPosition >= linalgOp.getNumDpsInits()) return false;
      auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
                            ->get()
                            .getType()
                            .dyn_cast<ShapedType>();
      return shapedType && shapedType.getElementType().isIntOrFloat() &&
             shapedType.getElementType().getIntOrFloatBitWidth() == width.value;
    });
    return *this;
  }

  /// Adds a predicate checking that the output of the structured op is produced
  /// by a reduction with a single-operation combinator (such as addf or mulf,
  /// but not a compare+select pair).
  StructuredOpMatcher &output(int64_t position, SingleCombinerReduction tag) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      int64_t updatedPosition =
          position >= 0 ? position : linalgOp.getNumDpsInits() + position;
      if (updatedPosition >= linalgOp.getNumDpsInits()) return false;
      SmallVector<Operation *> combinerOps;
      return matchReduction(linalgOp.getRegionOutputArgs(), updatedPosition,
                            combinerOps) &&
             llvm::hasSingleElement(combinerOps);
    });
    return *this;
  }

 private:
  /// Additional predicates to be checked on the structured op.
  SmallVector<PredicateFn> predicates;
};

StructuredOpMatcher m_StructuredOp() { return StructuredOpMatcher(); }
}  // namespace

static constexpr unsigned cudaWarpSize = 32;

/// Matcher.
static bool matchGPUReduction(linalg::LinalgOp op, TileSizesListType &tileSizes,
                              std::array<int64_t, 3> &workgroupSize) {
  // TODO: match the sequence the strategy supports.
  auto pattern = m_StructuredOp()
                     .dim(AllDims(), ShapeKind::Static)
                     .dim(-1, utils::IteratorType::reduction)
                     .dim(-1, DivisibleBy(cudaWarpSize))
                     // Can be extended to projected permutation with broadcast.
                     .input(AllOperands(), IsPermutation())
                     .output(NumEqualsTo(1))
                     // Only single combiner over 32 bits for now due to
                     // reduction distribution.
                     .output(0, ElementTypeBitWidth(32))
                     .output(0, SingleCombinerReduction());
  if (!matchPattern(op, pattern)) return false;

  // Hardcoded workagroup size, this could be deduced from the reduction dim.
  workgroupSize = {32, 2, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  SmallVector<int64_t, 4> fillTileSize = {1, 0, 0};
  SmallVector<int64_t, 4> genericTileSize = {1, 1, 0};
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  tileSizes.emplace_back(std::move(fillTileSize));
  tileSizes.emplace_back(std::move(genericTileSize));
  return true;
}

static bool matchCPUReduction(linalg::LinalgOp op,
                              SmallVector<int64_t> &tileSize,
                              std::array<int64_t, 3> &workgroupSize) {
  // TODO: match the sequence the strategy supports.
  auto pattern = m_StructuredOp()
                     .dim(AllDims(), ShapeKind::Static)
                     .dim(-1, utils::IteratorType::reduction)
                     .output(NumEqualsTo(1));

  // Hardcoded workagroup size, this could be deduced from the reduction dim.
  workgroupSize = {32, 2, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  tileSize.append(numLoops, 1);
  return true;
}

using StrategyBuilderFn = std::function<void(ImplicitLocOpBuilder &, Value)>;

static void createTransformRegion(func::FuncOp entryPoint,
                                  StrategyBuilderFn buildStrategy) {
  MLIRContext *ctx = entryPoint.getContext();
  Location loc = entryPoint.getLoc();
  OpBuilder b(ctx);
  auto mod = entryPoint->getParentOfType<ModuleOp>();
  b.setInsertionPointAfter(mod);
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  b.create<::transform_ext::CanonicalizedSequenceOp>(
      loc, transform::FailurePropagationMode::Suppress,
      [&](OpBuilder &b, Location loc, Value variantH) {
        ImplicitLocOpBuilder ib(loc, b);
        buildStrategy(ib, variantH);
        b.create<transform::YieldOp>(loc);
      });
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
static LogicalResult buildReductionCpuStrategy(
    ImplicitLocOpBuilder &b, Value variantH, ArrayRef<int64_t> workgroupSize,
    ArrayRef<int64_t> tileSizes0Generic) {
  // Step 0. Fetch transform information from the config and materialize it in
  // the payload IR.
  // TODO: this still requires specific knowledge of ops present in the IR
  // and is very brittle.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = buildReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, originalGenericH, tileSizes0Generic);

  // Step 2. Rank-reduce and buildVectorizeStrategy.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildVectorizeStrategy(b, funcH);

  // Step 3. Bufferize.
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);

  // Step 4. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);

  return success();
}

LogicalResult matchAndSetGPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op) {
  // 1. Match
  TileSizesListType tileSizes;
  std::array<int64_t, 3> workgroupSize;
  if (!matchGPUReduction(op, tileSizes, workgroupSize)) return failure();
  auto startegyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return buildReductionCudaStrategy(b, variant, tileSizes, workgroupSize);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, startegyBuilder);
  return success();
}

LogicalResult matchAndSetCPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op) {
  // 1. Match
  SmallVector<int64_t> tileSize;
  std::array<int64_t, 3> workgroupSize;
  if (!matchCPUReduction(op, tileSize, workgroupSize)) return failure();
  auto startegyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return buildReductionCpuStrategy(b, variant, tileSize, workgroupSize);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, startegyBuilder);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
