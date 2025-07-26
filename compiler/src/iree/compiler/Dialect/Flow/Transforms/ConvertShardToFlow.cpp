// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/ShardToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/Folding.h"
#include "iree/compiler/Utils/Indexing.h"
#include "iree/compiler/Utils/OpVisitor.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CONVERTSHARDTOFLOWPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static bool hasMoreThanOneShard(Operation *op) {
  int shardCount = 0;
  op->walk([&shardCount](shard::ShardOp shard) {
    ++shardCount;
    return shardCount > 1 ? WalkResult::interrupt() : WalkResult::advance();
  });
  return shardCount > 1;
}

static SmallVector<shard::GridAxis> getAllGridAxes(shard::GridOp grid) {
  SmallVector<shard::ShardAxis> res(grid.getRank());
  std::iota(res.begin(), res.end(), 0);
  return res;
}

using GridAndAxesSet =
    DenseSet<std::tuple<shard::GridOp, SmallVector<shard::ShardAxis>>>;

template <typename Op>
struct CollectiveOpVisitor {
  CollectiveOpVisitor(GridAndAxesSet &gridAndAxesSet,
                      SymbolTableCollection &symbolTableCollection)
      : gridAndAxesSet(gridAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    gridAndAxesSet.insert(std::make_tuple(
        symbolTableCollection.lookupNearestSymbolFrom<shard::GridOp>(
            op, op.getGridAttr()),
        llvm::to_vector(op.getGridAxes())));
  }

private:
  GridAndAxesSet &gridAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

template <typename Op>
struct CollectiveOpWithoutShardAxesVisitor {
  CollectiveOpWithoutShardAxesVisitor(
      GridAndAxesSet &gridAndAxesSet,
      SymbolTableCollection &symbolTableCollection)
      : gridAndAxesSet(gridAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    shard::GridOp grid =
        symbolTableCollection.lookupNearestSymbolFrom<shard::GridOp>(
            op, op.getGridAttr());
    gridAndAxesSet.insert(std::make_tuple(grid, getAllGridAxes(grid)));
  }

private:
  GridAndAxesSet &gridAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

void populateShardAndAxes(Operation *op, GridAndAxesSet &shardAndAxesSet,
                          SymbolTableCollection &symbolTableCollection) {
  OpVisitorCollection opVisitors;
  opVisitors.emplaceVisitors<
      CollectiveOpVisitor<shard::AllGatherOp>,
      CollectiveOpVisitor<shard::AllReduceOp>,
      CollectiveOpVisitor<shard::AllToAllOp>,
      CollectiveOpVisitor<shard::ReduceScatterOp>,
      CollectiveOpWithoutShardAxesVisitor<shard::ProcessLinearIndexOp>>(
      shardAndAxesSet, symbolTableCollection);

  op->walk([&opVisitors](Operation *op) {
    opVisitors(op);
    return WalkResult::advance();
  });
}

// Derives a channel symbol name from the given shard axes.
static SmallString<64> getGridChannelName(shard::GridOp grid,
                                          ArrayRef<shard::ShardAxis> axes) {
  SmallString<64> res;
  llvm::raw_svector_ostream stream(res);
  stream << "_shard_" << grid.getSymName();
  if (axes.empty()) {
    return res;
  }
  stream << "_axes";
  for (shard::ShardAxis axis : axes) {
    stream << "_" << axis;
  }
  return res;
}

static bool isDefaultChannel(shard::GridOp grid,
                             ArrayRef<shard::GridAxis> gridAxes) {
  if (grid.getRank() != static_cast<int64_t>(gridAxes.size())) {
    return false;
  }
  return isIdentityPermutation(gridAxes);
}

static Value getDefaultChannel(Location loc, shard::GridOp grid,
                               bool useNamedDefaultChannels,
                               OpBuilder &builder) {
  if (useNamedDefaultChannels)
    return builder.create<IREE::Flow::ChannelDefaultOp>(loc, grid.getSymName());
  else
    return builder.create<IREE::Flow::ChannelDefaultOp>(loc);
}

static Value buildCachedChannelLoading(Location loc, shard::GridOp grid,
                                       ArrayRef<shard::GridAxis> gridAxes,
                                       bool useNamedDefaultChannels,
                                       OpBuilder &builder) {
  if (isDefaultChannel(grid, gridAxes)) {
    return getDefaultChannel(loc, grid, useNamedDefaultChannels, builder);
  }
  // TODO: lookup the shard name instead of generating it again - today this
  // will fail if there are any conflicting names during channel creation.
  return builder.create<IREE::Util::GlobalLoadOp>(
      loc, builder.getType<IREE::Flow::ChannelType>(),
      getGridChannelName(grid, gridAxes));
}

// Remove from `values` elements that have indices present in filter.
static SmallVector<Value> filterOutByIndex(ArrayRef<Value> values,
                                           ArrayRef<shard::ShardAxis> filter) {
  SmallVector<Value> res;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!llvm::is_contained(filter, i)) {
      res.push_back(values[i]);
    }
  }
  return res;
}

static Value buildChannelCreation(shard::GridOp grid,
                                  ArrayRef<shard::GridAxis> gridAxes,
                                  bool useNamedDefaultChannels,
                                  ImplicitLocOpBuilder &builder) {
  assert(grid);
  Value shardChannel = getDefaultChannel(builder.getLoc(), grid,
                                         useNamedDefaultChannels, builder);
  SmallVector<Value> gridProcessMultiIndex =
      builder.create<shard::ProcessMultiIndexOp>(grid).getResults();
  SmallVector<Value> gridShape =
      builder.create<shard::GridShapeOp>(grid).getResults();
  SmallVector<Value> reorderedGridIndex =
      permute(ArrayRef<Value>(gridProcessMultiIndex), gridAxes);
  SmallVector<Value> reorderedGridShape =
      permute(ArrayRef<Value>(gridShape), gridAxes);
  SmallVector<Value> groupIndex =
      filterOutByIndex(gridProcessMultiIndex, gridAxes);
  SmallVector<Value> groupsShape = filterOutByIndex(gridShape, gridAxes);
  OpFoldResult reorderedProcessLinearIndex =
      linearIndexFromShape(toOpFoldResults(reorderedGridIndex),
                           toOpFoldResults(reorderedGridShape), builder);
  OpFoldResult color = linearIndexFromShape(
      toOpFoldResults(groupIndex), toOpFoldResults(groupsShape), builder);
  return builder.create<IREE::Flow::ChannelSplitOp>(
      shardChannel,
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(), color),
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(),
                                      reorderedProcessLinearIndex));
}

static void buildChannelInitializer(shard::GridOp grid,
                                    ArrayRef<shard::GridAxis> gridAxes,
                                    bool useNamedDefaultChannels,
                                    ImplicitLocOpBuilder &builder) {
  IREE::Util::InitializerOp initOp =
      builder.create<IREE::Util::InitializerOp>();
  Block *block = builder.createBlock(&initOp.getBody());
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(block);
  Value channel =
      buildChannelCreation(grid, gridAxes, useNamedDefaultChannels, builder);
  builder.create<IREE::Util::GlobalStoreOp>(channel,
                                            getGridChannelName(grid, gridAxes));
  builder.create<IREE::Util::ReturnOp>();
}

// Construct a Flow channel inside `module` using
// util.global and util.initializer.
static void buildGlobalChannelCreation(shard::GridOp grid,
                                       ArrayRef<shard::GridAxis> gridAxes,
                                       bool useNamedDefaultChannels,
                                       ModuleOp module, OpBuilder &opBuilder) {
  if (isDefaultChannel(grid, gridAxes)) {
    return;
  }

  ImplicitLocOpBuilder builder(grid.getLoc(), opBuilder);
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(&module.getBodyRegion().getBlocks().front());

  auto channelName = getGridChannelName(grid, gridAxes);
  builder.create<IREE::Util::GlobalOp>(
      builder.getStringAttr("private"), channelName,
      builder.getType<IREE::Flow::ChannelType>(), false, TypedAttr(),
      builder.getAttr<IREE::Util::InlineNeverAttr>());
  buildChannelInitializer(grid, gridAxes, useNamedDefaultChannels, builder);
}

static void createChannels(ModuleOp moduleOp,
                           SymbolTableCollection &symbolTableCollection,
                           GridAndAxesSet &gridAndAxesSet,
                           bool useNamedDefaultChannels) {
  populateShardAndAxes(moduleOp, gridAndAxesSet, symbolTableCollection);

  OpBuilder builder(moduleOp->getContext());

  // Sort for deterministic testing with FileCheck.
  auto gridAndAxesSetSorted = llvm::to_vector(gridAndAxesSet);
  llvm::sort(gridAndAxesSetSorted, [](auto &a, auto &b) {
    int nameCompareRes =
        std::get<0>(a).getSymName().compare(std::get<0>(b).getSymName());
    if (nameCompareRes == 0)
      return std::get<1>(a) < std::get<1>(b);
    return nameCompareRes < 0;
  });
  for (auto &[shard, shardAxes] : llvm::make_range(
           gridAndAxesSetSorted.rbegin(), gridAndAxesSetSorted.rend())) {
    buildGlobalChannelCreation(shard, shardAxes, useNamedDefaultChannels,
                               moduleOp, builder);
  }
}

static LogicalResult
convertCollectives(ModuleOp moduleOp,
                   SymbolTableCollection &symbolTableCollection,
                   bool useNamedDefaultChannels) {
  RewritePatternSet patterns(moduleOp->getContext());
  IREE::Flow::populateShardToFlowCollectivesPatterns(
      patterns, symbolTableCollection,
      [&](Location loc, shard::GridOp grid,
          std::optional<ArrayRef<shard::GridAxis>> gridAxes,
          OpBuilder &builder) {
        if (gridAxes.has_value()) {
          return buildCachedChannelLoading(loc, grid, *gridAxes,
                                           useNamedDefaultChannels, builder);
        } else {
          return buildCachedChannelLoading(loc, grid, getAllGridAxes(grid),
                                           useNamedDefaultChannels, builder);
        }
      });
  return applyPatternsGreedily(moduleOp, std::move(patterns));
}

static void removeShardOps(GridAndAxesSet &gridAndAxesSet) {
  auto gridRange =
      llvm::map_range(gridAndAxesSet, [](auto &v) { return std::get<0>(v); });
  DenseSet<shard::GridOp> gridOpsSet(std::begin(gridRange),
                                     std::end(gridRange));
  for (shard::GridOp op : gridOpsSet) {
    if (op)
      op.erase();
  }
}

struct ConvertShardToFlowPass
    : public impl::ConvertShardToFlowPassBase<ConvertShardToFlowPass> {
  void runOnOperation() override {
    SymbolTableCollection symbolTableCollection;
    GridAndAxesSet shardAndAxesSet;
    const bool useNamedDefaultChannels = hasMoreThanOneShard(getOperation());

    createChannels(getOperation(), symbolTableCollection, shardAndAxesSet,
                   useNamedDefaultChannels);
    if (failed(convertCollectives(getOperation(), symbolTableCollection,
                                  useNamedDefaultChannels))) {
      return signalPassFailure();
    }

    // Cleanup shard definition ops that are no longer referenced.
    removeShardOps(shardAndAxesSet);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
