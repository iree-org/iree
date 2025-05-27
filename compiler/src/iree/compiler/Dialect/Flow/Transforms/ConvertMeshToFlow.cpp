// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/Folding.h"
#include "iree/compiler/Utils/Indexing.h"
#include "iree/compiler/Utils/OpVisitor.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CONVERTMESHTOFLOWPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static bool hasMoreThanOneMesh(Operation *op) {
  int meshCount = 0;
  op->walk([&meshCount](mesh::MeshOp mesh) {
    ++meshCount;
    return meshCount > 1 ? WalkResult::interrupt() : WalkResult::advance();
  });
  return meshCount > 1;
}

static SmallVector<mesh::MeshAxis> getAllMeshAxes(mesh::MeshOp mesh) {
  SmallVector<mesh::MeshAxis> res(mesh.getRank());
  std::iota(res.begin(), res.end(), 0);
  return res;
}

using MeshAndAxesSet =
    DenseSet<std::tuple<mesh::MeshOp, SmallVector<mesh::MeshAxis>>>;

template <typename Op>
struct CollectiveOpVisitor {
  CollectiveOpVisitor(MeshAndAxesSet &meshAndAxesSet,
                      SymbolTableCollection &symbolTableCollection)
      : meshAndAxesSet(meshAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    meshAndAxesSet.insert(std::make_tuple(
        symbolTableCollection.lookupNearestSymbolFrom<mesh::MeshOp>(
            op, op.getMeshAttr()),
        llvm::to_vector(op.getMeshAxes())));
  }

private:
  MeshAndAxesSet &meshAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

template <typename Op>
struct CollectiveOpWithoutMeshAxesVisitor {
  CollectiveOpWithoutMeshAxesVisitor(
      MeshAndAxesSet &meshAndAxesSet,
      SymbolTableCollection &symbolTableCollection)
      : meshAndAxesSet(meshAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    mesh::MeshOp mesh =
        symbolTableCollection.lookupNearestSymbolFrom<mesh::MeshOp>(
            op, op.getMeshAttr());
    meshAndAxesSet.insert(std::make_tuple(mesh, getAllMeshAxes(mesh)));
  }

private:
  MeshAndAxesSet &meshAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

void populateMeshAndAxes(Operation *op, MeshAndAxesSet &meshAndAxesSet,
                         SymbolTableCollection &symbolTableCollection) {
  OpVisitorCollection opVisitors;
  opVisitors.emplaceVisitors<
      CollectiveOpVisitor<mesh::AllGatherOp>,
      CollectiveOpVisitor<mesh::AllReduceOp>,
      CollectiveOpVisitor<mesh::AllToAllOp>,
      CollectiveOpVisitor<mesh::ReduceScatterOp>,
      CollectiveOpWithoutMeshAxesVisitor<mesh::ProcessLinearIndexOp>>(
      meshAndAxesSet, symbolTableCollection);

  op->walk([&opVisitors](Operation *op) {
    opVisitors(op);
    return WalkResult::advance();
  });
}

// Derives a channel symbol name from the given mesh axes.
static SmallString<64> getMeshChannelName(mesh::MeshOp mesh,
                                          ArrayRef<mesh::MeshAxis> axes) {
  SmallString<64> res;
  llvm::raw_svector_ostream stream(res);
  stream << "_mesh_" << mesh.getSymName();
  if (axes.empty()) {
    return res;
  }
  stream << "_axes";
  for (mesh::MeshAxis axis : axes) {
    stream << "_" << axis;
  }
  return res;
}

static bool isDefaultChannel(mesh::MeshOp mesh,
                             ArrayRef<mesh::MeshAxis> meshAxes) {
  if (mesh.getRank() != static_cast<int64_t>(meshAxes.size())) {
    return false;
  }
  return isIdentityPermutation(meshAxes);
}

static Value getDefaultChannel(Location loc, mesh::MeshOp mesh,
                               bool useNamedDefaultChannels,
                               OpBuilder &builder) {
  if (useNamedDefaultChannels)
    return builder.create<IREE::Flow::ChannelDefaultOp>(loc, mesh.getSymName());
  else
    return builder.create<IREE::Flow::ChannelDefaultOp>(loc);
}

static Value buildCachedChannelLoading(Location loc, mesh::MeshOp meshOp,
                                       ArrayRef<mesh::MeshAxis> meshAxes,
                                       bool useNamedDefaultChannels,
                                       OpBuilder &builder) {
  if (isDefaultChannel(meshOp, meshAxes)) {
    return getDefaultChannel(loc, meshOp, useNamedDefaultChannels, builder);
  }
  // TODO: lookup the mesh name instead of generating it again - today this will
  // fail if there are any conflicting names during channel creation.
  return builder.create<IREE::Util::GlobalLoadOp>(
      loc, builder.getType<IREE::Flow::ChannelType>(),
      getMeshChannelName(meshOp, meshAxes));
}

// Remove from `values` elements that have indices present in filter.
static SmallVector<Value> filterOutByIndex(ArrayRef<Value> values,
                                           ArrayRef<mesh::MeshAxis> filter) {
  SmallVector<Value> res;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!llvm::is_contained(filter, i)) {
      res.push_back(values[i]);
    }
  }
  return res;
}

static Value buildChannelCreation(mesh::MeshOp mesh,
                                  ArrayRef<mesh::MeshAxis> meshAxes,
                                  bool useNamedDefaultChannels,
                                  ImplicitLocOpBuilder &builder) {
  assert(mesh);
  Value meshChannel = getDefaultChannel(builder.getLoc(), mesh,
                                        useNamedDefaultChannels, builder);
  SmallVector<Value> meshProcessMultiIndex =
      builder.create<mesh::ProcessMultiIndexOp>(mesh).getResults();
  SmallVector<Value> meshShape =
      builder.create<mesh::MeshShapeOp>(mesh).getResults();
  SmallVector<Value> reorderedMeshIndex =
      permute(ArrayRef<Value>(meshProcessMultiIndex), meshAxes);
  SmallVector<Value> reorderedMeshShape =
      permute(ArrayRef<Value>(meshShape), meshAxes);
  SmallVector<Value> groupIndex =
      filterOutByIndex(meshProcessMultiIndex, meshAxes);
  SmallVector<Value> groupsShape = filterOutByIndex(meshShape, meshAxes);
  OpFoldResult reorderedProcessLinearIndex =
      linearIndexFromShape(toOpFoldResults(reorderedMeshIndex),
                           toOpFoldResults(reorderedMeshShape), builder);
  OpFoldResult color = linearIndexFromShape(
      toOpFoldResults(groupIndex), toOpFoldResults(groupsShape), builder);
  return builder.create<IREE::Flow::ChannelSplitOp>(
      meshChannel,
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(), color),
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(),
                                      reorderedProcessLinearIndex));
}

static void buildChannelInitializer(mesh::MeshOp mesh,
                                    ArrayRef<mesh::MeshAxis> meshAxes,
                                    bool useNamedDefaultChannels,
                                    ImplicitLocOpBuilder &builder) {
  IREE::Util::InitializerOp initOp =
      builder.create<IREE::Util::InitializerOp>();
  Block *block = builder.createBlock(&initOp.getBody());
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(block);
  Value channel =
      buildChannelCreation(mesh, meshAxes, useNamedDefaultChannels, builder);
  builder.create<IREE::Util::GlobalStoreOp>(channel,
                                            getMeshChannelName(mesh, meshAxes));
  builder.create<IREE::Util::ReturnOp>();
}

// Construct a Flow channel inside `module` using
// util.global and util.initializer.
static void buildGlobalChannelCreation(mesh::MeshOp mesh,
                                       ArrayRef<mesh::MeshAxis> meshAxes,
                                       bool useNamedDefaultChannels,
                                       ModuleOp module, OpBuilder &opBuilder) {
  if (isDefaultChannel(mesh, meshAxes)) {
    return;
  }

  ImplicitLocOpBuilder builder(mesh.getLoc(), opBuilder);
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(&module.getBodyRegion().getBlocks().front());

  auto channelName = getMeshChannelName(mesh, meshAxes);
  builder.create<IREE::Util::GlobalOp>(
      builder.getStringAttr("private"), channelName,
      builder.getType<IREE::Flow::ChannelType>(), false, TypedAttr(),
      builder.getAttr<IREE::Util::InlineNeverAttr>());
  buildChannelInitializer(mesh, meshAxes, useNamedDefaultChannels, builder);
}

static void createChannels(ModuleOp moduleOp,
                           SymbolTableCollection &symbolTableCollection,
                           MeshAndAxesSet &meshAndAxesSet,
                           bool useNamedDefaultChannels) {
  populateMeshAndAxes(moduleOp, meshAndAxesSet, symbolTableCollection);

  OpBuilder builder(moduleOp->getContext());

  // Sort for deterministic testing with FileCheck.
  auto meshAndAxesSetSorted = llvm::to_vector(meshAndAxesSet);
  llvm::sort(meshAndAxesSetSorted, [](auto &a, auto &b) {
    int nameCompareRes =
        std::get<0>(a).getSymName().compare(std::get<0>(b).getSymName());
    if (nameCompareRes == 0)
      return std::get<1>(a) < std::get<1>(b);
    return nameCompareRes < 0;
  });
  for (auto &[mesh, meshAxes] : llvm::make_range(meshAndAxesSetSorted.rbegin(),
                                                 meshAndAxesSetSorted.rend())) {
    buildGlobalChannelCreation(mesh, meshAxes, useNamedDefaultChannels,
                               moduleOp, builder);
  }
}

static LogicalResult
convertCollectives(ModuleOp moduleOp,
                   SymbolTableCollection &symbolTableCollection,
                   bool useNamedDefaultChannels) {
  RewritePatternSet patterns(moduleOp->getContext());
  IREE::Flow::populateMeshToFlowCollectivesPatterns(
      patterns, symbolTableCollection,
      [&](Location loc, mesh::MeshOp meshOp,
          std::optional<ArrayRef<mesh::MeshAxis>> meshAxes,
          OpBuilder &builder) {
        if (meshAxes.has_value()) {
          return buildCachedChannelLoading(loc, meshOp, *meshAxes,
                                           useNamedDefaultChannels, builder);
        } else {
          return buildCachedChannelLoading(loc, meshOp, getAllMeshAxes(meshOp),
                                           useNamedDefaultChannels, builder);
        }
      });
  return applyPatternsGreedily(moduleOp, std::move(patterns));
}

static void removeMeshOps(MeshAndAxesSet &meshAndAxesSet) {
  auto meshRange =
      llvm::map_range(meshAndAxesSet, [](auto &v) { return std::get<0>(v); });
  DenseSet<mesh::MeshOp> meshOpsSet(std::begin(meshRange), std::end(meshRange));
  for (mesh::MeshOp op : meshOpsSet) {
    if (op)
      op.erase();
  }
}

struct ConvertMeshToFlowPass
    : public IREE::Flow::impl::ConvertMeshToFlowPassBase<
          ConvertMeshToFlowPass> {
  void runOnOperation() override {
    SymbolTableCollection symbolTableCollection;
    MeshAndAxesSet meshAndAxesSet;
    const bool useNamedDefaultChannels = hasMoreThanOneMesh(getOperation());

    createChannels(getOperation(), symbolTableCollection, meshAndAxesSet,
                   useNamedDefaultChannels);
    if (failed(convertCollectives(getOperation(), symbolTableCollection,
                                  useNamedDefaultChannels))) {
      return signalPassFailure();
    }

    // Cleanup mesh definition ops that are no longer referenced.
    removeMeshOps(meshAndAxesSet);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
