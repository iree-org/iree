// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific logic for lowering StableHLO collective ops to Flow
// dialect ops.

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Work in progress. The implementation is planned as several stages.
//
// For the first stage, a few simplifications are made to support simple models.
//
//   1. Single stream with deterministic order of execution
//   2. Single replica group for all collective ops
//   3. Only replicas without partition_id used
//
// These allow us to use a default channel for all communications, and there is
// 1:1 mapping from the replica IDs to the communication ranks. The attribute,
// use_global_device_ids, is always set in this case.
//
// The next stage is to support multiple replica groups. This needs a channel
// creation with a subset of processes, which should have another communication
// among the group. A possible strategy is to have the root process in the group
// (the first rank of the group) creates a channel and the other processes query
// the channel info from the root process. A key-value store using gRPC might be
// a good solution.
//
// Supporting partition_id comes next. This includes the support for various
// mode combinations for cross-replica and cross partition communication. See
// the stablehlo specification for more details about the different modes.

static std::optional<IREE::Flow::CollectiveElementType>
convertToFlowCollectiveElementType(Type type) {
  if (type.isF32()) {
    return IREE::Flow::CollectiveElementType::Float32;
  }

  if (type.isInteger(32)) {
    if (type.isSignedInteger()) {
      return IREE::Flow::CollectiveElementType::Sint32;
    }
    return IREE::Flow::CollectiveElementType::Uint32;
  }

  if (type.isF16()) {
    return IREE::Flow::CollectiveElementType::Float16;
  }

  if (type.isInteger(8)) {
    if (type.isSignedInteger()) {
      return IREE::Flow::CollectiveElementType::Sint8;
    }
    return IREE::Flow::CollectiveElementType::Uint8;
  }

  if (type.isInteger(16)) {
    if (type.isSignedInteger()) {
      return IREE::Flow::CollectiveElementType::Sint16;
    }
    return IREE::Flow::CollectiveElementType::Uint16;
  }

  if (type.isBF16()) {
    return IREE::Flow::CollectiveElementType::BFloat16;
  }

  if (type.isF64()) {
    return IREE::Flow::CollectiveElementType::Float64;
  }

  if (type.isInteger(64)) {
    if (type.isSignedInteger()) {
      return IREE::Flow::CollectiveElementType::Sint64;
    }
    return IREE::Flow::CollectiveElementType::Uint64;
  }

  return std::nullopt;
}

static std::optional<IREE::Flow::CollectiveReductionOp>
convertToFlowCollectiveReductionOp(const Operation &op) {
  if (isa<mlir::stablehlo::AddOp>(op)) {
    return IREE::Flow::CollectiveReductionOp::ReductionSum;
  }
  if (isa<mlir::stablehlo::MulOp>(op)) {
    return IREE::Flow::CollectiveReductionOp::ReductionProduct;
  }
  if (isa<mlir::stablehlo::MinOp>(op)) {
    return IREE::Flow::CollectiveReductionOp::ReductionMinimum;
  }
  if (isa<mlir::stablehlo::MaxOp>(op)) {
    return IREE::Flow::CollectiveReductionOp::ReductionMaximum;
  }
  // TODO: we may be able to detect an average operation and convert it
  // into IREE::Flow::CollectiveReductionOp::ReductionAverage.
  return std::nullopt;
}

static IREE::Flow::CollectiveElementTypeAttr getCollectiveElementTypeAttr(
    MLIRContext *context, RankedTensorType type) {
  std::optional<IREE::Flow::CollectiveElementType> collectiveElemType =
      convertToFlowCollectiveElementType(type.getElementType());
  if (!collectiveElemType) {
    return IREE::Flow::CollectiveElementTypeAttr();
  }
  return IREE::Flow::CollectiveElementTypeAttr::get(context,
                                                    *collectiveElemType);
}

template <typename T>
static LogicalResult checkCollectiveAttrs(T op, PatternRewriter &rewriter) {
  // Note that the channel handle attribute consists of two 64-bit values,
  // handle and type.
  int64_t handle =
      op.getChannelHandle() ? op.getChannelHandleAttr().getHandle() : 0;
  if (handle <= 0) {
    // When the channel handle attribute is not present, it means the
    // handle (a.k.a. channel_id in stablehlo) is 0. When this case is combined
    // with `use_global_device_ids=false`, the communication type is
    // `cross-replica`, but since there is only one replica group, it is
    // effectively the same as `flatten_ids`, which is supported.
    if (op.getUseGlobalDeviceIds()) {
      return rewriter.notifyMatchFailure(
          op, "must not set use_global_device_ids when channel_id <= 0");
    }
  }

  return success();
}

/// Returns `color` and `key` parameter values indexed by the rank of the
/// participant in |baseChannel|.
///
/// Examples:
///   (0),(1)     => colors=[0,1], keys=[0,0]
///   (0,1),(2,3) => colors=[0,0,1,1], keys=[0,1,0,1]
static std::pair<Value, Value> makeSplitColorAndKey(Location loc,
                                                    Value baseChannel,
                                                    DenseIntElementsAttr groups,
                                                    OpBuilder &builder) {
  IndexSet indexSet(loc, builder);
  Value noColor = indexSet.get(-1);
  if (!groups) return std::make_pair(noColor, noColor);

  auto groupsType = groups.getType().cast<RankedTensorType>();
  assert(groupsType.getRank() == 2);
  int64_t rows = groupsType.getShape()[0];
  int64_t cols = groupsType.getShape()[1];
  auto values = groups.getValues<int64_t>();

  // Find the max rank so we can size our tables. Today the tables are always
  // dense starting from rank 0 but we could offset the rank lookup if for
  // example all ranks started at some offset.
  int64_t maxRank = 0;
  for (int64_t rank : values) {
    maxRank = std::max(maxRank, rank);
  }

  // Table of <color, key> pairs indexed by rank. -1 is used to indicate that
  // a particular rank does not participate in any group.
  SmallVector<Value> colorTable(maxRank + 1, noColor);
  SmallVector<Value> keyTable(maxRank + 1, noColor);

  // Sparsely populate table with each rank getting a color/key pair.
  // Rows equate to colors (groups) and columns equate to keys (local ranks).
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      const int64_t index = i * cols + j;
      int64_t rank = values[index];
      // -1 represents a null value in a group, where the group does not
      // fully occupy the space in the row, e.g., [[0,1,2,3], [4,5,-1,-1]].
      if (rank != -1) {
        colorTable[rank] = indexSet.get(i);
        keyTable[rank] = indexSet.get(j);
      }
    }
  }

  // Lookup the color/key split parameters by indexing into the tables we
  // generated from the static op information.
  Value rank = builder.create<IREE::Flow::ChannelRankOp>(loc, baseChannel);
  Value color =
      builder.create<IREE::Util::SwitchOp>(loc, rank, noColor, colorTable);
  Value key =
      builder.create<IREE::Util::SwitchOp>(loc, rank, noColor, keyTable);
  return std::make_pair(color, key);
}

static DenseIntElementsAttr convertToRankGroupsByCrossReplica(
    DenseIntElementsAttr replicaGroups, int32_t numPartitions,
    OpBuilder &builder) {
  if (numPartitions < 1) {
    // Treat as a single partition.
    return replicaGroups;
  }

  auto groupsType = replicaGroups.getType().cast<RankedTensorType>();
  assert(groupsType.getRank() == 2);
  int rows = groupsType.getShape()[0];
  int cols = groupsType.getShape()[1];
  auto values = replicaGroups.getValues<int64_t>();
  SmallVector<Attribute> newValues;

  // The number of groups is (rows * numPartitions).
  for (int i = 0; i < rows; ++i) {
    for (int p = 0; p < numPartitions; ++p) {
      // Each group starts here. The group size is the same as the column size.
      for (int j = 0; j < cols; ++j) {
        const int index = i * cols + j;
        const int64_t replicaId = values[index];
        const int64_t value =
            (replicaId == -1) ? -1 : replicaId * numPartitions + p;
        newValues.push_back(builder.getI64IntegerAttr(value));
      }
    }
  }

  auto type =
      RankedTensorType::get({rows * numPartitions, cols}, builder.getI64Type());
  return DenseIntElementsAttr::get(type, newValues);
}

static DenseIntElementsAttr convertToRankGroupsByCrossReplicaAndPartition(
    DenseIntElementsAttr replicaGroups, int32_t numPartitions,
    OpBuilder &builder) {
  if (numPartitions < 1) {
    // Treat as a single partition.
    return replicaGroups;
  }

  auto groupsType = replicaGroups.getType().cast<RankedTensorType>();
  assert(groupsType.getRank() == 2);
  int rows = groupsType.getShape()[0];
  int cols = groupsType.getShape()[1];
  auto values = replicaGroups.getValues<int64_t>();
  SmallVector<Attribute> newValues;

  // The number of groups is the same as the number of rows.
  for (int i = 0; i < rows; ++i) {
    // Each group starts here. The group size is (numPartitions * cols).
    for (int p = 0; p < numPartitions; ++p) {
      for (int j = 0; j < cols; ++j) {
        const int index = i * cols + j;
        const int64_t replicaId = values[index];
        const int64_t value =
            (replicaId == -1) ? -1 : replicaId * numPartitions + p;
        newValues.push_back(builder.getI64IntegerAttr(value));
      }
    }
  }
  auto type =
      RankedTensorType::get({rows, numPartitions * cols}, builder.getI64Type());
  return DenseIntElementsAttr::get(type, newValues);
}

/// Creates a channel matching the given |channelHandleAttr| scoped to the
/// requested group.
static Value createChannelWithGroupInfo(
    Location loc, mlir::stablehlo::ChannelHandleAttr channelHandleAttr,
    int32_t numReplicas, int32_t numPartitions,
    DenseIntElementsAttr replicaGroups, bool useGlobalDeviceIds,
    OpBuilder &builder) {
  // Set numPartitions to 1 if not set by the user.
  if (numPartitions == -1) numPartitions = 1;

  // Base channel that may be split by the group info.
  Value baseChannel =
      builder.create<IREE::Flow::ChannelDefaultOp>(loc, /*group=*/StringAttr{});

  // No need to split if there is a single group.
  ShapedType replicaGroupType = replicaGroups.getType();
  assert(replicaGroupType.getRank() == 2);
  if (numPartitions == 1 && replicaGroupType.getDimSize(0) == 1) {
    return baseChannel;
  }

  // Convert replica_groups into flattened IDs.
  DenseIntElementsAttr rankGroups;
  int64_t channelId = channelHandleAttr ? channelHandleAttr.getHandle() : 0;
  if (channelId <= 0) {
    assert(!useGlobalDeviceIds);
    rankGroups = convertToRankGroupsByCrossReplica(replicaGroups, numPartitions,
                                                   builder);
  } else {
    if (useGlobalDeviceIds) {
      // already flattened.
      rankGroups = replicaGroups;
    } else {
      rankGroups = convertToRankGroupsByCrossReplicaAndPartition(
          replicaGroups, numPartitions, builder);
    }
  }

  // Construct lookups for color and key split parameters.
  // Note that `replica_groups` can be interpreted in multiple ways based on the
  // other attributes.
  auto [color, key] =
      makeSplitColorAndKey(loc, baseChannel, rankGroups, builder);

  // Split the channel. Note that this is an expensive operation.
  return builder.create<IREE::Flow::ChannelSplitOp>(loc, baseChannel, color,
                                                    key);
}

static int32_t getNumReplicas(ModuleOp moduleOp) {
  if (!moduleOp) {
    return -1;
  }
  if (auto numReplicasAttr =
          moduleOp->getAttrOfType<IntegerAttr>("mhlo.num_replicas")) {
    return numReplicasAttr.getInt();
  } else {
    return -1;
  }
}

static int32_t getNumPartitions(ModuleOp moduleOp) {
  if (!moduleOp) {
    return -1;
  }
  if (auto numPartitionsAttr =
          moduleOp->getAttrOfType<IntegerAttr>("mhlo.num_partitions")) {
    return numPartitionsAttr.getInt();
  } else {
    return -1;
  }
}

static Value emitTranspose(ConversionPatternRewriter &rewriter, Location loc,
                           Value input, int64_t srcDim, int64_t dstDim) {
  // Creates a transpose op that swaps dimensions srcDim and dstDim in the
  // input.
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> inputShape(inputType.getShape());
  SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(0, inputShape.size()));
  std::swap(permutation[srcDim], permutation[dstDim]);
  std::swap(inputShape[srcDim], inputShape[dstDim]);
  DenseIntElementsAttr permutationAttr = rewriter.getI64VectorAttr(permutation);
  return rewriter.create<mlir::stablehlo::TransposeOp>(
      loc, RankedTensorType::get(inputShape, inputType.getElementType()), input,
      permutationAttr);
}

/// Converts stablehlo.partition_id to (flow.channel.rank % numPartitions)
struct PartitionIdOpConversion
    : public OpConversionPattern<mlir::stablehlo::PartitionIdOp> {
  using OpConversionPattern<
      mlir::stablehlo::PartitionIdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::PartitionIdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // PartitionId = rank % numPartitions
    auto moduleOp = op->getParentOfType<ModuleOp>();
    int32_t numPartitions = getNumPartitions(moduleOp);
    Value value;
    if (numPartitions <= 1) {
      value = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    } else {
      auto channel = rewriter.create<IREE::Flow::ChannelDefaultOp>(
          loc, /*group=*/StringAttr{});
      Value rank = rewriter.create<IREE::Flow::ChannelRankOp>(loc, channel);
      auto cst =
          rewriter.create<arith::ConstantIndexOp>(loc,
                                                  /*value=*/numPartitions);
      value = rewriter.create<arith::RemUIOp>(loc, rank, cst);
    }
    auto resultType = op.getType().cast<RankedTensorType>();  // tensor<ui32>
    auto elemType = resultType.getElementType();
    // index -> ui32
    auto rankElem = rewriter.create<arith::IndexCastUIOp>(loc, elemType, value);
    // tensor<ui32>
    auto rankTensor = rewriter.create<tensor::FromElementsOp>(
        loc, resultType, rankElem.getResult());
    rewriter.replaceOp(op, rankTensor.getResult());
    return success();
  }
};

/// Converts stablehlo.replica_id to floor_div(flow.channel.rank, numPartitions)
struct ReplicaIdOpConversion
    : public OpConversionPattern<mlir::stablehlo::ReplicaIdOp> {
  using OpConversionPattern<mlir::stablehlo::ReplicaIdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::ReplicaIdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto channel = rewriter.create<IREE::Flow::ChannelDefaultOp>(
        loc, /*group=*/StringAttr{});
    Value rank = rewriter.create<IREE::Flow::ChannelRankOp>(loc, channel);

    // ReplicaId = floor_div(rank, numPartitions)
    auto moduleOp = op->getParentOfType<ModuleOp>();
    int32_t numPartitions = getNumPartitions(moduleOp);
    auto cst = rewriter.create<arith::ConstantIndexOp>(loc,
                                                       /*value=*/numPartitions);
    if (numPartitions > 1) {
      rank = rewriter.create<arith::DivUIOp>(loc, rank, cst);
    }

    auto resultType = op.getType().cast<RankedTensorType>();  // tensor<ui32>
    auto elemType = resultType.getElementType();
    // index -> ui32
    auto rankElem = rewriter.create<arith::IndexCastUIOp>(loc, elemType, rank);
    // tensor<ui32>
    auto rankTensor = rewriter.create<tensor::FromElementsOp>(
        loc, resultType, rankElem.getResult());
    rewriter.replaceOp(op, rankTensor.getResult());
    return success();
  }
};

struct AllGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::AllGatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::AllGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (checkCollectiveAttrs(op, rewriter).failed()) {
      return failure();
    }

    Location loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    int32_t numReplicas = getNumReplicas(moduleOp);
    int32_t numPartitions = getNumPartitions(moduleOp);

    // Create a channel.
    Value channel = createChannelWithGroupInfo(
        loc, op.getChannelHandleAttr(), numReplicas, numPartitions,
        op.getReplicaGroups(), op.getUseGlobalDeviceIds(), rewriter);

    // Get the collective element type attribute.
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    IREE::Flow::CollectiveElementTypeAttr elementTypeAttr =
        getCollectiveElementTypeAttr(op.getContext(), resultType);
    if (!elementTypeAttr) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for collective op");
    }
    uint64_t allGatherDim = op.getAllGatherDim();
    Value gatherInput = op.getOperand();
    SmallVector<int64_t> gatherResultShape(resultType.getShape());

    // When all_gather_dim != 0, we need to transpose between 0 and
    // all_gather_dim before and after the flow all_gather op.
    bool requiresTranspose = allGatherDim != 0;
    if (requiresTranspose) {
      std::swap(gatherResultShape[0], gatherResultShape[allGatherDim]);
      gatherInput = emitTranspose(rewriter, loc, gatherInput, 0, allGatherDim);
    }

    // Create an empty tensor for the result.
    Value target = rewriter.create<tensor::EmptyOp>(
        loc, gatherResultShape, resultType.getElementType());
    Value gatherResult = rewriter.create<IREE::Flow::CollectiveAllGatherOp>(
        op.getLoc(), elementTypeAttr, target, gatherInput, channel);

    if (requiresTranspose) {
      gatherResult =
          emitTranspose(rewriter, loc, gatherResult, allGatherDim, 0);
    }

    rewriter.replaceOp(op, gatherResult);
    return success();
  }
};

struct AllReduceOpConversion final
    : OpConversionPattern<mlir::stablehlo::AllReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::AllReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (checkCollectiveAttrs(op, rewriter).failed()) {
      return failure();
    }

    // Only single elementwise op is supported.
    Block &block = op.getComputation().front();

    if (block.empty() || llvm::hasSingleElement(block) ||
        std::next(block.begin(), 2) != block.end()) {
      return rewriter.notifyMatchFailure(op, "must have two ops in the block");
    }

    if (block.getNumArguments() != 2) {
      return rewriter.notifyMatchFailure(op, "must have two block args");
    }

    Operation &op1 = block.front();
    Operation &op2 = *(++block.begin());

    if (op1.getNumResults() != 1 ||
        !op1.hasTrait<::mlir::OpTrait::Elementwise>()) {
      return rewriter.notifyMatchFailure(op, "must have elementwise trait");
    }

    // Convert stablehlo reduction op into flow reduction op.
    std::optional<IREE::Flow::CollectiveReductionOp> redOp =
        convertToFlowCollectiveReductionOp(op1);
    if (!redOp) {
      return rewriter.notifyMatchFailure(op, "unsupported operation.");
    }

    if (!op2.mightHaveTrait<OpTrait::IsTerminator>()) {
      return rewriter.notifyMatchFailure(op,
                                         "the second op must be a terminator");
    }

    Location loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    int32_t numReplicas = getNumReplicas(moduleOp);
    int32_t numPartitions = getNumPartitions(moduleOp);

    // Create a channel.
    Value channel = createChannelWithGroupInfo(
        loc, op.getChannelHandleAttr(), numReplicas, numPartitions,
        op.getReplicaGroups(), op.getUseGlobalDeviceIds(), rewriter);

    // Convert stablehlo reduction op into flow reduction op.
    auto reductionOpAttr =
        IREE::Flow::CollectiveReductionOpAttr::get(op.getContext(), *redOp);

    auto inputType = cast<RankedTensorType>(op.getOperand().getType());

    // Get the collective element type attribute.
    IREE::Flow::CollectiveElementTypeAttr elementTypeAttr =
        getCollectiveElementTypeAttr(op.getContext(), inputType);
    if (!elementTypeAttr) {
      return rewriter.notifyMatchFailure(op, "unsupported input type");
    }

    // Create an empty tensor for the result.
    ArrayRef<int64_t> inputShape = inputType.getShape();
    Value target = rewriter.create<tensor::EmptyOp>(loc, inputShape,
                                                    inputType.getElementType());
    auto allReduceOp = rewriter.create<IREE::Flow::CollectiveAllReduceOp>(
        op.getLoc(), reductionOpAttr, elementTypeAttr, target, op.getOperand(),
        channel);
    rewriter.replaceOp(op, allReduceOp.getResult());
    return success();
  }
};

Value splitAndConcatForAllToAll(ConversionPatternRewriter &rewriter,
                                Location loc, Value input, uint64_t splitDim,
                                uint64_t concatDim, uint64_t splitCount) {
  // Helper function to rearrange data after all-to-all.
  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Reshape
  const int64_t rank = inputShape.size();
  llvm::SmallVector<int64_t> newShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != splitDim) {
      newShape.push_back(inputShape[i]);
      continue;
    }
    newShape.push_back(splitCount);
    newShape.push_back(inputShape[i] / splitCount);
  }
  Value result = rewriter.create<mlir::stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(newShape, inputType.getElementType()), input);

  // Transpose
  SmallVector<int64_t> permutation;
  permutation.reserve(rank + 1);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dimAfterReshape = i >= splitDim ? i + 1 : i;
    if (i == concatDim) {
      permutation.push_back(splitDim);
    }
    permutation.push_back(dimAfterReshape);
  }
  SmallVector<int64_t> transposeResultShape;
  transposeResultShape.reserve(rank + 1);
  for (int64_t i = 0; i < rank + 1; ++i) {
    transposeResultShape.push_back(newShape[permutation[i]]);
  }

  result = rewriter.create<mlir::stablehlo::TransposeOp>(
      loc,
      RankedTensorType::get(transposeResultShape, inputType.getElementType()),
      result, rewriter.getI64VectorAttr(permutation));

  // Reshape
  llvm::SmallVector<int64_t> finalShape(inputShape);
  finalShape[concatDim] *= splitCount;
  finalShape[splitDim] /= splitCount;
  return rewriter.create<mlir::stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(finalShape, inputType.getElementType()),
      result);
}

struct AllToAllOpConversion final
    : OpConversionPattern<mlir::stablehlo::AllToAllOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::AllToAllOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the channel used for communication.
    // TODO: update to use createChannelWithGroupInfo.
    Value channel = rewriter.create<IREE::Flow::ChannelDefaultOp>(
        loc, /*group=*/StringAttr{});

    // Get the collective element type attribute.
    auto resultType = cast<RankedTensorType>(op.getType());
    IREE::Flow::CollectiveElementTypeAttr elementTypeAttr =
        getCollectiveElementTypeAttr(op.getContext(), resultType);
    if (!elementTypeAttr) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for collective op");
    }

    uint64_t splitDim = op.getSplitDimension();
    uint64_t concatDim = op.getConcatDimension();
    uint64_t splitCount = op.getSplitCount();
    Value allToAllInput = op.getOperand();

    // When splitDim != 0, we need to transpose splitDim to 0 before and after
    // the all-to-all.
    bool requiresTranspose = splitDim != 0;
    // When the concatDim != splitDim, we need to rearrange the data after the
    // all-to-all.
    bool requiresSplitAndConcat = concatDim != splitDim;
    if (requiresTranspose) {
      allToAllInput = emitTranspose(rewriter, loc, allToAllInput, 0, splitDim);
    }

    // Create an empty tensor for the result.
    Value target = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(allToAllInput.getType()).getShape(),
        resultType.getElementType());
    // Create all-to-all.
    Value allToAllResult = rewriter.create<IREE::Flow::CollectiveAllToAllOp>(
        op.getLoc(), elementTypeAttr, target, allToAllInput, channel);

    if (requiresTranspose) {
      allToAllResult =
          emitTranspose(rewriter, loc, allToAllResult, splitDim, 0);
    }
    if (requiresSplitAndConcat) {
      allToAllResult = splitAndConcatForAllToAll(
          rewriter, loc, allToAllResult, splitDim, concatDim, splitCount);
    }

    rewriter.replaceOp(op, allToAllResult);
    return success();
  }
};

struct ReduceScatterOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (checkCollectiveAttrs(op, rewriter).failed()) {
      return failure();
    }

    // Only single elementwise op is supported.
    Block &block = op.getComputation().front();

    if (block.empty() || llvm::hasSingleElement(block) ||
        std::next(block.begin(), 2) != block.end()) {
      return rewriter.notifyMatchFailure(op, "must have two ops in the block");
    }

    if (block.getNumArguments() != 2) {
      return rewriter.notifyMatchFailure(op, "must have two block args");
    }

    Operation &op1 = block.front();
    Operation &op2 = *(++block.begin());

    if (op1.getNumResults() != 1 ||
        !op1.hasTrait<::mlir::OpTrait::Elementwise>()) {
      return rewriter.notifyMatchFailure(op, "must have elementwise trait");
    }

    // Convert stablehlo reduction op into flow reduction op.
    std::optional<IREE::Flow::CollectiveReductionOp> redOp =
        convertToFlowCollectiveReductionOp(op1);
    if (!redOp) {
      return rewriter.notifyMatchFailure(op, "unsupported operation.");
    }

    if (!op2.mightHaveTrait<OpTrait::IsTerminator>()) {
      return rewriter.notifyMatchFailure(op,
                                         "the second op must be a terminator");
    }

    // Convert stablehlo reduction op into flow reduction op.
    auto reductionOpAttr =
        IREE::Flow::CollectiveReductionOpAttr::get(op.getContext(), *redOp);

    Location loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    int32_t numReplicas = getNumReplicas(moduleOp);
    int32_t numPartitions = getNumPartitions(moduleOp);

    // Create a channel.
    Value channel = createChannelWithGroupInfo(
        loc, op.getChannelHandleAttr(), numReplicas, numPartitions,
        op.getReplicaGroups(), op.getUseGlobalDeviceIds(), rewriter);

    // Get the collective element type attribute.
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    IREE::Flow::CollectiveElementTypeAttr elementTypeAttr =
        getCollectiveElementTypeAttr(op.getContext(), resultType);
    if (!elementTypeAttr) {
      return rewriter.notifyMatchFailure(op, "unsupported input type");
    }

    // When scatter_dimension != 0, we need to transpose between 0 and
    // scatter_dimension before and after the flow reduce_scatter op.
    uint64_t scatterDim = op.getScatterDimension();
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    SmallVector<int64_t> reduceInputShape(inputType.getShape());
    Value reduceInput = op.getOperand();
    DenseIntElementsAttr permutationAttr;

    SmallVector<int64_t> scatterResultShape(resultType.getShape());
    auto elemType = resultType.getElementType();

    if (scatterDim != 0) {
      auto permutation =
          llvm::to_vector(llvm::seq<int64_t>(0, scatterResultShape.size()));
      std::swap(permutation[0], permutation[scatterDim]);
      permutationAttr = rewriter.getI64VectorAttr(permutation);
      std::swap(reduceInputShape[0], reduceInputShape[scatterDim]);
      std::swap(scatterResultShape[0], scatterResultShape[scatterDim]);
      // Transpose the input.
      reduceInput = rewriter.create<mlir::stablehlo::TransposeOp>(
          loc, RankedTensorType::get(reduceInputShape, elemType), reduceInput,
          permutationAttr);
    }

    // Create an empty tensor for the result.
    Value target = rewriter.create<tensor::EmptyOp>(
        loc, scatterResultShape, resultType.getElementType());
    Value scatterResult =
        rewriter.create<IREE::Flow::CollectiveReduceScatterOp>(
            op.getLoc(), reductionOpAttr, elementTypeAttr, target, reduceInput,
            channel);

    if (scatterDim != 0) {
      scatterResult = rewriter.create<mlir::stablehlo::TransposeOp>(
          loc, resultType, scatterResult, permutationAttr);
    }

    rewriter.replaceOp(op, scatterResult);
    return success();
  }
};

}  // namespace

void populateStableHloCollectivesConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {
  patterns->add<AllGatherOpConversion, AllReduceOpConversion,
                AllToAllOpConversion, PartitionIdOpConversion,
                ReduceScatterOpConversion, ReplicaIdOpConversion>(typeConverter,
                                                                  context);
}

}  // namespace mlir::iree_compiler::stablehlo
