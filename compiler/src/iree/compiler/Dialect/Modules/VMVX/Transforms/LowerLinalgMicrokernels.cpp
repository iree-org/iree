// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

namespace {

struct BufferDescriptor {
  // The base buffer, corresponding to a row-major, contiguous memory layout.
  Value baseBuffer;

  // Size/offset/strides of the buffer.
  Value offset;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;

  BufferDescriptor(Value baseBuffer, unsigned rank) : baseBuffer(baseBuffer) {
    sizes.resize(rank);
    strides.resize(rank);
  }

  unsigned getRank() { return strides.size(); }

  /// Returns whether the innermost stride is statically 1. Many kernels
  /// require this, so we provide the convenience here.
  bool isUnitInnerStride() {
    if (getRank() == 0) return true;
    APInt stride;
    if (!matchPattern(strides.back(), m_ConstantInt(&stride))) return false;
    return stride.getZExtValue() == 1;
  }

  /// Casts the memref to a memref<?x...> that is safe for linear access
  /// with element-based addressing.
  Value castToLinear(Location loc, OpBuilder &builder) {
    BaseMemRefType sourceType = baseBuffer.getType().cast<MemRefType>();
    if (sourceType.getRank() <= 1) return baseBuffer;

    // Insert the cast just after the original def to keep inner loops tidy.
    OpBuilder::InsertionGuard restoreIp(builder);
    Operation *def = baseBuffer.getDefiningOp();
    if (def) builder.setInsertionPointAfter(def);

    // Collapse to 1D.
    ReassociationIndices reassociation;
    reassociation.resize(sourceType.getRank());
    for (int i = 0; i < sourceType.getRank(); ++i) {
      reassociation[i] = i;
    }
    return builder.create<memref::CollapseShapeOp>(loc, baseBuffer,
                                                   reassociation);
  }
};

/// Computes a BufferDescriptor from a given |buffer| (expected to be of
/// MemRefType). If the buffer is already in an identity layout, then size,
/// strides, and offsets corresponding to that layout are returned. If not
/// an identity layout, then will resolve one layer of memref::SubViewOp (
/// run ComposeSubViews or equivalent to merge chaines of subviews before
/// doing this).
/// Returns a BufferDescriptor if parameters can be computed. Otherwise,
/// returns nothing.
/// TODO: All of this should be replaced with an upstream op to get
/// offset/strides/sizes.
Optional<BufferDescriptor> computeBufferDescriptor(Location loc, Value buffer,
                                                   OpBuilder &builder) {
  auto constant = [&](int64_t idxValue) -> Value {
    return builder.create<arith::ConstantIndexOp>(loc, idxValue);
  };
  auto fillSize = [&](BufferDescriptor &desc, Value v) {
    auto t = buffer.getType().cast<MemRefType>();
    for (int i = 0; i < t.getRank(); ++i) {
      if (t.isDynamicDim(i)) {
        desc.sizes[i] = builder.create<memref::DimOp>(loc, buffer, i);
      } else {
        desc.sizes[i] = constant(t.getDimSize(i));
      }
    }
  };

  // Non-identity: Resolve to a bufferization.to_memref. Bufferization will
  // insert these with a tensor argument and a symbolic/strided layout that
  // cannot be resolved when coming from a constant. This seems like a bug
  // (TODO: check into this). For now, we just rewrite it.
  if (auto toMemref = llvm::dyn_cast_or_null<bufferization::ToMemrefOp>(
          buffer.getDefiningOp())) {
    auto t = buffer.getType().cast<MemRefType>();
    if (!t.getLayout().isIdentity()) {
      OpBuilder::InsertionGuard restoreIp(builder);
      builder.setInsertionPoint(toMemref);
      buffer = builder.create<bufferization::ToMemrefOp>(
          toMemref.getLoc(),
          MemRefType::get(t.getShape(), t.getElementType(), t.getMemorySpace()),
          toMemref.getOperand());
    }
  }

  // If identity, construct an identity layout.
  auto bufferType = buffer.getType().cast<MemRefType>();
  if (bufferType.getLayout().isIdentity()) {
    int rank = bufferType.getRank();
    BufferDescriptor desc(buffer, rank);
    // Size.
    fillSize(desc, buffer);
    // Strides.
    desc.strides[rank - 1] = constant(1);
    for (int i = rank - 2; i >= 0; --i) {
      desc.strides[i] = builder.create<arith::MulIOp>(loc, desc.strides[i + 1],
                                                      desc.sizes[i + 1]);
    }
    // Offset.
    desc.offset = constant(0);
    return desc;
  }

  // Non-identity: Resolve to a subview op.
  auto subViewOp = buffer.getDefiningOp<memref::SubViewOp>();
  if (!subViewOp) return None;

  // Insert before the subview op we are working on since we know everything
  // dominates here.
  OpBuilder::InsertionGuard restoreIp(builder);
  builder.setInsertionPoint(subViewOp);

  // Recursively resolve the descriptor of the subview's base.
  auto sourceDesc =
      computeBufferDescriptor(loc, subViewOp.getSource(), builder);
  if (!sourceDesc) return None;

  // TODO: For the moment, don't deal with the rank reducing subview case.
  if (bufferType.getRank() != sourceDesc->getRank()) return None;

  // Compose the source descriptor by:
  //   1. For each source stride, multiply by the subview stride (these are
  //      really "stride multipliers", not strides).
  //   2. Discard the source size, using the destination.
  //   3. Add the offset, computing using the new strides.
  BufferDescriptor composedDesc(sourceDesc->baseBuffer, bufferType.getRank());
  fillSize(composedDesc, buffer);

  // Stride multipliers.
  for (int idx = 0; idx < composedDesc.getRank(); ++idx) {
    if (subViewOp.isDynamicStride(idx)) {
      // Dynamic stride multiplier.
      composedDesc.strides[idx] = builder.create<arith::MulIOp>(
          loc, sourceDesc->strides[idx], subViewOp.getDynamicStride(idx));
    } else {
      // Handle static strides, dealing with the 0/1 common cases without
      // generating math ops.
      int64_t staticStrideMultiplier = subViewOp.getStaticStride(idx);
      if (staticStrideMultiplier == 1) {
        composedDesc.strides[idx] = sourceDesc->strides[idx];
      } else if (staticStrideMultiplier == 0) {
        composedDesc.strides[idx] = constant(0);
      } else {
        Value strideMultiplier = constant(staticStrideMultiplier);
        composedDesc.strides[idx] = builder.create<arith::MulIOp>(
            loc, sourceDesc->strides[idx], strideMultiplier);
      }
    }
  }

  // Compute offset.
  composedDesc.offset = sourceDesc->offset;
  for (int idx = 0; idx < composedDesc.getRank(); ++idx) {
    Value logicalOffset;
    if (subViewOp.isDynamicOffset(idx)) {
      logicalOffset = subViewOp.getDynamicOffset(idx);
    } else {
      int64_t staticOffset = subViewOp.getStaticOffset(idx);
      if (staticOffset == 0) {
        // Can just omit since all terms are added and this will multiply
        // to 0.
        continue;
      }
      logicalOffset = constant(staticOffset);
    }
    Value physicalOffset = builder.create<arith::MulIOp>(
        loc, logicalOffset, composedDesc.strides[idx]);
    composedDesc.offset =
        builder.create<arith::AddIOp>(loc, composedDesc.offset, physicalOffset);
  }

  return composedDesc;
}

struct LinalgFillConversion : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;
  struct OpInfo {
    linalg::FillOp op;
    Value scalar;
    Value out;
    Optional<BufferDescriptor> outDesc;

    int64_t getRank() { return outDesc->getRank(); }

    OpInfo(linalg::FillOp op) : op(op) {
      scalar = op.inputs().front();
      out = op.outputs().front();
    }
  };

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    OpInfo info(op);
    info.outDesc = computeBufferDescriptor(loc, info.out, rewriter);
    if (!info.outDesc) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for out");
    }

    // Switch based on specialization.
    if (info.getRank() == 2 && info.outDesc->isUnitInnerStride()) {
      return handle2DTile(info, rewriter);
    }

    return rewriter.notifyMatchFailure(op, "unhandled fill variant");
  }

  LogicalResult handle2DTile(OpInfo &info, PatternRewriter &rewriter) const {
    auto loc = info.op.getLoc();

    Value m = info.outDesc->sizes[0];
    Value n = info.outDesc->sizes[1];
    Value stride = info.outDesc->strides[0];
    Value outBuffer = info.outDesc->castToLinear(loc, rewriter);

    rewriter.replaceOpWithNewOp<VMVX::Fill2DOp>(
        info.op, info.scalar, outBuffer, info.outDesc->offset, stride, m, n);
    return success();
  }
};

/// Decomposes a generic op for emission to support primitives.
struct GenericPipelineDecomposer {
  linalg::LinalgOp genericOp;
  SmallVector<BufferDescriptor> inputAndOutputBufferDescriptors;

  // Map each scalar at the edge (block arguments, yield values) to a buffer
  // description.
  class BufferInfo {
   public:
    unsigned operandIndex;
    Value scalarValue;
    AffineMap indexingMap;

    // If this is an output buffer that directly yields from an input, then
    // the corresponding input will be noted in this field.
    BufferInfo *directlyYieldsInput = nullptr;

    BufferInfo(unsigned operandIndex, Value scalarValue, AffineMap indexingMap)
        : operandIndex(operandIndex),
          scalarValue(scalarValue),
          indexingMap(indexingMap) {}

    bool isProjectedPermutation() {
      return indexingMap.isProjectedPermutation();
    }
  };
  DenseMap<Value, BufferInfo> inputValueToBuffer;
  DenseMap<Value, BufferInfo> outputValueToBuffer;

  GenericPipelineDecomposer(linalg::LinalgOp genericOp) : genericOp(genericOp) {
    // Map scalar arguments within the body to input buffers.
    auto indexingMaps = genericOp.getIndexingMaps();
    Block *block = genericOp.getBlock();
    auto inputCount = genericOp.getNumInputs();
    for (auto it : llvm::enumerate(block->getArguments())) {
      Value scalarValue = it.value();
      // Inputs are first in the indexing maps, starting from 0.
      AffineMap indexingMap = indexingMaps[it.index()];
      inputValueToBuffer.insert(std::make_pair(
          scalarValue, BufferInfo(it.index(), scalarValue, indexingMap)));
    }

    // And map terminator (yield) arguments to out buffers.
    Operation *terminator = genericOp.getBlock()->getTerminator();
    for (auto it : llvm::enumerate(terminator->getOperands())) {
      Value scalarValue = it.value();
      AffineMap indexingMap = indexingMaps[inputCount + it.index()];
      auto insertIt = outputValueToBuffer.insert(std::make_pair(
          scalarValue,
          BufferInfo(inputCount + it.index(), scalarValue, indexingMap)));
      // If directly yielding an input, just note that.
      auto foundIt = inputValueToBuffer.find(scalarValue);
      if (foundIt != inputValueToBuffer.end()) {
        insertIt.first->second.directlyYieldsInput = &foundIt->second;
      }
    }
  }

  Location getLoc() { return genericOp.getLoc(); }

  // Gets a descriptor associated with a buffer. Can only be called after
  // beginTransformation.
  BufferDescriptor &getDescriptor(BufferInfo &buffer) {
    return inputAndOutputBufferDescriptors[buffer.operandIndex];
  }

  LogicalResult beginTransformation(PatternRewriter &rewriter) {
    // Compute buffer descriptors for inputs/outputs.
    for (auto it : llvm::enumerate(genericOp.getInputAndOutputOperands())) {
      auto desc =
          computeBufferDescriptor(getLoc(), it.value()->get(), rewriter);
      if (!desc) {
        return rewriter.notifyMatchFailure(genericOp, [&](Diagnostic &d) {
          d << "could not compute buffer descriptor for operand #"
            << it.index();
        });
      }
      inputAndOutputBufferDescriptors.push_back(std::move(*desc));
    }
    return success();
  }

  void emitCopy(BufferInfo outputBuffer, BufferInfo inputBuffer,
                PatternRewriter &rewriter) {
    auto &outputDesc = getDescriptor(outputBuffer);
    auto &inputDesc = getDescriptor(inputBuffer);
    AffineMap inputMap = inputBuffer.indexingMap;
    AffineMap outputMap = outputBuffer.indexingMap;
    bool isProjectedPermutation =
        inputMap.isProjectedPermutation() && outputMap.isProjectedPermutation();
    unsigned rank = outputMap.getNumDims();

    // Handle cases.
    if (isProjectedPermutation && rank > 0 && rank <= 2) {
      bool isTransposed = false;
      if (rank == 2 && inputMap.isPermutation()) {
        // Determine whether the map is transposed.
        if (inputMap.getPermutedPosition(0) == 1 &&
            inputMap.getPermutedPosition(1) == 0) {
          isTransposed = true;
        }
      }

      auto inpStrides = inputDesc.strides;
      auto outStrides = outputDesc.strides;
      auto outSize = outputDesc.sizes;
      while (outStrides.size() < 2) {
        // Left pad the input strides with zero (broadcasts leading dims).
        outStrides.insert(outStrides.begin(),
                          rewriter.create<arith::ConstantIndexOp>(getLoc(), 0));
      }
      while (inpStrides.size() < outStrides.size()) {
        // Left pad the input strides with zero (broadcasts leading dims).
        inpStrides.insert(inpStrides.begin(),
                          rewriter.create<arith::ConstantIndexOp>(getLoc(), 0));
      }
      if (isTransposed) {
        std::swap(inpStrides[0], inpStrides[1]);
      }

      auto inpBuffer = inputDesc.castToLinear(getLoc(), rewriter);
      auto outBuffer = outputDesc.castToLinear(getLoc(), rewriter);
      rewriter.create<VMVX::CopyOp>(getLoc(),
                                    // INP
                                    inpBuffer, inputDesc.offset, inpStrides,
                                    // OUT
                                    outBuffer, outputDesc.offset, outStrides,
                                    // Dims
                                    outSize);
      return;
    }

    // TODO: Fallback create new generic.
  }
};

struct LinalgGenericEmitter
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("vmvxIgnore")) {
      return failure();
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
      return rewriter.notifyMatchFailure(op, "not elementwise");
    }

    // TODO: Just trying to match a yield op now.
    if (op.getBlock()->getOperations().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "TODO: Support fully general unrolling");
    }

    GenericPipelineDecomposer gpd(op);
    if (failed(gpd.beginTransformation(rewriter))) {
      rewriter.startRootUpdate(op);
      op.getOperation()->setAttr("vmvxIgnore", rewriter.getUnitAttr());
      rewriter.finalizeRootUpdate(op);
      return success();
    }

    // TODO: Unroll normal body ops.

    // Finally, emit any direct yields as a copy.
    for (auto it : gpd.outputValueToBuffer) {
      auto &outputBufferInfo = it.second;
      if (outputBufferInfo.directlyYieldsInput) {
        gpd.emitCopy(outputBufferInfo, *outputBufferInfo.directlyYieldsInput,
                     rewriter);
      }
    }

    // Fully converted.
    rewriter.eraseOp(op);
    return success();
  }
};

struct LinalgGenericElementwiseUnroller
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  struct OpInfo {
    linalg::ContractionOpInterface contract;
    linalg::LinalgOp op;
    Block *body;
    Operation *terminator;
    SmallVector<BufferDescriptor> inputAndOutputBuffers;
    // linalg::OpOperandVector inputAndOutputOperands;

    // Mapping of scalar (internal to the Linalg op) SSA values and
    // corresponding output buffers. These are assembled in reverse order while
    // unrolling the op up from the terminator.
    BlockAndValueMapping scalarToBufferMapping;

    OpInfo(linalg::LinalgOp op) : op(op) {
      body = op.getBlock();
      terminator = body->getTerminator();
    }

    BufferDescriptor &getInput(int idx) { return inputAndOutputBuffers[idx]; }

    BufferDescriptor &getOutput(int idx) {
      return inputAndOutputBuffers[op.getNumInputs() + idx];
    }
  };

  struct InputBuffer {
    AffineMap inputMap;
    BufferDescriptor &desc;
  };

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OpInfo info(op);
    if (op.getNumParallelLoops() != op.getNumLoops()) {
      return rewriter.notifyMatchFailure(op, "not elementwise");
    }

    if (info.body->getOperations().size() != 2 ||
        (!llvm::isa<arith::AddFOp>(info.body->front()))) {
      // TODO: Support full generality before landing. Right now restricting to
      // a single inner op (plus yield).
      return rewriter.notifyMatchFailure(
          op, "TODO: Support fully general unrolling");
    }

    // Compute buffer descriptors for inputs/outputs.
    for (auto it : llvm::enumerate(op.getInputAndOutputOperands())) {
      auto desc = computeBufferDescriptor(loc, it.value()->get(), rewriter);
      if (!desc) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &d) {
          d << "could not compute buffer descriptor for operand #"
            << it.index();
        });
      }
      info.inputAndOutputBuffers.push_back(std::move(*desc));
    }

    // Match every SSA value from the terminator to the corresponding output of
    // the op. This equates to:
    //   linalg.generic ... outs(%1, %2) {
    //     %3 = ...
    //     %4 = ...
    //     linalg.yield %3, %4
    //   }
    auto indexingMaps = info.op.getIndexingMaps();
    for (auto it : llvm::enumerate(info.terminator->getOperands())) {
      int outputIndex = it.index();
      Value scalar = it.value();
      BufferDescriptor &outputBuffer = info.getOutput(it.index());
      AffineMap outputMap = indexingMaps[info.op.getNumInputs() + outputIndex];
      // TODO: Special case if yielding directly from a BlockArgument, as that
      // implies a copy.
      unrollOutputBufferProducers(info, scalar, outputBuffer, outputMap,
                                  rewriter);
    }

    rewriter.eraseOp(op);
    return success();
  }

  void unrollOutputBufferProducers(OpInfo &info, Value outputScalar,
                                   BufferDescriptor &outputBuffer,
                                   AffineMap outputMap,
                                   PatternRewriter &rewriter) const {
    llvm::SmallPtrSet<Value, 4> neededProducers;
    neededProducers.insert(outputScalar);

    // First iterate from bottom to top in order to identify which scalar ops
    // contribute to the output.
    for (Operation *scalarOp = info.terminator->getPrevNode(); scalarOp;
         scalarOp = scalarOp->getPrevNode()) {
      if (llvm::any_of(scalarOp->getResults(),
                       [&](Value v) { return neededProducers.contains(v); })) {
        // Add all operands to the needed set.
        for (auto operand : scalarOp->getOperands()) {
          neededProducers.insert(operand);
        }
      }
    }

    // Associate the top-level scalar inputs to buffers and indexing map.
    auto indexingMaps = info.op.getIndexingMaps();
    llvm::DenseMap<Value, InputBuffer> scalarValueToInputBuffer;
    for (auto it : llvm::enumerate(info.body->getArguments())) {
      AffineMap inputMap = indexingMaps[it.index()];
      scalarValueToInputBuffer.insert(std::make_pair(
          it.value(),
          InputBuffer{inputMap, info.inputAndOutputBuffers[it.index()]}));
    }

    // Now iterate from top to bottom, processing each operation that
    // contributes to the output. Because we proceed from the inputs, which
    // are often going to involve reading/computing less (i.e. they are the
    // input to broadcasting), this gives us some extra flexibility to
    // use intermediates that are sized to the correspondingly smaller
    // number of elements. If we proceeded bottom-up, we would always be
    // materializing output sized buffers. This optimization is not always
    // possible, but it can only ever be done if iterating in this order.
    for (Operation &scalarOp : info.body->getOperations()) {
      if (!llvm::any_of(scalarOp.getResults(),
                        [&](Value v) { return neededProducers.contains(v); })) {
        // Not part of this output.
        continue;
      }

      // For each result, get an output buffer to write into.
      // TODO: For now we only match this against the actual overall output
      // buffers, but we likely need to allocate temporaries.
      SmallVector<InputBuffer> operandInputBuffers;
      SmallVector<BufferDescriptor> resultOutputBuffers;
      auto allocateOutputBuffer = [&](Value result) -> BufferDescriptor {
        if (result == outputScalar) {
          return outputBuffer;
        }
        assert(false && "oops. don't support intermediates yet");
        return outputBuffer;
      };
      for (Value result : scalarOp.getResults()) {
        resultOutputBuffers.push_back(allocateOutputBuffer(result));
        BufferDescriptor &resultOutputBuffer = resultOutputBuffers.back();

        // Associate this as an input buffer for a later stage.
        // Note that here we make the assumption that we eagerly promote to the
        // output buffer indexing map.
        scalarValueToInputBuffer.insert(
            std::make_pair(result, InputBuffer{outputMap, resultOutputBuffer}));
      }
      for (Value operand : scalarOp.getOperands()) {
        auto found_it = scalarValueToInputBuffer.find(operand);
        if (found_it == scalarValueToInputBuffer.end()) {
          // TODO: Something weird with the generic: punt to emitting this as a
          // standalone thing.
          assert(false && "missing mapped input buffer");
        }
        operandInputBuffers.push_back(found_it->second);
      }

      // Switch on the op and emit.
      if (llvm::isa<arith::AddFOp, arith::AddIOp>(scalarOp)) {
        unrollVMVXBinaryOp<VMVX::AddOp>(info, scalarOp, operandInputBuffers,
                                        resultOutputBuffers, outputMap,
                                        rewriter);
      } else {
        assert(false && "should emit unrecognized op as generic");
      }
    }
  }

  template <typename TargetOpTy>
  void unrollVMVXBinaryOp(OpInfo &info, Operation &srcOp,
                          SmallVector<InputBuffer> &operandInputBuffers,
                          SmallVector<BufferDescriptor> resultOutputBuffers,
                          AffineMap outputMap,
                          PatternRewriter &rewriter) const {
    auto loc = info.op.getLoc();
    Value oneSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto expandRankTo = [&](SmallVector<Value> &dims, unsigned rank,
                            Value fillIndex) {
      while (dims.size() < rank) {
        dims.insert(dims.begin(), fillIndex);
      }
    };

    // TODO: This is not right and just so I can go home on a Thursday night.
    // What we should be doing: affine.apply'ing the corresponding maps and
    // then rank expanding.
    auto lhsStrides = operandInputBuffers[0].desc.strides;
    auto rhsStrides = operandInputBuffers[1].desc.strides;
    auto outStrides = resultOutputBuffers[0].strides;
    auto outSize = resultOutputBuffers[0].sizes;
    unsigned commonRank = std::max(
        std::max(lhsStrides.size(), rhsStrides.size()), outStrides.size());
    expandRankTo(lhsStrides, commonRank, zeroOffset);
    expandRankTo(rhsStrides, commonRank, zeroOffset);
    expandRankTo(outStrides, commonRank, zeroOffset);
    expandRankTo(outSize, commonRank, oneSize);

    auto lhsBuffer = operandInputBuffers[0].desc.castToLinear(loc, rewriter);
    auto rhsBuffer = operandInputBuffers[1].desc.castToLinear(loc, rewriter);
    auto outBuffer = resultOutputBuffers[0].castToLinear(loc, rewriter);

    rewriter.create<TargetOpTy>(
        loc,
        // LHS.
        lhsBuffer, operandInputBuffers[0].desc.offset, lhsStrides,
        // RHS.
        rhsBuffer, operandInputBuffers[1].desc.offset, rhsStrides,
        // OUT.
        outBuffer, resultOutputBuffers[0].offset, outStrides,
        // Dims.
        outSize);
  }
};

struct LinalgMatmulConversion
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  struct OpInfo {
    linalg::ContractionOpInterface contract;
    linalg::LinalgOp op;

    Value lhs;
    Value rhs;
    Value out;

    Optional<BufferDescriptor> lhsDesc;
    Optional<BufferDescriptor> rhsDesc;
    Optional<BufferDescriptor> outDesc;

    OpInfo(linalg::ContractionOpInterface contract)
        : contract(contract),
          op(llvm::cast<linalg::LinalgOp>(contract.getOperation())) {
      lhs = contract.lhs();
      rhs = contract.rhs();
      out = op.outputs().front();
    }

    Value getOneValue(PatternRewriter &rewriter) {
      Location loc = op.getLoc();
      Type elementType = out.getType().cast<MemRefType>().getElementType();
      if (auto floatType = elementType.dyn_cast<FloatType>()) {
        return rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(floatType, 1.0));
      } else if (elementType.isa<IntegerType>()) {
        return rewriter.create<arith::ConstantIntOp>(loc, 1, elementType);
      }

      assert(false && "unknown element type");
      return nullptr;
    }
  };

  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    OpInfo info(op);

    // Check that buffer descriptors could be computed.
    info.lhsDesc = computeBufferDescriptor(loc, info.lhs, rewriter);
    if (!info.lhsDesc) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for lhs");
    }
    info.rhsDesc = computeBufferDescriptor(loc, info.rhs, rewriter);
    if (!info.rhsDesc) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for rhs");
    }
    info.outDesc = computeBufferDescriptor(loc, info.out, rewriter);
    if (!info.outDesc) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for out");
    }

    // Check for unit inner strides.
    if (!info.lhsDesc->isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "lhs has non-unit inner stride");
    }
    if (!info.rhsDesc->isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "rhs has non-unit inner stride");
    }
    if (!info.outDesc->isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "out has non-unit inner stride");
    }

    // Switch on contraction type.
    if (info.contract.isRowMajorMatmul() ||
        info.contract.isColumnMajorMatmul()) {
      return handleConformingMatmul2D(info, rewriter);
    }

    return rewriter.notifyMatchFailure(op, "unsupported contraction variant");
  }

  LogicalResult handleConformingMatmul2D(OpInfo &info,
                                         PatternRewriter &rewriter) const {
    auto loc = info.op.getLoc();
    // Determine m, n, k based on dims.
    int flags = 0;
    Value m, n, k;
    if (info.contract.isRowMajorMatmul()) {
      m = info.lhsDesc->sizes[0];
      k = info.rhsDesc->sizes[0];
      n = info.rhsDesc->sizes[1];
    } else if (info.contract.isColumnMajorMatmul()) {
      m = info.lhsDesc->sizes[0];
      k = info.rhsDesc->sizes[1];
      n = info.rhsDesc->sizes[0];
      // TODO: Flag constants somewhere.
      flags |= 1;
    } else {
      return failure();
    }

    // Alpha/beta: We always start the lowering with alpha/beta set to 1.
    // Simplification patterns within VMVX will simplify this if possible.
    Value alpha = info.getOneValue(rewriter);
    Value beta = alpha;

    auto lhsBuffer = info.lhsDesc->castToLinear(loc, rewriter);
    auto rhsBuffer = info.rhsDesc->castToLinear(loc, rewriter);
    auto outBuffer = info.outDesc->castToLinear(loc, rewriter);

    rewriter.replaceOpWithNewOp<VMVX::MatmulOp>(
        info.op,
        // LHS
        lhsBuffer, info.lhsDesc->offset, info.lhsDesc->strides[0],
        // RHS
        rhsBuffer, info.rhsDesc->offset, info.rhsDesc->strides[0],
        // Out
        outBuffer, info.outDesc->offset, info.outDesc->strides[0],
        // m,n,k
        m, n, k,
        // alpha, beta
        alpha, beta,
        // flags
        rewriter.getI32IntegerAttr(flags));
    return success();
  }
};

}  // namespace

class LowerLinalgMicrokernelsPass
    : public LowerLinalgMicrokernelsBase<LowerLinalgMicrokernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VMVX::VMVXDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<LinalgFillConversion, LinalgGenericElementwiseUnroller,
                    LinalgGenericEmitter, LinalgMatmulConversion>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLowerLinalgMicrokernelsPass() {
  return std::make_unique<LowerLinalgMicrokernelsPass>();
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
