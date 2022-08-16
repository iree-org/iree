// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Permutes raw strides against a projected permutation map returning a
// vector of strides that is the permutation with expansion positions
// set to zero.
SmallVector<Value> permuteStrides(Location loc, AffineMap indexingMap,
                                  SmallVectorImpl<Value> &rawStrides,
                                  OpBuilder &builder) {
  unsigned rank = indexingMap.getNumDims();
  assert(indexingMap.getNumResults() == rawStrides.size() &&
         "mismatched strides and indexing map");
  // Construct the permuted strides.
  SmallVector<Value> strides;
  strides.resize(rank);
  for (unsigned resultPos = 0; resultPos < indexingMap.getNumResults();
       ++resultPos) {
    unsigned inputPos = indexingMap.getDimPosition(resultPos);
    strides[inputPos] = rawStrides[resultPos];
  }
  // Fill any unset stride with 0.
  Value zero;
  for (Value &stride : strides) {
    if (!stride) {
      if (!zero) {
        zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      }
      stride = zero;
    }
  }
  return strides;
}

/// Left pads a vector of Values to a minimum rank, adding the given pad
/// value as needed.
void leftPadToRank(Location loc, SmallVectorImpl<Value> &indices,
                   unsigned minRank, unsigned padIndex, OpBuilder &builder) {
  Value padValue;
  while (indices.size() < minRank) {
    if (!padValue) {
      padValue = builder.create<arith::ConstantIndexOp>(loc, padIndex);
    }
    indices.insert(indices.begin(), padValue);
  }
}

bool isMemRefUnitInnerStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (type.getLayout().isIdentity()) {
    return true;
  }

  if (failed(mlir::getStridesAndOffset(type, strides, offset))) {
    return false;
  }
  return strides.back() == 1;
}

struct StridedBufferDescriptor {
  MemRefType memRefType;

  // Size/offset/strides of the buffer.
  Value offset;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;

  StridedBufferDescriptor() = default;

  unsigned getRank() { return strides.size(); }
  Type getElementType() { return memRefType.getElementType(); }
  TypeAttr getElementTypeAttr() { return TypeAttr::get(getElementType()); }

  /// Returns whether the innermost stride is statically 1. Many kernels
  /// require this, so we provide the convenience here.
  bool isUnitInnerStride() { return isMemRefUnitInnerStride(memRefType); }

  /// Casts the memref to a memref<?x...> that is safe for linear access
  /// with element-based addressing.
  Value castToLinear(Location loc, OpBuilder &builder) { return baseBuffer; }

 private:
  // The base !util.buffer
  Value baseBuffer;
  friend class StridedBufferAnalysis;
};

/// Holds the results of an analysis which indicates whether a given memref
/// can be decomposed into fully known static or dynamic base, strides, offset
/// and sizes. If this holds, then a StridedBufferDescriptor is guaranteed to
/// be producible (this is an IR mutation so must be separated).
///
/// Analyzing the buffer involves tracking up through the stack of composed
/// SubviewOps to some memref with an identity layout (i.e. not offsets/strides
/// applied).
class StridedBufferAnalysis {
 public:
  StridedBufferAnalysis(Value buffer) : buffer(buffer) {}

  // Whether analysis was successful.
  bool isValid() { return true; }

  // Gets the type of the buffer being analyzed.
  MemRefType getType() { return buffer.getType().cast<MemRefType>(); }

  // Gets the rank of the buffer being analyzed.
  unsigned getRank() { return getType().getRank(); }

  /// Returns whether the innermost stride is statically 1. Many kernels
  /// require this, so we provide the convenience here.
  bool isUnitInnerStride() { return isMemRefUnitInnerStride(getType()); }

  StridedBufferDescriptor &getDesc(OpBuilder &builder) {
    assert(isValid() && "invalid StridedBufferAnalysis");
    if (desc) return *desc;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(buffer);

    Location loc = buffer.getLoc();
    desc = StridedBufferDescriptor();
    desc->memRefType = buffer.getType().cast<MemRefType>();

    int rank = getType().getRank();
    SmallVector<Type> sizeStrideTypes;
    IndexType indexType = builder.getIndexType();
    for (int i = 0; i < rank; ++i) {
      sizeStrideTypes.push_back(indexType);
    }

    auto op = builder.create<IREE::VMVX::GetBufferDescriptorOp>(
        loc, builder.getType<IREE::Util::BufferType>(), builder.getIndexType(),
        sizeStrideTypes, sizeStrideTypes, buffer);

    desc->baseBuffer = op.getBaseBuffer();
    desc->offset = op.getOffset();
    desc->sizes = op.getSizes();
    desc->strides = op.getStrides();

    return *desc;
  }

 private:
  Value buffer;
  Optional<StridedBufferDescriptor> desc;
};

/// Emits a vmvx binary op.
struct BinaryEmitter {
  enum class OpType {
    // Emits a vmvx.binary op with a given opcode.
    GenericBinary,
  };
  struct OpSelection {
    OpType opType;
    // If the OpType takes an opcode, this is it.
    StringRef opcode;

    static OpSelection genericBinary(StringRef opcode) {
      return OpSelection{OpType::GenericBinary, opcode};
    }
  };
  struct Descriptor {
    Value buffer;
    AffineMap indexingMap;
    StridedBufferAnalysis bufferAnal;
    StridedBufferDescriptor *bufferDesc = nullptr;
    Descriptor(Value buffer, AffineMap indexingMap)
        : buffer(buffer), indexingMap(indexingMap), bufferAnal(buffer) {}
    unsigned getRank() { return indexingMap.getNumDims(); }
  };
  std::pair<Descriptor, Descriptor> operands;
  Descriptor result;
  OpSelection selection;

  BinaryEmitter(Descriptor operand0, Descriptor operand1, Descriptor result,
                OpSelection selection)
      : operands(std::make_pair(operand0, operand1)),
        result(result),
        selection(selection) {}

  bool isProjectedPermutation() {
    return operands.first.indexingMap.isProjectedPermutation() &&
           operands.second.indexingMap.isProjectedPermutation() &&
           result.indexingMap.isProjectedPermutation();
  }

  unsigned maxRank() {
    return std::max(operands.first.getRank(),
                    std::max(operands.second.getRank(), result.getRank()));
  }

  LogicalResult initialize(Location loc, PatternRewriter &rewriter) {
    if (!isProjectedPermutation())
      return rewriter.notifyMatchFailure(loc, "not projected permutation");
    if (maxRank() > 2) return rewriter.notifyMatchFailure(loc, "rank > 2");
    if (!operands.first.bufferAnal.isValid() ||
        !operands.second.bufferAnal.isValid() || !result.bufferAnal.isValid()) {
      return rewriter.notifyMatchFailure(loc,
                                         "could not compute buffer descriptor");
    }

    // All pre-conditions pass. Mutate IR.
    operands.first.bufferDesc = &operands.first.bufferAnal.getDesc(rewriter);
    operands.second.bufferDesc = &operands.second.bufferAnal.getDesc(rewriter);
    result.bufferDesc = &result.bufferAnal.getDesc(rewriter);
    return success();
  }

  void emit(Location loc, PatternRewriter &rewriter) {
    struct EmitParams {
      SmallVector<Value> in0Strides;
      SmallVector<Value> in1Strides;
      SmallVector<Value> outStrides;
      SmallVector<Value> sizes;
      Value in0Buffer;
      Value in1Buffer;
      Value outBuffer;
    };
    EmitParams params;
    params.in0Strides =
        permuteStrides(loc, operands.first.indexingMap,
                       operands.first.bufferDesc->strides, rewriter);
    params.in1Strides =
        permuteStrides(loc, operands.second.indexingMap,
                       operands.second.bufferDesc->strides, rewriter);
    params.outStrides = permuteStrides(loc, result.indexingMap,
                                       result.bufferDesc->strides, rewriter);
    params.sizes = result.bufferDesc->sizes;
    assert(params.outStrides.size() == result.bufferDesc->strides.size() &&
           "output projection mismatched strides");
    params.in0Buffer = operands.first.bufferDesc->castToLinear(loc, rewriter);
    params.in1Buffer = operands.second.bufferDesc->castToLinear(loc, rewriter);
    params.outBuffer = result.bufferDesc->castToLinear(loc, rewriter);

    // Binary ops support minimum of 2d indexing. Pad.
    leftPadToRank(loc, params.in0Strides, 2, 0, rewriter);
    leftPadToRank(loc, params.in1Strides, 2, 0, rewriter);
    leftPadToRank(loc, params.outStrides, 2, 0, rewriter);
    leftPadToRank(loc, params.sizes, 2, 1, rewriter);

    switch (selection.opType) {
      case OpType::GenericBinary: {
        rewriter.create<IREE::VMVX::BinaryOp>(
            loc, rewriter.getStringAttr(selection.opcode),
            // LHS
            params.in0Buffer, operands.first.bufferDesc->offset,
            params.in0Strides,
            // RHS
            params.in1Buffer, operands.second.bufferDesc->offset,
            params.in1Strides,
            // OUT
            params.outBuffer, result.bufferDesc->offset, params.outStrides,
            // Sizes
            params.sizes,
            // Attributes
            operands.first.bufferDesc->getElementTypeAttr());

        break;
      }
      default:
        assert(false && "unhandled OpType");
    }
  }
};

/// Emits a vmvx unary op.
struct UnaryEmitter {
  enum class OpType {
    // Emits a vmvx.unary op with a given opcode.
    GenericUnary,
  };
  struct OpSelection {
    OpType opType;
    // If the OpType takes an opcode, this is it.
    StringRef opcode;

    static OpSelection genericUnary(StringRef opcode) {
      return OpSelection{OpType::GenericUnary, opcode};
    }
  };
  struct Descriptor {
    Value buffer;
    AffineMap indexingMap;
    StridedBufferAnalysis bufferAnal;
    StridedBufferDescriptor *bufferDesc = nullptr;
    Descriptor(Value buffer, AffineMap indexingMap)
        : buffer(buffer), indexingMap(indexingMap), bufferAnal(buffer) {}
    unsigned getRank() { return indexingMap.getNumDims(); }
  };
  Descriptor operand;
  Descriptor result;
  OpSelection selection;

  UnaryEmitter(Descriptor operand, Descriptor result, OpSelection selection)
      : operand(operand), result(result), selection(selection) {}

  bool isProjectedPermutation() {
    return operand.indexingMap.isProjectedPermutation() &&
           result.indexingMap.isProjectedPermutation();
  }

  unsigned maxRank() { return std::max(operand.getRank(), result.getRank()); }

  LogicalResult initialize(Location loc, PatternRewriter &rewriter) {
    if (!isProjectedPermutation())
      return rewriter.notifyMatchFailure(loc, "not projected permutation");
    if (maxRank() > 2) return rewriter.notifyMatchFailure(loc, "rank > 2");
    if (!operand.bufferAnal.isValid() || !result.bufferAnal.isValid()) {
      return rewriter.notifyMatchFailure(loc,
                                         "could not compute buffer descriptor");
    }

    // All pre-conditions pass. Mutate IR.
    operand.bufferDesc = &operand.bufferAnal.getDesc(rewriter);
    result.bufferDesc = &result.bufferAnal.getDesc(rewriter);
    return success();
  }

  void emit(Location loc, PatternRewriter &rewriter) {
    struct EmitParams {
      SmallVector<Value> inStrides;
      SmallVector<Value> outStrides;
      SmallVector<Value> sizes;
      Value inBuffer;
      Value outBuffer;
    };
    EmitParams params;
    params.inStrides = permuteStrides(loc, operand.indexingMap,
                                      operand.bufferDesc->strides, rewriter);
    params.outStrides = permuteStrides(loc, result.indexingMap,
                                       result.bufferDesc->strides, rewriter);
    params.sizes = result.bufferDesc->sizes;
    assert(params.outStrides.size() == result.bufferDesc->strides.size() &&
           "output projection mismatched strides");
    params.inBuffer = operand.bufferDesc->castToLinear(loc, rewriter);
    params.outBuffer = result.bufferDesc->castToLinear(loc, rewriter);

    // Binary ops support minimum of 2d indexing. Pad.
    leftPadToRank(loc, params.inStrides, 2, 0, rewriter);
    leftPadToRank(loc, params.outStrides, 2, 0, rewriter);
    leftPadToRank(loc, params.sizes, 2, 1, rewriter);

    switch (selection.opType) {
      case OpType::GenericUnary: {
        rewriter.create<IREE::VMVX::UnaryOp>(
            loc, rewriter.getStringAttr(selection.opcode),
            // IN
            params.inBuffer, operand.bufferDesc->offset, params.inStrides,
            // OUT
            params.outBuffer, result.bufferDesc->offset, params.outStrides,
            // Sizes
            params.sizes,
            // Attributes
            operand.bufferDesc->getElementTypeAttr());

        break;
      }
      default:
        assert(false && "unhandled OpType");
    }
  }
};

/// Emits a vmvx.copy op from/to a buffer/indexingMap pair.
/// Only projected permutations are supported.
struct CopyEmitter {
  struct Descriptor {
    Value buffer;
    AffineMap indexingMap;
    StridedBufferAnalysis bufferAnal;
    StridedBufferDescriptor *bufferDesc = nullptr;
    Descriptor(Value buffer, AffineMap indexingMap)
        : buffer(buffer), indexingMap(indexingMap), bufferAnal(buffer) {}

    unsigned getRank() { return indexingMap.getNumDims(); }
  };
  using DescriptorPair = std::pair<Descriptor, Descriptor>;
  SmallVector<DescriptorPair, 1> copies;

  bool isProjectedPermutation() {
    return llvm::all_of(copies, [](DescriptorPair &copy) {
      return copy.first.indexingMap.isProjectedPermutation() &&
             copy.second.indexingMap.isProjectedPermutation();
    });
  }

  unsigned maxRank() {
    unsigned rank = 0;
    for (auto &copy : copies) {
      rank =
          std::max(rank, std::max(copy.first.getRank(), copy.second.getRank()));
    }
    return rank;
  }

  LogicalResult initialize(Location loc, PatternRewriter &rewriter) {
    if (!isProjectedPermutation())
      return rewriter.notifyMatchFailure(loc, "not projected permutation");
    if (maxRank() > 2) return rewriter.notifyMatchFailure(loc, "rank > 2");

    // Initialize buffer descriptors.
    for (auto &copy : copies) {
      if (!copy.first.bufferAnal.isValid() ||
          !copy.second.bufferAnal.isValid()) {
        return rewriter.notifyMatchFailure(
            loc, "could not compute buffer descriptor");
      }
    }

    // All pre-conditions pass. Mutate IR.
    for (auto &copy : copies) {
      copy.first.bufferDesc = &copy.first.bufferAnal.getDesc(rewriter);
      copy.second.bufferDesc = &copy.second.bufferAnal.getDesc(rewriter);
    }
    return success();
  }

  void emit(Location loc, PatternRewriter &rewriter) {
    for (auto &copy : copies) {
      emitCopy(loc, copy.first, copy.second, rewriter);
    }
  }

  void emitCopy(Location loc, Descriptor &in, Descriptor &out,
                PatternRewriter &rewriter) {
    SmallVector<Value> inStrides =
        permuteStrides(loc, in.indexingMap, in.bufferDesc->strides, rewriter);
    SmallVector<Value> outStrides =
        permuteStrides(loc, out.indexingMap, out.bufferDesc->strides, rewriter);
    SmallVector<Value> sizes = out.bufferDesc->sizes;
    assert(outStrides.size() == out.bufferDesc->strides.size() &&
           "output projection mismatched strides");
    auto inBuffer = in.bufferDesc->castToLinear(loc, rewriter);
    auto outBuffer = out.bufferDesc->castToLinear(loc, rewriter);

    // Copy only supports >= 2d at present. Pad.
    leftPadToRank(loc, inStrides, 2, 0, rewriter);
    leftPadToRank(loc, outStrides, 2, 0, rewriter);
    leftPadToRank(loc, sizes, 2, 1, rewriter);

    rewriter.create<IREE::VMVX::CopyOp>(
        loc,
        // IN
        inBuffer, in.bufferDesc->offset, inStrides,
        // OUT
        outBuffer, out.bufferDesc->offset, outStrides,
        // Sizes
        sizes,
        // Element type.
        in.bufferDesc->getElementTypeAttr());
  }
};

/// Matches a generic which contains an expressible binary operation, emitting
/// as a vmvx op.
struct LinalgBinaryGenericConversion
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto &children = op.getBlock()->getOperations();
    // Only match two children (op + yield).
    if (children.size() != 2) return failure();
    // Only match parallel loops.
    if (op.getNumParallelLoops() != op.getNumLoops()) return failure();

    // Match:
    //   %0 = someop %arg2, %arg3
    //   yield %0
    Operation *binaryOp = &children.front();
    Operation *yieldOp = op.getBlock()->getTerminator();
    if (binaryOp->getNumOperands() != 2 || yieldOp->getNumOperands() != 1 ||
        yieldOp->getOperand(0) != binaryOp->getResult(0)) {
      return failure();
    }
    BlockArgument operandScalar0 =
        binaryOp->getOperands()[0].dyn_cast<BlockArgument>();
    BlockArgument operandScalar1 =
        binaryOp->getOperands()[1].dyn_cast<BlockArgument>();
    if (!operandScalar0 || !operandScalar1) return failure();

    // Construct the emitter and start lowering.
    // Note that the operands may map to an out if the aliasing is safe,
    // so we use getOpOperand() vs restricting to just the generic ins.
    OpOperand *operand0 = &op->getOpOperand(operandScalar0.getArgNumber());
    OpOperand *operand1 = &op->getOpOperand(operandScalar1.getArgNumber());
    OpOperand *result = op.getOutputOperand(0);

    // Returns an emitter for a generic binary compatible operation where
    // |binaryOp| has a 1:1 correspondance with |opcode|.
    auto configureGenericBinary =
        [&](Operation *binaryOp, StringRef opcode) -> Optional<BinaryEmitter> {
      SmallVector<BinaryEmitter::Descriptor, 2> operands;
      // Make sure that the binary op has operands that map to the
      // ins and detect the order.
      auto selection = BinaryEmitter::OpSelection::genericBinary(opcode);
      if (binaryOp->getOperand(0) == operandScalar0 &&
          binaryOp->getOperand(1) == operandScalar1) {
        // 1:1 matching.
        return BinaryEmitter(
            BinaryEmitter::Descriptor(operand0->get(),
                                      op.getTiedIndexingMap(operand0)),
            BinaryEmitter::Descriptor(operand1->get(),
                                      op.getTiedIndexingMap(operand1)),
            BinaryEmitter::Descriptor(result->get(),
                                      op.getTiedIndexingMap(result)),
            selection);
      } else if (binaryOp->getOperand(1) == operandScalar0 &&
                 binaryOp->getOperand(0) == operandScalar1) {
        // Inverted operands.
        return BinaryEmitter(
            BinaryEmitter::Descriptor(operand1->get(),
                                      op.getTiedIndexingMap(operand1)),
            BinaryEmitter::Descriptor(operand0->get(),
                                      op.getTiedIndexingMap(operand0)),
            BinaryEmitter::Descriptor(result->get(),
                                      op.getTiedIndexingMap(result)),
            selection);
      } else {
        return None;
      }
    };

    // Select the op to lower to and configure the emitter.
    // Emit from the iree_ukernel_x32b_opcode_t table.
    Optional<BinaryEmitter> emitter =
        TypeSwitch<Operation *, Optional<BinaryEmitter>>(binaryOp)
            .Case([&](arith::AddFOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "add");
              }
              return None;
            })
            .Case([&](arith::AddIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "add");
              }
              return None;
            })
            .Case([&](arith::AndIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "and");
              }
              return None;
            })
            .Case([&](arith::DivFOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "div");
              }
              return None;
            })
            .Case([&](arith::DivSIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "divs");
              }
              return None;
            })
            .Case([&](arith::DivUIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "divu");
              }
              return None;
            })
            .Case([&](arith::MulFOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "mul");
              }
              return None;
            })
            .Case([&](arith::MulIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "mul");
              }
              return None;
            })
            .Case([&](arith::OrIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "or");
              }
              return None;
            })
            .Case([&](arith::ShLIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "shl");
              }
              return None;
            })
            .Case([&](arith::ShRSIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "shrs");
              }
              return None;
            })
            .Case([&](arith::XOrIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "xor");
              }
              return None;
            })
            .Case([&](arith::SubFOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "sub");
              }
              return None;
            })
            .Case([&](arith::SubIOp op) -> Optional<BinaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "sub");
              }
              return None;
            })
            .Default([](Operation *) { return None; });

    // Determine op type to lower to.
    if (!emitter) {
      return rewriter.notifyMatchFailure(op, "unrecognized binary op");
    }
    if (failed(emitter->initialize(op.getLoc(), rewriter))) return failure();

    emitter->emit(op.getLoc(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Matches a generic which contains an expressible unary operation, emitting
/// as a vmvx op.
struct LinalgUnaryGenericConversion
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto &children = op.getBlock()->getOperations();
    // Only match two children (op + yield).
    if (children.size() != 2) return failure();
    // Only match parallel loops.
    if (op.getNumParallelLoops() != op.getNumLoops()) return failure();

    // Match:
    //   %0 = someop %arg2
    //   yield %0
    Operation *unaryOp = &children.front();
    Operation *yieldOp = op.getBlock()->getTerminator();
    if (unaryOp->getNumOperands() != 1 || yieldOp->getNumOperands() != 1 ||
        yieldOp->getOperand(0) != unaryOp->getResult(0)) {
      return failure();
    }
    BlockArgument operandScalar0 =
        unaryOp->getOperands()[0].dyn_cast<BlockArgument>();
    if (!operandScalar0) return failure();

    // Construct the emitter and start lowering.
    // Note that the operands may map to an out if the aliasing is safe,
    // so we use getOpOperand() vs restricting to just the generic ins.
    OpOperand *operand0 = &op->getOpOperand(operandScalar0.getArgNumber());
    OpOperand *result = op.getOutputOperand(0);

    // Returns an emitter for a generic binary compatible operation where
    // |binaryOp| has a 1:1 correspondance with |opcode|.
    auto configureGenericUnary =
        [&](Operation *unaryOp, StringRef opcode) -> Optional<UnaryEmitter> {
      SmallVector<UnaryEmitter::Descriptor, 2> operands;
      // Make sure that the binary op has operands that map to the
      // ins and detect the order.
      auto selection = UnaryEmitter::OpSelection::genericUnary(opcode);
      return UnaryEmitter(UnaryEmitter::Descriptor(
                              operand0->get(), op.getTiedIndexingMap(operand0)),
                          UnaryEmitter::Descriptor(
                              result->get(), op.getTiedIndexingMap(result)),
                          selection);
    };

    // Select the op to lower to and configure the emitter.
    // Emit from the iree_ukernel_x32b_opcode_t table.
    Optional<UnaryEmitter> emitter =
        TypeSwitch<Operation *, Optional<UnaryEmitter>>(unaryOp)
            .Case([&](math::AbsFOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "abs");
              }
              return None;
            })
            .Case([&](math::CeilOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "ceil");
              }
              return None;
            })
            .Case([&](math::CountLeadingZerosOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "ctlz");
              }
              return None;
            })
            .Case([&](math::ExpOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "exp");
              }
              return None;
            })
            .Case([&](math::FloorOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "floor");
              }
              return None;
            })
            .Case([&](math::LogOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "log");
              }
              return None;
            })
            .Case([&](arith::NegFOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "neg");
              }
              return None;
            })
            .Case([&](math::RsqrtOp op) -> Optional<UnaryEmitter> {
              if (op.getResult().getType().getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "rsqrt");
              }
              return None;
            })
            .Default([](Operation *) { return None; });

    // Determine op type to lower to.
    if (!emitter) {
      return rewriter.notifyMatchFailure(op, "unrecognized unary op");
    }
    if (failed(emitter->initialize(op.getLoc(), rewriter))) return failure();

    emitter->emit(op.getLoc(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Matches a "trivial" generic which only yields, emitting as copy
/// operation(s).
struct LinalgTrivialGenericConversion
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto &children = op.getBlock()->getOperations();
    // Only match one child (yield).
    if (children.size() != 1) return failure();
    // Only match parallel loops.
    if (op.getNumParallelLoops() != op.getNumLoops()) return failure();

    // Presumed to be a yield terminator: configure the emitter.
    CopyEmitter emitter;
    auto allOperands = op.getInputAndOutputOperands();
    Operation &yieldOp = children.front();
    for (auto it : llvm::enumerate(yieldOp.getOperands())) {
      unsigned outputIndex = it.index();
      Value yieldOperand = it.value();
      if (auto blockArg = yieldOperand.dyn_cast<BlockArgument>()) {
        unsigned inputIndex = blockArg.getArgNumber();
        OpOperand *input = op.getInputOperand(inputIndex);
        OpOperand *output = op.getOutputOperand(outputIndex);
        emitter.copies.emplace_back(
            CopyEmitter::Descriptor{input->get(), op.getTiedIndexingMap(input)},
            CopyEmitter::Descriptor{output->get(),
                                    op.getTiedIndexingMap(output)});
      } else {
        return rewriter.notifyMatchFailure(op, "does not yield blockargs");
      }
    }

    if (failed(emitter.initialize(op.getLoc(), rewriter))) return failure();
    emitter.emit(op.getLoc(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LinalgFillConversion : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;
  struct OpInfo {
    linalg::FillOp op;
    Value scalar;
    Value out;
    StridedBufferAnalysis outAnal;
    int64_t getRank() { return outAnal.getRank(); }

    OpInfo(linalg::FillOp op) : op(op), outAnal(op.getOutputs().front()) {
      scalar = op.getInputs().front();
      out = op.getOutputs().front();
    }
  };

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    OpInfo info(op);
    if (!info.outAnal.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for out");
    }

    // Switch based on specialization.
    if (info.getRank() == 2 && info.outAnal.isUnitInnerStride()) {
      return handle2DTile(info, rewriter);
    }

    return rewriter.notifyMatchFailure(op, "unhandled fill variant");
  }

  LogicalResult handle2DTile(OpInfo &info, PatternRewriter &rewriter) const {
    auto loc = info.op.getLoc();
    StridedBufferDescriptor &outDesc = info.outAnal.getDesc(rewriter);
    Value m = outDesc.sizes[0];
    Value n = outDesc.sizes[1];
    Value stride = outDesc.strides[0];
    Value outBuffer = outDesc.castToLinear(loc, rewriter);

    rewriter.replaceOpWithNewOp<IREE::VMVX::Fill2DOp>(
        info.op, info.scalar, outBuffer, outDesc.offset, stride, m, n);
    return success();
  }
};

/// Convert a linalg.matmul.
struct LinalgMatmulConversion
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  struct OpInfo {
    linalg::ContractionOpInterface contract;
    linalg::LinalgOp op;

    Value lhs;
    Value rhs;
    Value out;

    StridedBufferAnalysis lhsAnal;
    StridedBufferAnalysis rhsAnal;
    StridedBufferAnalysis outAnal;

    OpInfo(linalg::ContractionOpInterface contract)
        : contract(contract),
          op(llvm::cast<linalg::LinalgOp>(contract.getOperation())),
          lhsAnal(contract.lhs()),
          rhsAnal(contract.rhs()),
          outAnal(op.outputs().front()) {
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
    OpInfo info(op);

    // Check that buffer descriptors could be computed.
    if (!info.lhsAnal.isValid() || !info.rhsAnal.isValid() ||
        !info.outAnal.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for operands");
    }

    // Check for unit inner strides.
    if (!info.lhsAnal.isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "lhs has non-unit inner stride");
    }
    if (!info.rhsAnal.isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "rhs has non-unit inner stride");
    }
    if (!info.outAnal.isUnitInnerStride()) {
      return rewriter.notifyMatchFailure(op, "out has non-unit inner stride");
    }

    // Switch on contraction type.
    if (info.contract.isRowMajorMatmul() ||
        info.contract.isColumnMajorMatmul()) {
      if (succeeded(handleConformingMatmul2D(info, rewriter))) {
        return success();
      }
    }

    // Match failure.
    return rewriter.notifyMatchFailure(op, "unsupported contraction variant");
  }

  LogicalResult handleConformingMatmul2D(OpInfo &info,
                                         PatternRewriter &rewriter) const {
    auto loc = info.op.getLoc();
    auto &lhsDesc = info.lhsAnal.getDesc(rewriter);
    auto &rhsDesc = info.rhsAnal.getDesc(rewriter);
    auto &outDesc = info.outAnal.getDesc(rewriter);
    // Determine m, n, k based on dims.
    int flags = 0;
    Value m, n, k;
    if (info.contract.isRowMajorMatmul()) {
      m = lhsDesc.sizes[0];
      k = rhsDesc.sizes[0];
      n = rhsDesc.sizes[1];
    } else if (info.contract.isColumnMajorMatmul()) {
      m = lhsDesc.sizes[0];
      k = rhsDesc.sizes[1];
      n = rhsDesc.sizes[0];
      // TODO: Flag constants somewhere.
      flags |= 1;
    } else {
      return failure();
    }

    // Alpha/beta: We always start the lowering with alpha/beta set to 1.
    // Simplification patterns within VMVX will simplify this if possible.
    Value alpha = info.getOneValue(rewriter);
    Value beta = alpha;

    auto lhsBuffer = lhsDesc.castToLinear(loc, rewriter);
    auto rhsBuffer = rhsDesc.castToLinear(loc, rewriter);
    auto outBuffer = outDesc.castToLinear(loc, rewriter);

    rewriter.replaceOpWithNewOp<IREE::VMVX::MatmulOp>(
        info.op,
        // LHS
        lhsBuffer, lhsDesc.offset, lhsDesc.strides[0],
        // RHS
        rhsBuffer, rhsDesc.offset, rhsDesc.strides[0],
        // Out
        outBuffer, outDesc.offset, outDesc.strides[0],
        // m,n,k
        m, n, k,
        // alpha, beta
        alpha, beta,
        // flags
        lhsDesc.getElementTypeAttr(), rhsDesc.getElementTypeAttr(),
        outDesc.getElementTypeAttr(), rewriter.getI32IntegerAttr(flags));
    return success();
  }
};

}  // namespace

class VMVXLowerLinalgMicrokernelsPass
    : public VMVXLowerLinalgMicrokernelsBase<VMVXLowerLinalgMicrokernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::VMVX::VMVXDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<LinalgBinaryGenericConversion, LinalgFillConversion,
                    LinalgMatmulConversion, LinalgTrivialGenericConversion,
                    LinalgUnaryGenericConversion>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    if (warnOnUnconverted) {
      getOperation()->walk([](Operation *op) {
        if (llvm::isa<linalg::LinalgOp>(op)) {
          auto diag = op->emitWarning(
              "Linalg op not converted to microkernel and will be implemented "
              "with fallback scalar loops");
          diag.attachNote(op->getLoc()) << "unmatched op: " << *op;
        }
      });
    }
  }
};

std::unique_ptr<Pass> createVMVXLowerLinalgMicrokernelsPass() {
  return std::make_unique<VMVXLowerLinalgMicrokernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
