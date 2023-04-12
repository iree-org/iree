// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

// Returns true if all inner dimensions (that is, all but the outer-most dim)
// are contiguous row-major.
//
// TODO(#11633): Dynamic dimensions are currently assumed to be row-major.
//
// This is vacuously true for
// rank<=1 (as there are no inner dims). For rank 2, this is equivalent to
// asking for the inner dimension to have unit stride. For rank>=3, this is
// asking for the strides of all but the outermost dimension to equal the
// product of the static sizes inner dimensions past them --- so in particular,
// this is requiring all but the outer two dimensions to have a static size.
bool verifyMemRefInnerDimsContiguousRowMajor(MemRefType type) {
  int rank = type.getRank();
  if (rank <= 1) {
    return true;
  }

  SmallVector<int64_t> strides;
  int64_t offset;
  if (type.getLayout().isIdentity()) {
    return true;
  }

  if (failed(mlir::getStridesAndOffset(type, strides, offset))) {
    return false;
  }

  ArrayRef<int64_t> sizes = type.getShape();
  assert(rank >= 2);  // Ensured by above early return.
  if (strides[rank - 1] != 1) {
    return false;
  }
  int64_t product_of_inner_sizes = 1;
  for (int i = rank - 1; i >= 2; --i) {
    if (sizes[i] == ShapedType::kDynamic) {
      // TODO(#11633): Dynamic dimensions are currently assumed to be row-major.
      product_of_inner_sizes = ShapedType::kDynamic;
    } else {
      if (product_of_inner_sizes != ShapedType::kDynamic) {
        product_of_inner_sizes *= sizes[i];
      }
    }
    if (strides[i - 1] != product_of_inner_sizes) {
      return false;
    }
  }

  return true;
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

  // Returns true if all inner dimensions (that is, all but the outer-most dim)
  // are statically known to be contiguous row-major.
  bool areInnerDimsContiguousRowMajor() const {
    return verifyMemRefInnerDimsContiguousRowMajor(memRefType);
  }

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

  // Returns true if all inner dimensions (that is, all but the outer-most dim)
  // are statically known to be contiguous row-major.
  bool areInnerDimsContiguousRowMajor() {
    return verifyMemRefInnerDimsContiguousRowMajor(getType());
  }

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
  std::optional<StridedBufferDescriptor> desc;
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
            loc,
            // Input buffers
            params.in0Buffer, operands.first.bufferDesc->offset,
            params.in1Buffer, operands.second.bufferDesc->offset,
            // Output buffers
            params.outBuffer, result.bufferDesc->offset,
            // Other operands
            rewriter.getStringAttr(selection.opcode),
            // LHS
            params.in0Strides,
            // RHS
            params.in1Strides,
            // OUT
            params.outStrides,
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
            loc,
            // Input buffers
            params.inBuffer, operand.bufferDesc->offset,
            // Output buffers
            params.outBuffer, result.bufferDesc->offset,
            // Other operands
            rewriter.getStringAttr(selection.opcode),
            // IN
            params.inStrides,
            // OUT
            params.outStrides,
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

    rewriter.create<IREE::VMVX::CopyOp>(loc,
                                        // Input buffers
                                        inBuffer, in.bufferDesc->offset,
                                        // Output buffers
                                        outBuffer, out.bufferDesc->offset,
                                        // Other operands
                                        // IN
                                        inStrides,
                                        // OUT
                                        outStrides,
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
    OpOperand *result = op.getDpsInitOperand(0);

    // Returns an emitter for a generic binary compatible operation where
    // |binaryOp| has a 1:1 correspondance with |opcode|.
    auto configureGenericBinary =
        [&](Operation *binaryOp,
            StringRef opcode) -> std::optional<BinaryEmitter> {
      SmallVector<BinaryEmitter::Descriptor, 2> operands;
      // Make sure that the binary op has operands that map to the
      // ins and detect the order.
      auto selection = BinaryEmitter::OpSelection::genericBinary(opcode);
      if (binaryOp->getOperand(0) == operandScalar0 &&
          binaryOp->getOperand(1) == operandScalar1) {
        // 1:1 matching.
        return BinaryEmitter(
            BinaryEmitter::Descriptor(operand0->get(),
                                      op.getMatchingIndexingMap(operand0)),
            BinaryEmitter::Descriptor(operand1->get(),
                                      op.getMatchingIndexingMap(operand1)),
            BinaryEmitter::Descriptor(result->get(),
                                      op.getMatchingIndexingMap(result)),
            selection);
      } else if (binaryOp->getOperand(1) == operandScalar0 &&
                 binaryOp->getOperand(0) == operandScalar1) {
        // Inverted operands.
        return BinaryEmitter(
            BinaryEmitter::Descriptor(operand1->get(),
                                      op.getMatchingIndexingMap(operand1)),
            BinaryEmitter::Descriptor(operand0->get(),
                                      op.getMatchingIndexingMap(operand0)),
            BinaryEmitter::Descriptor(result->get(),
                                      op.getMatchingIndexingMap(result)),
            selection);
      } else {
        return std::nullopt;
      }
    };

    // Select the op to lower to and configure the emitter.
    // Emit from the iree_ukernel_x32b_opcode_t table.
    Type resultType = binaryOp->getResult(0).getType();
    if (!resultType.isIntOrFloat()) return failure();
    std::optional<BinaryEmitter> emitter =
        TypeSwitch<Operation *, std::optional<BinaryEmitter>>(binaryOp)
            .Case([&](arith::AddFOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "add");
              }
              return std::nullopt;
            })
            .Case([&](arith::AddIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "add");
              }
              return std::nullopt;
            })
            .Case([&](arith::AndIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "and");
              }
              return std::nullopt;
            })
            .Case([&](arith::DivFOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "div");
              }
              return std::nullopt;
            })
            .Case([&](arith::DivSIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "divs");
              }
              return std::nullopt;
            })
            .Case([&](arith::DivUIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "divu");
              }
              return std::nullopt;
            })
            .Case([&](arith::MulFOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "mul");
              }
              return std::nullopt;
            })
            .Case([&](arith::MulIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "mul");
              }
              return std::nullopt;
            })
            .Case([&](arith::OrIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "or");
              }
              return std::nullopt;
            })
            .Case([&](arith::ShLIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "shl");
              }
              return std::nullopt;
            })
            .Case([&](arith::ShRSIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "shrs");
              }
              return std::nullopt;
            })
            .Case([&](arith::XOrIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "xor");
              }
              return std::nullopt;
            })
            .Case([&](arith::SubFOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "sub");
              }
              return std::nullopt;
            })
            .Case([&](arith::SubIOp op) -> std::optional<BinaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericBinary(op, "sub");
              }
              return std::nullopt;
            })
            .Default([](Operation *) { return std::nullopt; });

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
    OpOperand *result = op.getDpsInitOperand(0);

    // Returns an emitter for a generic binary compatible operation where
    // |binaryOp| has a 1:1 correspondance with |opcode|.
    auto configureGenericUnary =
        [&](Operation *unaryOp,
            StringRef opcode) -> std::optional<UnaryEmitter> {
      SmallVector<UnaryEmitter::Descriptor, 2> operands;
      // Make sure that the binary op has operands that map to the
      // ins and detect the order.
      auto selection = UnaryEmitter::OpSelection::genericUnary(opcode);
      return UnaryEmitter(
          UnaryEmitter::Descriptor(operand0->get(),
                                   op.getMatchingIndexingMap(operand0)),
          UnaryEmitter::Descriptor(result->get(),
                                   op.getMatchingIndexingMap(result)),
          selection);
    };

    // Select the op to lower to and configure the emitter.
    // Emit from the iree_ukernel_x32b_opcode_t table.
    Type resultType = unaryOp->getResult(0).getType();
    if (!resultType.isIntOrFloat()) return failure();
    std::optional<UnaryEmitter> emitter =
        TypeSwitch<Operation *, std::optional<UnaryEmitter>>(unaryOp)
            .Case([&](math::AbsFOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "abs");
              }
              return std::nullopt;
            })
            .Case([&](math::CeilOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "ceil");
              }
              return std::nullopt;
            })
            .Case([&](math::CountLeadingZerosOp op)
                      -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "ctlz");
              }
              return std::nullopt;
            })
            .Case([&](math::ExpOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "exp");
              }
              return std::nullopt;
            })
            .Case([&](math::FloorOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "floor");
              }
              return std::nullopt;
            })
            .Case([&](math::LogOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "log");
              }
              return std::nullopt;
            })
            .Case([&](arith::NegFOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "neg");
              }
              return std::nullopt;
            })
            .Case([&](math::RsqrtOp op) -> std::optional<UnaryEmitter> {
              if (resultType.getIntOrFloatBitWidth() == 32) {
                return configureGenericUnary(op, "rsqrt");
              }
              return std::nullopt;
            })
            .Default([](Operation *) { return std::nullopt; });

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
    Operation &yieldOp = children.front();
    for (auto [outputIndex, yieldOperand] :
         llvm::enumerate(yieldOp.getOperands())) {
      if (auto blockArg = yieldOperand.dyn_cast<BlockArgument>()) {
        unsigned inputIndex = blockArg.getArgNumber();
        OpOperand *input = op.getDpsInputOperand(inputIndex);
        OpOperand *output = op.getDpsInitOperand(outputIndex);
        emitter.copies.emplace_back(
            CopyEmitter::Descriptor{input->get(),
                                    op.getMatchingIndexingMap(input)},
            CopyEmitter::Descriptor{output->get(),
                                    op.getMatchingIndexingMap(output)});
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
    if (info.getRank() == 2 && info.outAnal.areInnerDimsContiguousRowMajor()) {
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

    rewriter.replaceOpWithNewOp<IREE::VMVX::Fill2DOp>(info.op,
                                                      // Input buffers (none)
                                                      // Output buffers
                                                      outBuffer, outDesc.offset,
                                                      // Other operands
                                                      info.scalar, stride, m,
                                                      n);
    return success();
  }
};

struct LinalgExtPackConversion
    : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern::OpRewritePattern;
  static bool isSupportedElementTypes(Type inElType, Type outElType) {
    if (inElType.isF32() && outElType.isF32()) {
      return true;
    }
    if (inElType.isSignlessInteger(8) && inElType.isSignlessInteger(8)) {
      return true;
    }
    if (inElType.isSignlessInteger(32) && inElType.isSignlessInteger(32)) {
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType inType = op.getInputType().cast<MemRefType>();
    MemRefType outType = op.getOutputType().cast<MemRefType>();

    if (!isSupportedElementTypes(inType.getElementType(),
                                 outType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "unsupported combination of in/out element types");
    }

    if (inType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected input to be 2D");
    }

    if (outType.getRank() != 4) {
      return rewriter.notifyMatchFailure(op, "expected output to be 4D");
    }

    int64_t innerDimsPos[2] = {0, 1};
    ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
    if (!innerDimsPosArr.empty()) {
      innerDimsPos[0] = innerDimsPosArr[0];
      innerDimsPos[1] = innerDimsPosArr[1];
    }

    int64_t outerDimsPerm[2] = {0, 1};
    ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
    if (!outerDimsPosArr.empty()) {
      outerDimsPerm[0] = outerDimsPosArr[0];
      outerDimsPerm[1] = outerDimsPosArr[1];
    }

    int flags = 0;

    if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
      // nothing to do
    } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
      flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
    }

    if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
      // nothing to do
    } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
      flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
    }

    SmallVector<int64_t> inStrides;
    int64_t inOffset;
    if (failed(mlir::getStridesAndOffset(inType, inStrides, inOffset))) {
      return rewriter.notifyMatchFailure(
          op, "failed to getStridesAndOffset for input");
    }

    if (inStrides[1] != 1) {
      return rewriter.notifyMatchFailure(
          op, "unsupported input layout with non-contiguous inner dimension");
    }

    if (!verifyMemRefInnerDimsContiguousRowMajor(outType)) {
      return rewriter.notifyMatchFailure(
          op,
          "expected the output's inner dimensions to be contiguous row-major");
    }

    StridedBufferAnalysis inAnalysis(op.getInput());
    StridedBufferAnalysis outAnalysis(op.getOutput());

    if (!inAnalysis.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for input");
    }

    if (!outAnalysis.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for output");
    }

    Location loc = op.getLoc();

    Value paddingValue = op.getPaddingValue();
    if (!paddingValue) {
      paddingValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(inType.getElementType()),
          inType.getElementType());
    }

    StridedBufferDescriptor &inDesc = inAnalysis.getDesc(rewriter);
    StridedBufferDescriptor &outDesc = outAnalysis.getDesc(rewriter);

    Value inBuffer = inDesc.castToLinear(loc, rewriter);
    Value outBuffer = outDesc.castToLinear(loc, rewriter);
    Value inSize0 = inDesc.sizes[0];
    Value inSize1 = inDesc.sizes[1];
    Value inStride0 = inDesc.strides[0];
    Value outSize0 = outDesc.sizes[0];
    Value outSize1 = outDesc.sizes[1];
    Value outSize2 = outDesc.sizes[2];
    Value outSize3 = outDesc.sizes[3];
    Value outStride0 = outDesc.strides[0];

    rewriter.replaceOpWithNewOp<IREE::VMVX::PackOp>(
        op,
        // Input buffers
        inBuffer, inDesc.offset,
        // Output buffers
        outBuffer, outDesc.offset,
        // Other operands
        // input
        inStride0,
        // output
        outStride0,
        // input shape
        inSize0, inSize1,
        // output shape
        outSize0, outSize1, outSize2, outSize3, paddingValue,
        // flags
        inDesc.getElementTypeAttr(), outDesc.getElementTypeAttr(),
        rewriter.getI32IntegerAttr(flags));
    return success();
  }
};

struct LinalgExtUnpackConversion
    : public OpRewritePattern<IREE::LinalgExt::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;
  static bool isSupportedElementTypes(Type inElType, Type outElType) {
    if (inElType.isF32() && outElType.isF32()) {
      return true;
    }
    if (inElType.isSignlessInteger(8) && inElType.isSignlessInteger(8)) {
      return true;
    }
    if (inElType.isSignlessInteger(32) && inElType.isSignlessInteger(32)) {
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(IREE::LinalgExt::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType inType = op.getInputType().cast<MemRefType>();
    MemRefType outType = op.getOutputType().cast<MemRefType>();

    if (!isSupportedElementTypes(inType.getElementType(),
                                 outType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "unsupported combination of in/out element types");
    }

    if (inType.getRank() != 4) {
      return rewriter.notifyMatchFailure(op, "expected input to be 4D");
    }

    if (outType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected output to be 2D");
    }

    int64_t innerDimsPos[2] = {0, 1};
    ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
    if (!innerDimsPosArr.empty()) {
      innerDimsPos[0] = innerDimsPosArr[0];
      innerDimsPos[1] = innerDimsPosArr[1];
    }

    int64_t outerDimsPerm[2] = {0, 1};
    ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
    if (!outerDimsPosArr.empty()) {
      outerDimsPerm[0] = outerDimsPosArr[0];
      outerDimsPerm[1] = outerDimsPosArr[1];
    }

    int flags = 0;

    if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
      // nothing to do
    } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
      flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
    }

    if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
      // nothing to do
    } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
      flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
    }

    SmallVector<int64_t> outStrides;
    int64_t outOffset;
    if (failed(mlir::getStridesAndOffset(outType, outStrides, outOffset))) {
      return rewriter.notifyMatchFailure(
          op, "failed to getStridesAndOffset for output");
    }

    if (outStrides[1] != 1) {
      return rewriter.notifyMatchFailure(
          op, "unsupported output layout with non-contiguous inner dimension");
    }

    if (!verifyMemRefInnerDimsContiguousRowMajor(inType)) {
      return rewriter.notifyMatchFailure(
          op,
          "expected the input's inner dimensions to be contiguous row-major");
    }

    StridedBufferAnalysis inAnalysis(op.getInput());
    StridedBufferAnalysis outAnalysis(op.getOutput());

    if (!inAnalysis.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for input");
    }

    if (!outAnalysis.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for output");
    }

    Location loc = op.getLoc();

    StridedBufferDescriptor &inDesc = inAnalysis.getDesc(rewriter);
    StridedBufferDescriptor &outDesc = outAnalysis.getDesc(rewriter);

    Value inBuffer = inDesc.castToLinear(loc, rewriter);
    Value outBuffer = outDesc.castToLinear(loc, rewriter);
    Value inSize0 = inDesc.sizes[0];
    Value inSize1 = inDesc.sizes[1];
    Value inSize2 = inDesc.sizes[2];
    Value inSize3 = inDesc.sizes[3];
    Value inStride0 = inDesc.strides[0];
    Value outSize0 = outDesc.sizes[0];
    Value outSize1 = outDesc.sizes[1];
    Value outStride0 = outDesc.strides[0];

    rewriter.replaceOpWithNewOp<IREE::VMVX::UnpackOp>(
        op,
        // Input buffers
        inBuffer, inDesc.offset,
        // Output buffers
        outBuffer, outDesc.offset,
        // Other operands
        // input
        inStride0,
        // output
        outStride0,
        // input shape
        inSize0, inSize1, inSize2, inSize3,
        // output shape
        outSize0, outSize1,
        // flags
        inDesc.getElementTypeAttr(), outDesc.getElementTypeAttr(),
        rewriter.getI32IntegerAttr(flags));
    return success();
  }
};

int getNumberOfUses(Value v) {
  auto uses = v.getUses();
  return std::distance(uses.begin(), uses.end());
}

template <typename OpType>
OpType getUserOfType(Value v) {
  auto uses = v.getUses();
  for (const auto &u : uses) {
    if (OpType user = llvm::dyn_cast<OpType>(u.getOwner())) {
      return user;
    }
  }
  return nullptr;
}

linalg::FillOp findFillOpSolelyZeroingOutputOf(linalg::LinalgOp op) {
  Value out = op.getDpsInitOperand(0)->get();
  if (getNumberOfUses(out) != 2) {
    return nullptr;
  }
  linalg::FillOp fillOp = getUserOfType<linalg::FillOp>(out);
  if (!fillOp) {
    return nullptr;
  }
  if (!fillOp->isBeforeInBlock(op)) {
    return nullptr;
  }
  Value fillValue = fillOp.value();
  if (auto constIntOp = fillValue.getDefiningOp<arith::ConstantIntOp>()) {
    if (constIntOp.value() == 0) {
      return fillOp;
    }
  }
  if (auto constFloatOp = fillValue.getDefiningOp<arith::ConstantFloatOp>()) {
    if (constFloatOp.value().isZero()) {
      return fillOp;
    }
  }
  return nullptr;
}

/// Convert supported linalg contraction ops like matmul.
struct LinalgContractionConversion
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
          outAnal(op.getDpsInitOperands().front()->get()) {
      lhs = contract.lhs();
      rhs = contract.rhs();
      out = op.getDpsInitOperands().front()->get();
    }
  };

  static bool isSupportedElementTypes(const OpInfo &info) {
    Type lhsElType = info.lhs.getType().cast<ShapedType>().getElementType();
    Type rhsElType = info.rhs.getType().cast<ShapedType>().getElementType();
    Type outElType = info.out.getType().cast<ShapedType>().getElementType();
    if (lhsElType.isF32() && rhsElType.isF32() && outElType.isF32()) {
      return true;
    }
    if (lhsElType.isSignlessInteger(8) && rhsElType.isSignlessInteger(8) &&
        outElType.isSignlessInteger(32)) {
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    OpInfo info(op);

    if (!isSupportedElementTypes(info)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported combination of lhs/rhs/out element types");
    }

    // Check that buffer descriptors could be computed.
    if (!info.lhsAnal.isValid() || !info.rhsAnal.isValid() ||
        !info.outAnal.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "could not compute buffer descriptor for operands");
    }

    // Check for unit inner strides.
    if (!info.lhsAnal.areInnerDimsContiguousRowMajor()) {
      return rewriter.notifyMatchFailure(op, "lhs has non-unit inner stride");
    }
    if (!info.rhsAnal.areInnerDimsContiguousRowMajor()) {
      return rewriter.notifyMatchFailure(op, "rhs has non-unit inner stride");
    }
    if (!info.outAnal.areInnerDimsContiguousRowMajor()) {
      return rewriter.notifyMatchFailure(op, "out has non-unit inner stride");
    }

    // Switch on contraction type.
    if (info.contract.isRowMajorMatmul()) {
      if (succeeded(handleConformingMatmul2D(info, rewriter))) {
        return success();
      }
    }

    // Match failure.
    return rewriter.notifyMatchFailure(op, "unsupported contraction variant");
  }

  LogicalResult handleConformingMatmul2D(OpInfo &info,
                                         PatternRewriter &rewriter) const {
    int flags = 0;
    if (linalg::FillOp fillOp = findFillOpSolelyZeroingOutputOf(info.op)) {
      rewriter.eraseOp(fillOp);  // let the matmul overwrite the accumulator.
    } else {
      flags |= IREE_UK_FLAG_ACCUMULATE;  // accumulate into existing.
    }

    auto &lhsDesc = info.lhsAnal.getDesc(rewriter);
    auto &rhsDesc = info.rhsAnal.getDesc(rewriter);
    auto &outDesc = info.outAnal.getDesc(rewriter);

    Value m = lhsDesc.sizes[0];
    Value k = rhsDesc.sizes[0];
    Value n = rhsDesc.sizes[1];

    auto loc = info.op.getLoc();
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
    // Patterns that need to be applied first to match multiple linalg ops
    // before they have been lowered.
    {
      RewritePatternSet patterns(&getContext());
      // LinalgContractionConversion needs to match linalg::FillOp to determine
      // whether to set the 'accumulate' flag or just erase the FillOp.
      patterns.insert<LinalgContractionConversion>(&getContext());

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Other lowering patterns
    {
      RewritePatternSet patterns(&getContext());
      patterns
          .insert<LinalgBinaryGenericConversion, LinalgFillConversion,
                  LinalgTrivialGenericConversion, LinalgUnaryGenericConversion,
                  LinalgExtPackConversion, LinalgExtUnpackConversion>(
              &getContext());

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
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
