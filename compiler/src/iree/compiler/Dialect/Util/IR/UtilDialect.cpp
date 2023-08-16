// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"

#include "iree/compiler/Dialect/Util/IR/CASResources.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

//===----------------------------------------------------------------------===//
// UtilBytecodeDialectInterface
//===----------------------------------------------------------------------===//
namespace {

struct UtilBytecodeDialectInterface : public BytecodeDialectInterface {
  enum class AttributeCode : uint64_t {
    CASElements = 0,
  };
  UtilBytecodeDialectInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  LogicalResult writeAttribute(Attribute baseAttr,
                               DialectBytecodeWriter &writer) const override {
    if (auto attr = llvm::dyn_cast<CASElementsAttr>(baseAttr)) {
      writer.writeVarInt(static_cast<uint64_t>(AttributeCode::CASElements));
      writer.writeType(attr.getType());
      writer.writeResourceHandle(attr.getHandle());
      return success();
    }

    return failure(); // Fallback to text.
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    uint64_t codeValue;
    if (failed(reader.readVarInt(codeValue)))
      return {};
    switch (static_cast<AttributeCode>(codeValue)) {
    case AttributeCode::CASElements: {
      ShapedType t;
      FailureOr<UtilDialect::CASResourceHandle> handle;
      if (failed(reader.readType<ShapedType>(t)))
        return {};
      handle = reader.readResourceHandle<UtilDialect::CASResourceHandle>();
      if (failed(handle))
        return {};
      return CASElementsAttr::get(t, std::move(*handle));
      break;
    }
    }

    reader.emitError() << "unrecognized util dialect attribute code: "
                       << codeValue;
    return {};
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// UtilOpAsmDialectInterface
//===----------------------------------------------------------------------===//
namespace {

// Used for custom printing support.
struct UtilOpAsmInterface : public OpAsmDialectInterface {
  UtilOpAsmInterface(Dialect *dialect,
                     UtilDialect::BlobManagerInterface &blobManager)
      : OpAsmDialectInterface(dialect), blobManager(blobManager) {}
  /// Hooks for getting an alias identifier alias for a given symbol, that is
  /// not necessarily a part of this dialect. The identifier is used in place
  /// of the symbol when printing textual IR. These aliases must not contain
  /// `.` or end with a numeric digit([0-9]+). Returns success if an alias was
  /// provided, failure otherwise.

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto compositeAttr = llvm::dyn_cast<CompositeAttr>(attr)) {
      os << "composite_of_" << compositeAttr.getTotalLength() << "b";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<UtilDialect::CASResourceHandle>(handle).getKey().str();
  }

  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    llvm::errs() << "::: declareResource " << key << "\n";
    auto blobEntry = blobManager.insert(key);
    llvm::errs() << "  = " << blobEntry.getKey() << "\n";
    return blobEntry;
  }

  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();

    // Update the blob for this entry.
    llvm::errs() << "::: parseResource: " << entry.getKey() << "\n";
    blobManager.update(entry.getKey(), std::move(*blob));
    return success();
  }
  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    llvm::errs() << "::: buildResources\n";
    blobManager.buildResources(provider, referencedResources.getArrayRef());
  }

private:
  UtilDialect::BlobManagerInterface &blobManager;
};
} // namespace

//===----------------------------------------------------------------------===//
// UtilInlinerInterface
//===----------------------------------------------------------------------===//
namespace {
// Used to control inlining behavior.
struct UtilInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    auto returnOp = dyn_cast<IREE::Util::InitializerReturnOp>(op);
    if (!returnOp)
      return;
    // util.initialize.return takes no args - just branch to the new block.
    OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest, ValueRange{});
    op->erase();
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const final {
    // util.initialize.return takes no args.
  }
};
} // namespace

UtilDialect::UtilDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<UtilDialect>()) {
  auto &blobInterface = addInterface<BlobManagerInterface>();
  addInterface<UtilOpAsmInterface>(blobInterface);
  addInterface<UtilBytecodeDialectInterface>();
  addInterface<CASManagerDialectInterface>(blobInterface);
  addInterfaces<UtilInlinerInterface>();
  registerAttributes();
  registerTypes();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Util/IR/UtilOps.cpp.inc"
      >();
}

Operation *UtilDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (arith::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  }
  return nullptr;
}

template <typename DimOp>
struct FoldDimOp : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    auto shapeAwareOp =
        dyn_cast_or_null<ShapeAwareOpInterface>(op.getSource().getDefiningOp());
    if (!shapeAwareOp)
      return failure();

    // We only support static dimension indices today (as in general we only
    // support ranked shapes). If we find dynamic indices sneaking in we will
    // need to do something much more complex - or prevent them from sneaking
    // in.
    APInt index;
    if (!matchPattern(op.getIndex(), m_ConstantInt(&index))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-constant dim index unsupported");
    }

    // If it's a static dim then just fold to that.
    auto type = llvm::cast<ShapedType>(op.getSource().getType());
    int64_t staticDim = type.getDimSize(index.getZExtValue());
    if (staticDim != ShapedType::kDynamic) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, staticDim);
      return success();
    }

    // Otherwise try to get the dynamic dimension cheaply without the need to
    // insert new IR.
    unsigned dynamicIdx = type.getDynamicDimIndex(index.getZExtValue());
    auto dynamicDims =
        shapeAwareOp.getResultDynamicDimsFromValue(op.getSource());
    rewriter.replaceOp(op, dynamicDims[dynamicIdx]);

    return success();
  }
};

void UtilDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.insert<FoldDimOp<memref::DimOp>>(getContext());
  results.insert<FoldDimOp<tensor::DimOp>>(getContext());
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
