// Copyright 2019 Google LLC
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

//===- AdjustIntegerWidthPass.cpp ------------------------------*- C++//-*-===//
//
// Pass to adjust integer widths of operations.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IREEToSPIRVPass.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pass to
/// 1) Legalize 64-bit integer values to 32-bit integers values.
/// 2) Legalize !spv.array containing i1 type to !spv.array of i8 types.
struct AdjustIntegerWidthPass : public OperationPass<AdjustIntegerWidthPass> {
  void runOnOperation() override;
};

// Returns true if the type contains any IntegerType of the width specified by
// `widths`
bool hasIntTypeOfWidth(Type type, ArrayRef<int64_t> widths) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return llvm::is_contained(widths, intType.getWidth());
  } else if (auto structType = type.dyn_cast<spirv::StructType>()) {
    for (int64_t i = 0, e = structType.getNumElements(); i != e; ++i) {
      if (hasIntTypeOfWidth(structType.getElementType(i), widths)) return true;
    }
    return false;
  } else if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    return hasIntTypeOfWidth(arrayType.getElementType(), widths);
  } else if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    return hasIntTypeOfWidth(ptrType.getPointeeType(), widths);
  }
  return false;
}

// Legalizes the integer types in struct.
// 1) i1 -> i8,
// 2) i64 -> i32.
Type legalizeIntegerType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      return IntegerType::get(8, intType.getContext());
    } else if (intType.getWidth() == 64) {
      return IntegerType::get(32, intType.getContext());
    }
  } else if (auto structType = type.dyn_cast<spirv::StructType>()) {
    SmallVector<Type, 1> elementTypes;
    for (auto i : llvm::seq<unsigned>(0, structType.getNumElements())) {
      elementTypes.push_back(legalizeIntegerType(structType.getElementType(i)));
    }
    // TODO(ravishankarm): Use ABI attributes to legalize the struct type.
    spirv::StructType::LayoutInfo structSize = 0;
    VulkanLayoutUtils::Size structAlignment = 1;
    auto t = spirv::StructType::get(elementTypes);
    return VulkanLayoutUtils::decorateType(t, structSize, structAlignment);
  } else if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    return spirv::ArrayType::get(
        legalizeIntegerType(arrayType.getElementType()),
        arrayType.getNumElements());
  } else if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    return spirv::PointerType::get(
        legalizeIntegerType(ptrType.getPointeeType()),
        ptrType.getStorageClass());
  }
  return type;
}

/// Rewrite access chain operations where the pointee type contains i1 or i64
/// types.
struct AdjustAccessChainOp : public OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern<spirv::AccessChainOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::AccessChainOp op,
                                     PatternRewriter &rewriter) const override {
    if (!hasIntTypeOfWidth(op.component_ptr()->getType(), {1, 64})) {
      return matchFailure();
    }
    ValueRange indices(op.indices());
    Type newType = legalizeIntegerType(op.component_ptr()->getType());
    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(op, newType,
                                                      op.base_ptr(), indices);
    return matchSuccess();
  }
};

/// Rewrite address of operations which refers to global variables that contain
/// i1 or i64 types.
struct AdjustAddressOfOp : public OpRewritePattern<spirv::AddressOfOp> {
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::AddressOfOp op,
                                     PatternRewriter &rewriter) const override {
    if (!hasIntTypeOfWidth(op.pointer()->getType(), {1, 64})) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, legalizeIntegerType(op.pointer()->getType()),
        SymbolRefAttr::get(op.variable(), rewriter.getContext()));
    return matchSuccess();
  }
};

/// Rewrite global variable ops that contain i1 and i64 types to i8 and i32
/// types respectively.
struct AdjustGlobalVariableWidth
    : public OpRewritePattern<spirv::GlobalVariableOp> {
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::GlobalVariableOp op,
                                     PatternRewriter &rewriter) const override {
    if (!hasIntTypeOfWidth(op.type(), {1, 64})) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, legalizeIntegerType(op.type()), op.sym_name(),
        op.getAttr("descriptor_set").cast<IntegerAttr>().getInt(),
        op.getAttr("binding").cast<IntegerAttr>().getInt());
    return matchSuccess();
  }
};

/// Rewrite loads from !spv.ptr<i64,..> to load from !spv.ptr<i32,...>
/// Rewrite loads from !spv.ptr<i1,...> to load from !spv.ptr<i8,...> followed
/// by a truncate to i1 type.
struct AdjustLoadOp : public OpRewritePattern<spirv::LoadOp> {
  using OpRewritePattern<spirv::LoadOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::LoadOp op,
                                     PatternRewriter &rewriter) const override {
    Type valueType = op.value()->getType();
    if (!hasIntTypeOfWidth(valueType, {1, 64})) {
      return matchFailure();
    }

    Type newType = legalizeIntegerType(valueType);
    const auto loc = op.getLoc();
    auto loadOp = rewriter.create<spirv::LoadOp>(
        loc, newType, op.ptr(),
        op.getAttrOfType<IntegerAttr>(
            spirv::attributeName<spirv::MemoryAccess>()),
        op.getAttrOfType<IntegerAttr>("alignment"));
    Value *result = loadOp.getResult();

    // If this is a load of a i1, replace it with a load of i8, and truncate the
    // result. Use INotEqualOp because SConvert doesn't work for i1.
    if (hasIntTypeOfWidth(valueType, {1})) {
      auto zero = spirv::ConstantOp::getZero(newType, loc, &rewriter);
      result = rewriter.create<spirv::INotEqualOp>(loc, valueType, result, zero)
                   .getResult();
    }

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }
};

/// Rewrite store operation that contain i1 and i64 types to i8 and i32 types
/// respectively.
struct AdjustStoreOp : public OpRewritePattern<spirv::StoreOp> {
  using OpRewritePattern<spirv::StoreOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::StoreOp op,
                                     PatternRewriter &rewriter) const override {
    Type valueType = op.value()->getType();
    if (!hasIntTypeOfWidth(valueType, {1, 64})) {
      return matchFailure();
    }

    Type newType = legalizeIntegerType(valueType);
    const auto loc = op.getLoc();
    Value *value;
    if (hasIntTypeOfWidth(valueType, {1})) {
      Value *zero =
          spirv::ConstantOp::getZero(newType, loc, &rewriter).getResult();
      Value *one =
          spirv::ConstantOp::getOne(newType, loc, &rewriter).getResult();
      value = rewriter.create<spirv::SelectOp>(loc, op.value(), one, zero)
                  .getResult();
    } else {
      value = rewriter.create<spirv::SConvertOp>(loc, newType, op.value())
                  .getResult();
    }
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(
        op, op.ptr(), value,
        op.getAttrOfType<IntegerAttr>(
            spirv::attributeName<spirv::MemoryAccess>()),
        op.getAttrOfType<IntegerAttr>("alignment"));
    return matchSuccess();
  }
};

/// Some Adjust* OpRewritePattern will generate useless SConvert operations,
/// which are invalid operations. Remove the SConvert operation if this is an
/// nop, i.e., if the source type and destination type are the same, remove the
/// op. It relies on the furthur finialization to remove the op, and propagate
/// right operands to other operations.
struct RemoveNopSConvertOp : public OpRewritePattern<spirv::SConvertOp> {
  using OpRewritePattern<spirv::SConvertOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::SConvertOp op,
                                     PatternRewriter &rewriter) const override {
    Type t1 = op.operand()->getType();
    Type t2 = op.result()->getType();
    if (t1 != t2) return matchFailure();
    auto zero = spirv::ConstantOp::getZero(t1, op.getLoc(), &rewriter);
    rewriter.replaceOpWithNewOp<spirv::IAddOp>(op, op.operand(), zero);
    return matchSuccess();
  }
};

/// Rewrite i64 constants to i32 constants.
struct AdjustConstantOp : public OpRewritePattern<spirv::ConstantOp> {
  using OpRewritePattern<spirv::ConstantOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(spirv::ConstantOp op,
                                     PatternRewriter &rewriter) const {
    Type constantType = op.getType();
    if (!hasIntTypeOfWidth(constantType, {64})) {
      return matchFailure();
    }
    Type newType = legalizeIntegerType(constantType);
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, newType,
        IntegerAttr::get(newType, op.value().dyn_cast<IntegerAttr>().getInt()));
    return matchSuccess();
  }
};

/// Rewrite integer arithmetic operations that operate on 64-bit integers to
/// operate on 32-bit integers.
template <typename OpTy>
struct AdjustIntegerArithmeticOperations : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const {
    Type resultType = op.result()->getType();
    if (!hasIntTypeOfWidth(resultType, {64})) {
      return Pattern::matchFailure();
    }
    Type newType = legalizeIntegerType(op.getResult()->getType());
    ValueRange operands(op.getOperation()->getOperands());
    rewriter.replaceOpWithNewOp<OpTy>(op, newType, operands, op.getAttrs());
    return Pattern::matchSuccess();
  }
};

void AdjustIntegerWidthPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<
      // Arithmetic ops:
      AdjustIntegerArithmeticOperations<spirv::GLSLSAbsOp>,
      AdjustIntegerArithmeticOperations<spirv::GLSLSSignOp>,
      AdjustIntegerArithmeticOperations<spirv::IAddOp>,
      AdjustIntegerArithmeticOperations<spirv::ISubOp>,
      AdjustIntegerArithmeticOperations<spirv::IMulOp>,
      AdjustIntegerArithmeticOperations<spirv::SDivOp>,
      AdjustIntegerArithmeticOperations<spirv::SModOp>,
      AdjustIntegerArithmeticOperations<spirv::SRemOp>,
      AdjustIntegerArithmeticOperations<spirv::UDivOp>,
      AdjustIntegerArithmeticOperations<spirv::UModOp>,
      // Structure ops:
      AdjustConstantOp,
      // Others:
      AdjustAccessChainOp, AdjustAddressOfOp, AdjustGlobalVariableWidth,
      AdjustLoadOp, AdjustStoreOp, RemoveNopSConvertOp>(&getContext());
  Operation *op = getOperation();
  applyPatternsGreedily(op->getRegions(), patterns);
}

static PassRegistration<AdjustIntegerWidthPass> pass(
    "iree-spirv-adjust-integer-width",
    "Adjust integer width from i1 and i64 types to i8 and i32 types "
    "respectively");

}  // namespace

std::unique_ptr<Pass> createAdjustIntegerWidthPass() {
  return std::make_unique<AdjustIntegerWidthPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
