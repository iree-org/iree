// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Utils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler::GlobalOptimization {

std::optional<CastOpInterface> getDefiningNonI1ExtendingCastOp(Value input) {
  // Returns true if the source of cast op has i1 element type.
  auto isI1Src = [](CastOpInterface castOp) -> bool {
    auto elemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    return elemType.getIntOrFloatBitWidth() == 1;
  };

  auto isExtending = [](CastOpInterface castOp) -> bool {
    auto inElemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    auto outElemType = getElementTypeOrSelf(castOp->getResultTypes()[0]);
    return inElemType.getIntOrFloatBitWidth() <
           outElemType.getIntOrFloatBitWidth();
  };

  std::optional<CastOpInterface> result = std::nullopt;
  auto castOp = input.getDefiningOp<CastOpInterface>();
  if (castOp) {
    if (!isI1Src(castOp) && isExtending(castOp)) {
      result = castOp;
    }
    return result;
  }
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return std::nullopt;
  }
  Value castIn = castOp->getOperand(0);
  if (auto blockArg = dyn_cast<BlockArgument>(castIn)) {
    if (blockArg.getArgNumber() != 0) {
      return std::nullopt;
    }
  }
  if (!isI1Src(castOp) && isExtending(castOp)) {
    result = castOp;
  }
  return result;
}

std::optional<Type> getCastElemType(Value input) {
  std::optional<CastOpInterface> castOp =
      getDefiningNonI1ExtendingCastOp(input);
  if (!castOp) {
    return std::nullopt;
  }
  Type castSrcElemType = getElementTypeOrSelf(castOp.value()->getOperand(0));
  if (isa<arith::ExtUIOp>(castOp.value())) {
    int64_t bitWidth = castSrcElemType.getIntOrFloatBitWidth();
    return IntegerType::get(castOp->getContext(), bitWidth,
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return castSrcElemType;
}

Value createGenericElementwiseCastOp(
    OpBuilder &builder, Location loc, Value input, CastOpInterface castOp,
    ArrayRef<NamedAttribute> attrs,
    std::optional<IREE::Encoding::EncodingAttr> encoding) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<AffineMap> maps(
      2, AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                           builder.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto elementType = getElementTypeOrSelf(castOp->getResultTypes()[0]);
  auto castedType = inputType.clone(elementType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init = encoding ? tensor::EmptyOp::create(builder, loc, inputMixedSizes,
                                                  elementType, *encoding)
                        : tensor::EmptyOp::create(builder, loc, inputMixedSizes,
                                                  elementType);
  return linalg::GenericOp::create(
             builder, loc, castedType, input, init, maps, iteratorTypes,
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Operation *scalarCast = castOp->clone(
                   Operation::CloneOptions().withResultTypes({{elementType}}));
               scalarCast->setLoc(nestedLoc);
               scalarCast->setOperands(args.take_front(1));
               b.insert(scalarCast);
               Value castRes = scalarCast->getResult(0);
               linalg::YieldOp::create(b, nestedLoc, castRes);
             },
             attrs)
      .getResult(0);
}

Value sumReduceDimensionSubset(ImplicitLocOpBuilder &rewriter, Value val,
                               Type accETy, ArrayRef<bool> is_reduction) {
  auto context = val.getContext();
  RankedTensorType ty = cast<RankedTensorType>(val.getType());

  llvm::SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynSizes;
  for (int i = 0, s = is_reduction.size(); i < s; i++) {
    if (is_reduction[i]) {
      continue;
    }

    staticSizes.push_back(ty.getDimSize(i));
    if (ty.isDynamicDim(i)) {
      dynSizes.push_back(tensor::DimOp::create(rewriter, val, i));
    }
  }

  // Create a zero-filled accumulator.
  Value initAcc =
      tensor::EmptyOp::create(rewriter, staticSizes, accETy, dynSizes);
  Value zeroInt = arith::ConstantIntOp::create(rewriter, accETy, 0).getResult();
  Value zeroAcc =
      linalg::FillOp::create(rewriter, zeroInt, initAcc).getResult(0);

  SmallVector<AffineExpr> filterExprs(ty.getRank());
  SmallVector<AffineExpr> outputExprs;
  SmallVector<utils::IteratorType> iterators;

  for (int i = 0, s = ty.getRank(); i < s; i++) {
    if (!is_reduction[i]) {
      auto expr = rewriter.getAffineDimExpr(iterators.size());
      iterators.push_back(utils::IteratorType::parallel);

      outputExprs.push_back(expr);
      filterExprs[i] = expr;
    }
  }

  for (int i = 0, s = filterExprs.size(); i < s; i++) {
    if (!filterExprs[i]) {
      auto expr = mlir::getAffineDimExpr(iterators.size(), context);
      iterators.push_back(utils::IteratorType::reduction);
      filterExprs[i] = expr;
    }
  }

  SmallVector<AffineMap> affineMaps{
      AffineMap::get(ty.getRank(), 0, filterExprs, context),
      AffineMap::get(ty.getRank(), 0, outputExprs, context)};

  return linalg::GenericOp::create(
             rewriter, zeroAcc.getType(), ValueRange{val}, ValueRange{zeroAcc},
             affineMaps, iterators,
             [=](OpBuilder &b, Location loc, ValueRange args) {
               Value ext = arith::ExtSIOp::create(b, loc, accETy, args[0]);
               Value sum = arith::AddIOp::create(b, loc, ext, args[1]);
               linalg::YieldOp::create(b, loc, sum);
             })
      .getResult(0);
}

IREE::Util::GlobalOpInterface
getLoadedGlobal(Value value, SymbolTableCollection &symbolTables) {
  auto loadOp = value.getDefiningOp<IREE::Util::GlobalLoadOpInterface>();
  if (!loadOp) {
    return {};
  }
  return IREE::Util::lookupGlobalOp(loadOp, loadOp.getGlobalAttr(),
                                    symbolTables);
}

Value getReshapeSource(Value value) {
  while (isa_and_present<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
      value.getDefiningOp())) {
    value = value.getDefiningOp()->getOperand(0);
  }
  return value;
}

std::optional<SmallVector<OpOperand *>>
getUsesOfSameData(Value value, const SymbolUserMap &symbolUsers,
                  SymbolTableCollection &symbolTables) {
  Value source = getReshapeSource(value);
  SmallVector<Value> worklist;
  if (IREE::Util::GlobalOpInterface globalOp =
          getLoadedGlobal(source, symbolTables)) {
    if (!globalOp.isGlobalPrivate()) {
      return std::nullopt;
    }
    for (Operation *user : symbolUsers.getUsers(globalOp)) {
      auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(user);
      if (!loadOp) {
        return std::nullopt;
      }
      worklist.push_back(loadOp.getLoadedGlobalValue());
    }
  } else {
    worklist.push_back(source);
  }
  SmallVector<OpOperand *> uses;
  while (!worklist.empty()) {
    Value data = worklist.pop_back_val();
    for (OpOperand &use : data.getUses()) {
      if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(use.getOwner())) {
        worklist.push_back(use.getOwner()->getResult(0));
      } else {
        uses.push_back(&use);
      }
    }
  }
  return uses;
}

/// Returns the attribute holding `value` when it is a constant or a load of an
/// immutable global that only loads use, or null otherwise. For a global this
/// is its initial value, which may be a parameter reference rather than data.
static Attribute getConstantAttr(Value value, const SymbolUserMap &symbolUsers,
                                 SymbolTableCollection &symbolTables) {
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    return attr;
  }
  IREE::Util::GlobalOpInterface globalOp = getLoadedGlobal(value, symbolTables);
  if (!globalOp || globalOp.isGlobalMutable() ||
      !llvm::all_of(symbolUsers.getUsers(globalOp),
                    llvm::IsaPred<IREE::Util::GlobalLoadOpInterface>)) {
    return {};
  }
  return globalOp.getGlobalInitialValue();
}

/// Returns the integers in `attr`, read as unsigned if `isUnsigned`, or nullopt
/// when `attr` holds no readable integers. A splat gives a single value.
static std::optional<SmallVector<APSInt>> getIntegerValues(Attribute attr,
                                                           bool isUnsigned) {
  using Values = std::optional<SmallVector<APSInt>>;
  auto toAPSInt = [&](const APInt &value) { return APSInt(value, isUnsigned); };
  return llvm::TypeSwitch<Attribute, Values>(attr)
      .Case([&](IntegerAttr intAttr) -> Values {
        return SmallVector<APSInt>{toAPSInt(intAttr.getValue())};
      })
      .Case([&](DenseIntElementsAttr denseAttr) -> Values {
        if (denseAttr.isSplat()) {
          return SmallVector<APSInt>{
              toAPSInt(denseAttr.getSplatValue<APInt>())};
        }
        return llvm::map_to_vector(denseAttr.getValues<APInt>(), toAPSInt);
      })
      .Case<DenseI8ResourceElementsAttr, DenseI16ResourceElementsAttr,
            DenseI32ResourceElementsAttr, DenseI64ResourceElementsAttr>(
          [&](auto resourceAttr) -> Values {
            unsigned bitWidth = resourceAttr.getType().getElementTypeBitWidth();
            auto values = resourceAttr.tryGetAsArrayRef();
            if (!values) {
              return std::nullopt;
            }
            return llvm::map_to_vector(*values, [&](auto value) {
              return toAPSInt(APInt(bitWidth, value, /*isSigned=*/true));
            });
          })
      .Default(std::nullopt);
}

std::optional<SmallVector<APSInt>>
getConstantIntegerValues(Value value, bool isUnsigned,
                         const SymbolUserMap &symbolUsers,
                         SymbolTableCollection &symbolTables) {
  Attribute attr =
      getConstantAttr(getReshapeSource(value), symbolUsers, symbolTables);
  if (!attr) {
    return std::nullopt;
  }
  return getIntegerValues(attr, isUnsigned);
}

bool isComputedFromInlineData(
    const IREE::Util::ConstExprAnalysis::ConstValueInfo &info,
    const SymbolUserMap &symbolUsers, SymbolTableCollection &symbolTables) {
  if (!info.isConstExpr()) {
    return false;
  }
  return llvm::all_of(info.roots, [&](Value root) {
    return isa_and_present<ElementsAttr, IntegerAttr, FloatAttr>(
        getConstantAttr(root, symbolUsers, symbolTables));
  });
}

} // namespace mlir::iree_compiler::GlobalOptimization
