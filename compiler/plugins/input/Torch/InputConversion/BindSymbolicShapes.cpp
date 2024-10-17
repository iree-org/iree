// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <limits>

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_BINDSYMBOLICSHAPESPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

// We aribtrarily say that unbounded dimensions in a torch program cannot
// exceed 53bits, making the maximum safe dimension 9007199254740991. The
// astute reader will note that this is also the maximum safe value in
// JavaScript, which also "happens" to be the largest mantissa value in a
// 64bit double. We need a maximum and in the absence of a better choice,
// with this one we are at least in good company.
static constexpr uint64_t MAX_DIM_VALUE = (static_cast<uint64_t>(1) << 53) - 1;

// Torch "binds" symbolic shape information to all tensors in the program
// which are not static. It does this by emitting side-effecting
// torch.bind_symbolic_shape ops which are backed by torch.symbolic_int ops
// which match 1:1 to terminal symbols in the Torch program.
//
// This is a somewhat different representation than we need in order to be
// usable within IREE:
//
//   1. We only want shape information and assertion at the boundaries where
//      they can come from runtime values of unknown lineage.
//   2. IREE operates in terms of index values and "binding" them to tensors
//      so that later dim lookups are memoized.
//   3. IREE's value analyses operate on real index SSA values, not "symbolic"
//      values that only exist in the ether.
//
// These constraints can only be met if we assume that all Torch symbols are
// "backed" by a dimension or argument, so just a free-floating relational
// symbol. Such "backed" symbols are the most dominant form of Torch programs,
// but it is possible to create them such that symbols do not relate to any
// one dimension (although this typically does not happen naturally at
// program boundaries). In this pass we assume that any such relational
// symbols are not actionable by us, and we therefore drop them. It is possible
// for the frontend or user to fix this situation, and we therefore assume
// that anyone who cares will have done so. These cases are emitted as warnings
// in this pass because they signal potential missed optimization opportunties
// that we would like to know about.
//
// The approach we use from here will roughly map a torch.bind_symbolic_shape
// op to a flow.tensor.tie_shape op, preserving only the needed dynamic
// dimensions. Dimensions will be derived from util ops which annotate
// constraints and relationships.
//
// All other bind_symbolic_shape ops will be dropped.
class BindSymbolicShapesPass final
    : public impl::BindSymbolicShapesPassBase<BindSymbolicShapesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
  }

  bool isEligibleBinding(Torch::BindSymbolicShapeOp bindOp) {
    auto operand = bindOp.getOperand();
    // Torch programs are single block and use structured control flow, so
    // presume this is an entrypoint.
    if (llvm::isa<BlockArgument>(operand))
      return true;

    // Mutable tensors can exist at the boundary and must be "copied" to a
    // vtensor prior to use. Therefore, we anchor on the point of copy.
    if (operand.getDefiningOp<Torch::CopyToValueTensorOp>())
      return true;

    return false;
  }

  struct SymbolInfo {
    SymbolInfo(Torch::SymbolicIntOp symbolDefOp) : symbolDefOp(symbolDefOp) {
      auto minVal = symbolDefOp.getMinValAttr();
      auto maxVal = symbolDefOp.getMaxValAttr();
      if (minVal && maxVal) {
        uint64_t minValInt = minVal.getValue().getZExtValue();
        uint64_t maxValInt =
            std::min(maxVal.getValue().getZExtValue(), MAX_DIM_VALUE);
        if (maxValInt >= minValInt) {
          // Note that in Torch, min values are "weird" because they encode
          // some special cases about broadcast behavior. Here we just discard
          // them, but in the future, there may be more to derive here.
          minMaxBounds = std::make_pair(1, maxValInt);
        }
      }
    }

    // Gets the canonical dim for this symbol, returning {} if there
    // is no canonical dim.
    Value getCanonicalDimValue(OpBuilder &builder) {
      if (canonicalDimValue)
        return canonicalDimValue;
      if (equalityDimInfos.empty())
        return {};
      canonicalDimValue = getEqualityDimValue(builder, 0);
      return canonicalDimValue;
    }

    // Gets the dim value for one of the entries in equalityDimInfos,
    // materializing an op if needed.
    Value getEqualityDimValue(OpBuilder &builder, unsigned index) {
      auto [producer, position] = equalityDimInfos[index];
      // Scrunch all dim ops up as far as they will go so that they can be
      // shared among any legal consumers.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(producer);
      Value dimValue =
          builder.create<tensor::DimOp>(producer.getLoc(), producer, position);
      return dimValue;
    }

    Operation *symbolDefOp;

    // If the symbol carries min/max bounds, note them here.
    std::optional<std::pair<int64_t, int64_t>> minMaxBounds;

    // All dimensions that should be considered equal by {producer_tensor,
    // position}. When materializing shape expressions, we always use the
    // first from this list so that simple SSA equality can be used across
    // the graph.
    SmallVector<std::pair<Value, unsigned>> equalityDimInfos;

    Value canonicalDimValue;
  };

  struct TensorBinding {
    Operation *bindOp;

    // Symbol ops that that bind to symbols of the affine map.
    llvm::SmallVector<Value> symbols;

    // The value (tensor) this binding annotates.
    Value annotatesValue;

    // Torch type of the annotated tensor.
    Torch::ValueTensorType torchType;

    // Corresponding builtin tensor type.
    RankedTensorType builtinTensorType;

    // The affine map representing the dimensions.
    AffineMap shapeMap;

    // When prepared, we convert from the torch type to builtin and back. This
    // is the back value. Our work gets done feeding into this.
    TorchConversion::FromBuiltinTensorOp rewrittenTorchOp;

    // Anchor op for building IR on native types.
    Operation *anchorOp = nullptr;

    // All dim materializations we were able to make. If all are defined once
    // processing is complete, then we can tie the shape. This will be fully
    // populated after the associateEqualityDims phase, and subsequent
    // materializations should take the first value so that all related shapes
    // anchor the same.
    llvm::SmallVector<Value> materializedDims;

    // Perform IR preparation for any bindings we may want to preserve.
    void prepare() {
      OpBuilder builder(bindOp);
      TorchConversion::ToBuiltinTensorOp builtinConversion;
      {
        // Scrunch all ToBuiltinTensor ops as high up as they can go. We'll
        // hang tensor.dim ops off of these across all dependent bindings so
        // we need to make sure that it is always topologically legal. The
        // easiest way to do this is to put common dependencies like this
        // as far up as they will go, which means that each binding op (which
        // is already guaranteed to be topologically legal) stays so.
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfterValue(annotatesValue);
        builtinConversion = builder.create<TorchConversion::ToBuiltinTensorOp>(
            bindOp->getLoc(), builtinTensorType, annotatesValue);
      }
      rewrittenTorchOp = builder.create<TorchConversion::FromBuiltinTensorOp>(
          bindOp->getLoc(), torchType, builtinConversion.getResult());
      annotatesValue.replaceAllUsesExcept(rewrittenTorchOp.getResult(),
                                          builtinConversion);
      annotatesValue = builtinConversion.getResult();
      anchorOp = rewrittenTorchOp;

      materializedDims.resize(builtinTensorType.getRank());
    }

    std::optional<std::pair<int64_t, int64_t>>
    evaluateExprBounds(AffineExpr expr,
                       llvm::DenseMap<Value, SymbolInfo> &symbolInfos) {
      if (!expr.isSymbolicOrConstant())
        return {};
      llvm::SmallVector<std::optional<int64_t>> lowerBounds;
      llvm::SmallVector<std::optional<int64_t>> upperBounds;
      lowerBounds.reserve(symbols.size());
      upperBounds.reserve(symbols.size());
      for (auto [pos, symbolValue] : llvm::enumerate(symbols)) {
        const SymbolInfo &symbolInfo = symbolInfos.at(symbolValue);
        if (!symbolInfo.minMaxBounds) {
          lowerBounds.push_back(1);
          upperBounds.push_back(MAX_DIM_VALUE);
        } else {
          lowerBounds.push_back(symbolInfo.minMaxBounds->first);
          upperBounds.push_back(symbolInfo.minMaxBounds->second);
        }
      }

      auto upperBound = getBoundForAffineExpr(
          expr, /*numDims=*/0, /*numSymbols=*/symbols.size(), lowerBounds,
          upperBounds, /*isUpper=*/true);
      if (!upperBound)
        return {};

      auto lowerBound = getBoundForAffineExpr(
          expr, /*numDims=*/0, /*numSymbols=*/symbols.size(), lowerBounds,
          upperBounds, /*isUpper=*/false);
      if (!lowerBound)
        return {};

      return std::make_pair(*lowerBound, *upperBound);
    }

    // For any dims in the shapeMap that are terminal, set up the root
    // bindings.
    void associateEqualityDims(llvm::DenseMap<Value, SymbolInfo> &symbolInfos) {
      OpBuilder builder(anchorOp);
      for (auto [index, expr] : llvm::enumerate(shapeMap.getResults())) {
        if (expr.getKind() != AffineExprKind::SymbolId)
          continue;
        auto symbolPos = llvm::cast<AffineSymbolExpr>(expr).getPosition();
        Value symbol = symbols[symbolPos];
        auto symbolInfoIt = symbolInfos.find(symbol);
        assert(symbolInfoIt != symbolInfos.end() &&
               "No symbol info for symbol");
        auto &symbolInfo = symbolInfoIt->second;
        symbolInfo.equalityDimInfos.emplace_back(annotatesValue, index);
      }
    }

    Value materializeDimExpr(Location loc, OpBuilder &builder,
                             AffineExpr genericExpr,
                             llvm::DenseMap<Value, SymbolInfo> &symbolInfos) {
      if (auto binaryExpr = llvm::dyn_cast<AffineBinaryOpExpr>(genericExpr)) {
        auto lhs =
            materializeDimExpr(loc, builder, binaryExpr.getLHS(), symbolInfos);
        if (!lhs)
          return {};
        auto rhs =
            materializeDimExpr(loc, builder, binaryExpr.getRHS(), symbolInfos);
        if (!rhs)
          return {};

        switch (binaryExpr.getKind()) {
        case AffineExprKind::Add:
          return builder.create<arith::AddIOp>(loc, lhs, rhs);
        case AffineExprKind::Mul:
          return builder.create<arith::MulIOp>(loc, lhs, rhs);
        case AffineExprKind::Mod:
          return builder.create<arith::RemUIOp>(loc, lhs, rhs);
        case AffineExprKind::FloorDiv:
          return builder.create<arith::DivUIOp>(loc, lhs, rhs);
        case AffineExprKind::CeilDiv:
          return builder.create<arith::CeilDivUIOp>(loc, lhs, rhs);
        default:
          break;
        }
      }

      switch (genericExpr.getKind()) {
      case AffineExprKind::Constant:
        return builder.create<arith::ConstantOp>(
            loc, builder.getIndexAttr(
                     llvm::cast<AffineConstantExpr>(genericExpr).getValue()));
      case AffineExprKind::DimId:
        // Unsupported.
        break;
      case AffineExprKind::SymbolId: {
        auto symExpr = llvm::cast<AffineSymbolExpr>(genericExpr);
        auto pos = symExpr.getPosition();
        if (pos >= symbols.size())
          break;
        Value symbolValue = symbols[pos];
        auto foundIt = symbolInfos.find(symbolValue);
        if (foundIt == symbolInfos.end())
          break;
        SymbolInfo &info = foundIt->second;
        return info.getCanonicalDimValue(builder); // May legally return {}
      }
      default:
        break;
      }

      std::string s;
      llvm::raw_string_ostream os(s);
      genericExpr.print(os);
      emitWarning(loc) << "Symbolic shape expression not supported: " << s
                       << " (falling back to runtime symbol resolution)";
      return {};
    }

    void materializeDims(llvm::DenseMap<Value, SymbolInfo> &symbolInfos) {
      OpBuilder builder(anchorOp);
      for (auto [index, expr] : llvm::enumerate(shapeMap.getResults())) {
        if (!builtinTensorType.isDynamicDim(index))
          continue;

        Value dimValue =
            materializeDimExpr(anchorOp->getLoc(), builder, expr, symbolInfos);
        if (!dimValue) {
          // Certain classes of symbolic expressions may not terminate on
          // distinct dimensions (i.e. `s0 * 4` with no symbol that corresponds)
          // to `s0`. In this case, we just do runtime resolution of the symbol.
          dimValue = builder.create<tensor::DimOp>(bindOp->getLoc(),
                                                   annotatesValue, index);
        }

        // Add optimization assumptions if the divisor or bounds are known.
        int64_t divisor = expr.getLargestKnownDivisor();
        auto bounds = evaluateExprBounds(expr, symbolInfos);
        std::optional<uint64_t> optionalUmin;
        std::optional<uint64_t> optionalUmax;
        std::optional<int64_t> optionalDivisor;
        if (bounds) {
          optionalUmin = bounds->first;
          optionalUmax = bounds->second;
        }
        if (divisor != 1) {
          optionalDivisor = divisor;
        }
        if (optionalUmin || optionalUmax || optionalDivisor) {
          auto assumption = builder.getAttr<IREE::Util::IntAssumptionAttr>(
              /*umin=*/optionalUmin,
              /*umax=*/optionalUmax,
              /*divisor=*/optionalDivisor);
          dimValue = builder
                         .create<IREE::Util::AssumeIntOp>(bindOp->getLoc(),
                                                          dimValue, assumption)
                         .getResult(0);
        }

        materializedDims[index] = dimValue;
      }
    }

    void tieShape(llvm::DenseMap<Value, SymbolInfo> &symbolInfos) {
      llvm::SmallVector<Value> dynamicDims;
      dynamicDims.reserve(materializedDims.size());
      for (size_t pos = 0; pos < materializedDims.size(); ++pos) {
        if (builtinTensorType.isDynamicDim(pos)) {
          Value dimValue = materializedDims[pos];
          if (!dimValue) {
            emitWarning(bindOp->getLoc())
                << "Discarding symbolic shape information from PyTorch: Not "
                << "all symbols resolved to a known dim value (first missing "
                << "at position " << pos << ")";
            return;
          }

          dynamicDims.push_back(dimValue);
        }
      }

      OpBuilder builder(anchorOp);
      Value tieShape = builder.create<IREE::Flow::TensorTieShapeOp>(
          bindOp->getLoc(), builtinTensorType, annotatesValue, dynamicDims);
      rewrittenTorchOp.setOperand(tieShape);
    }
  };

  void runOnOperation() override {
    ConversionTarget target(getContext());
    TypeConverter typeConverter;
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    llvm::SmallVector<Operation *> cleanupOpList;
    llvm::SmallVector<TensorBinding> bindings;
    // Mapping of SSA value for a torch.symbolic_int (or related op) to its
    // info.
    llvm::DenseMap<Value, SymbolInfo> symbolInfos;

    // Walk the ops we care about and stash for analysis.
    getOperation()->walk([&](Operation *childOp) {
      if (auto symbolOp = llvm::dyn_cast<Torch::SymbolicIntOp>(childOp)) {
        cleanupOpList.push_back(symbolOp);
        symbolInfos.insert_or_assign(symbolOp.getResult(),
                                     SymbolInfo(symbolOp));
      } else if (auto bindOp =
                     llvm::dyn_cast<Torch::BindSymbolicShapeOp>(childOp)) {
        cleanupOpList.push_back(bindOp);
        if (!isEligibleBinding(bindOp))
          return;
        auto torchType =
            llvm::cast<Torch::ValueTensorType>(bindOp.getOperand().getType());
        auto builtinType = llvm::dyn_cast_or_null<RankedTensorType>(
            typeConverter.convertType(torchType));
        if (!builtinType) {
          emitError(childOp->getLoc())
              << "cannot convert torch type to builtin: " << torchType;
          return signalPassFailure();
        }
        bindings.push_back(TensorBinding{
            /*bindOp=*/childOp,
            /*symbols=*/bindOp.getShapeSymbols(),
            /*annotatesValue=*/bindOp.getOperand(),
            /*torchType=*/torchType,
            /*builtinType=*/builtinType,
            /*shapeMap=*/bindOp.getShapeExpressions().getAffineMap()});
      }
    });

    // For every tensor value of interest, convert to a builtin tensor type and
    // back, RAUW'ing the result. This will meet the eventual final conversion
    // with additional graph forking.
    for (auto &binding : bindings) {
      binding.prepare();
    }

    // Find all associations to a single symbol and set up the roots.
    for (auto &binding : bindings) {
      binding.associateEqualityDims(symbolInfos);
    }

    // Materialize all dimension expressions and constraints.
    for (auto &binding : bindings) {
      binding.materializeDims(symbolInfos);
    }

    // Now that all is known, insert tie shape.
    for (auto &binding : bindings) {
      binding.tieShape(symbolInfos);
    }

    // Erase all found ops.
    for (auto *op : llvm::reverse(cleanupOpList)) {
      op->erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::TorchInput
