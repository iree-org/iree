// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-codegen-utils"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

std::optional<IREE::HAL::ExecutableExportOp>
getEntryPoint(mlir::FunctionOpInterface funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) {
    return std::nullopt;
  }

  for (auto op : variantOp.getExportOps()) {
    if (op.getSymName() == funcOp.getName()) {
      return op;
    }
  }
  return std::nullopt;
}

bool isEntryPoint(mlir::FunctionOpInterface func) {
  return func.isPublic() && getEntryPoint(func);
}

std::optional<StringAttr> getConfigStringAttr(Attribute srcAttr,
                                              StringRef stringAttr) {
  if (!srcAttr) {
    return std::nullopt;
  }
  auto targetAttr = dyn_cast<IREE::HAL::ExecutableTargetAttr>(srcAttr);
  DictionaryAttr config;
  if (targetAttr) {
    config = targetAttr.getConfiguration();
  } else {
    config = dyn_cast<DictionaryAttr>(srcAttr);
  }
  if (!config) {
    return std::nullopt;
  }
  auto attr = config.getAs<StringAttr>(stringAttr);
  if (!attr) {
    return std::nullopt;
  }
  return attr;
}

std::optional<IntegerAttr> getConfigIntegerAttr(Attribute srcAttr,
                                                StringRef integerAttr) {
  if (!srcAttr) {
    return std::nullopt;
  }
  auto targetAttr = dyn_cast<IREE::HAL::ExecutableTargetAttr>(srcAttr);
  DictionaryAttr config;
  if (targetAttr) {
    config = targetAttr.getConfiguration();
  } else {
    config = dyn_cast<DictionaryAttr>(srcAttr);
  }
  if (!config) {
    return std::nullopt;
  }
  auto attr = config.getAs<IntegerAttr>(integerAttr);
  if (!attr) {
    return std::nullopt;
  }
  return attr;
}

std::optional<BoolAttr> getConfigBoolAttr(Attribute srcAttr,
                                          StringRef boolAttr) {
  if (!srcAttr) {
    return std::nullopt;
  }
  auto targetAttr = dyn_cast<IREE::HAL::ExecutableTargetAttr>(srcAttr);
  DictionaryAttr config;
  if (targetAttr) {
    config = targetAttr.getConfiguration();
  } else {
    config = dyn_cast<DictionaryAttr>(srcAttr);
  }
  if (!config) {
    return std::nullopt;
  }
  auto attr = config.getAs<BoolAttr>(boolAttr);
  if (!attr) {
    return std::nullopt;
  }
  return attr;
}

std::optional<llvm::Triple> getTargetTriple(Attribute attr) {
  auto triple = getConfigStringAttr(attr, "target_triple");
  if (!triple) {
    return std::nullopt;
  }
  return llvm::Triple(triple.value().str());
}

const char *getIreeArchNameForTargetTriple(llvm::Triple triple) {
  if (triple.isX86()) {
    return triple.isArch64Bit() ? "x86_64" : "x86_32";
  }
  if (triple.isWasm()) {
    return triple.isArch64Bit() ? "wasm_64" : "wasm_32";
  }
  if (triple.isAArch64()) {
    return "arm_64";
  }
  if (triple.isARM()) {
    return "arm_32";
  }
  if (triple.isRISCV64()) {
    return "riscv_64";
  }
  if (triple.isRISCV32()) {
    return "riscv_32";
  }
  return "unknown";
}

bool isLLVMCPUBackend(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return targetAttr && targetAttr.getBackend().getValue() == "llvm-cpu";
}

bool isVMVXBackend(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return targetAttr && targetAttr.getBackend().getValue().starts_with("vmvx");
}

bool isROCMBackend(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return targetAttr && targetAttr.getBackend().getValue().starts_with("rocm");
}

bool isWebGPUBackend(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return targetAttr && targetAttr.getBackend().getValue().starts_with("webgpu");
}

static const char *getDefaultEnabledUkernels(Attribute attr) {
  const char *kNone = "none";
  if (!attr) {
    return kNone;
  }
  auto targetAttr = dyn_cast<IREE::HAL::ExecutableTargetAttr>(attr);
  if (!targetAttr) {
    return kNone;
  }
  if (isX86_64(targetAttr)) {
    return "mmt4d";
  }
  if (isAArch64(targetAttr)) {
    return "mmt4d";
  }
  return kNone;
}

bool hasUkernel(Attribute attr, StringRef ukernelName) {
  auto enabledUkernels = getConfigStringAttr(attr, "ukernels");
  StringRef enabledUkernelsStr;
  if (enabledUkernels) {
    enabledUkernelsStr = enabledUkernels->getValue();
  } else {
    enabledUkernelsStr = "default";
  }
  // Resolve `default`.
  if (enabledUkernelsStr == "default") {
    enabledUkernelsStr = getDefaultEnabledUkernels(attr);
  }
  // Resolve `none`.
  if (enabledUkernelsStr == "none") {
    return false;
  }
  // Resolve `all`.
  if (enabledUkernelsStr == "all") {
    return true;
  }
  // If `ukernelName` is empty, the question is "are ukernels enabled at all?"
  // At this point, we already know that enabledUkernelsStr != "none".
  if (ukernelName.empty()) {
    return !enabledUkernelsStr.empty();
  }
  while (!enabledUkernelsStr.empty()) {
    auto split = enabledUkernelsStr.split(',');
    if (split.first == ukernelName) {
      return true;
    }
    enabledUkernelsStr = split.second;
  }
  return false;
}

std::optional<StringRef> getCpuFeatures(Attribute attr) {
  auto cpuFeatures = getConfigStringAttr(attr, "cpu_features");
  if (!cpuFeatures) {
    return std::nullopt;
  }
  return cpuFeatures->getValue();
}

// TODO(dcaballe): If we have to check for a significantly large number of
// features in the future, we may want to consider a persistent state to carry
// over processed HAL information or keeping the TTI instance alive and query
// subtarget features data structure.
bool hasFeature(Attribute attr, StringRef feature) {
  std::optional<StringRef> features = getCpuFeatures(attr);
  if (!features) {
    return false;
  }

  // Find feature string in list of features, making sure that we don't match a
  // sub-string.
  std::stringstream sstream(features->str());
  std::string str;
  while (std::getline(sstream, str, ',')) {
    if (str == feature) {
      return true;
    }
  }

  return false;
}

bool isX86(Attribute attr) {
  std::optional<llvm::Triple> triple = getTargetTriple(attr);
  return triple && triple.value().isX86();
}

bool isX86_64(Attribute attr) {
  std::optional<llvm::Triple> triple = getTargetTriple(attr);
  return triple && triple.value().getArch() == llvm::Triple::x86_64;
}

bool isAArch64(Attribute attr) {
  std::optional<llvm::Triple> triple = getTargetTriple(attr);
  return triple && triple.value().isAArch64();
}

bool isRISCV(Attribute attr) {
  std::optional<llvm::Triple> triple = getTargetTriple(attr);
  return triple && triple.value().isRISCV();
}

bool isRISCV32(Attribute attr) {
  std::optional<llvm::Triple> triple = getTargetTriple(attr);
  return triple && triple.value().isRISCV32();
}

bool isReadOnly(Value v) {
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp)
    return false;
  return TypeSwitch<Operation *, bool>(definingOp)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return true; })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto op) { return isReadOnly(op.getSrc()); })
      .Case<tensor::CastOp, tensor::ExtractSliceOp>(
          [&](auto op) { return isReadOnly(op.getSource()); })
      .Case<IREE::TensorExt::DispatchTensorLoadOp>(
          [&](IREE::TensorExt::DispatchTensorLoadOp loadOp) {
            return llvm::cast<IREE::TensorExt::DispatchTensorType>(
                       loadOp.getSource().getType())
                       .getAccess() == IREE::TensorExt::TensorAccess::ReadOnly;
          })
      .Default([&](Operation *op) { return false; });
}

LogicalResult duplicateTensorEmptyOps(OpBuilder &b, tensor::EmptyOp emptyOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(emptyOp);
  SmallVector<OpOperand *> uses = llvm::map_to_vector(
      emptyOp->getUses(), [](OpOperand &use) { return &use; });
  for (auto use : llvm::make_range(std::next(uses.begin()), uses.end())) {
    auto newOp = cast<tensor::EmptyOp>(b.clone(*emptyOp.getOperation()));
    Operation *user = use->getOwner();
    user->setOperand(use->getOperandNumber(), newOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Setting CustomOp Lowering config.
//===----------------------------------------------------------------------===//

static std::tuple<SmallVector<Operation *>, SetVector<Value>>
getNonConstantValuesDefinedFromAbove(Region &region) {
  llvm::SetVector<Value> valuesDefinedFromAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedFromAbove);
  SmallVector<Operation *> constants;
  SetVector<Value> erasedVals;
  for (auto value : valuesDefinedFromAbove) {
    Attribute constVal;
    if (!matchPattern(value, m_Constant(&constVal))) {
      continue;
    }
    if (!isa<IntegerAttr, FloatAttr>(constVal)) {
      continue;
    }
    constants.push_back(value.getDefiningOp());
    erasedVals.insert(value);
  }
  valuesDefinedFromAbove.set_subtract(erasedVals);
  return {constants, valuesDefinedFromAbove};
}

/// Listener to track mapping from operations in the body of a cloned custom op
/// back to the original operations in the body of the original custom op.
class CustomOpConfigListener : public RewriterBase::Listener {
public:
  CustomOpConfigListener(IREE::LinalgExt::CustomOp origCustomOp,
                         IREE::LinalgExt::CustomOp clonedCustomOp) {
    for (auto [origOp, clonedOp] :
         llvm::zip_equal(origCustomOp.getBody()->without_terminator(),
                         clonedCustomOp.getBody()->without_terminator())) {
      clonedOpToOrigOp[&clonedOp] = &origOp;
    }
  }
  void notifyOperationErased(Operation *op) override {
    clonedOpToOrigOp.erase(op);
  }
  void notifyOperationReplaced(Operation *op, Operation *replacement) override {
    auto it = clonedOpToOrigOp.find(op);
    if (it != clonedOpToOrigOp.end()) {
      Operation *origOp = it->second;
      clonedOpToOrigOp.erase(it);
      clonedOpToOrigOp[replacement] = origOp;
    }
  }
  void notifyOperationReplaced(Operation *op,
                               ValueRange replacements) override {
    Operation *replacementOp = nullptr;
    for (auto val : replacements) {
      Operation *definingOp = getDefiningOp(val);
      if (!definingOp) {
        // One of the replacements is definitely not from an op. Bail
        // immediately.
        return;
      }
      if (replacementOp) {
        if (definingOp != replacementOp) {
          // No consistent replacementOp. Bail.
          return;
        }
      } else {
        replacementOp = definingOp;
      }
    }
    if (replacementOp && replacementOp->getName() == op->getName()) {
      notifyOperationReplaced(op, replacementOp);
    }
  }

  // Helper methods to get back the orig op for the cloned op.
  std::optional<Operation *> getOrigOp(Operation *clonedOp) {
    auto it = clonedOpToOrigOp.find(clonedOp);
    if (it == clonedOpToOrigOp.end()) {
      return std::nullopt;
    }
    return it->second;
  }

private:
  llvm::MapVector<Operation *, Operation *> clonedOpToOrigOp;

  /// On cast propagation, the replacement value used is not the
  /// actual op that is used for replacement. Walk back the replacement
  /// value use-def chain to get to the real replacement. This is a
  /// bit of a hack, but the lowering config propagation is really
  /// best effort, so not incorrect.
  Operation *getDefiningOp(Value v) {
    Operation *definingOp = v.getDefiningOp();
    while (definingOp) {
      if (auto castOp = dyn_cast<tensor::CastOp>(definingOp)) {
        definingOp = castOp.getSource().getDefiningOp();
        continue;
      }
      // Default is to break out of the loop.
      break;
    }
    return definingOp;
  }
};

LogicalResult setDefaultCustomOpLoweringConfig(
    FunctionOpInterface funcOp, IREE::LinalgExt::CustomOp customOp,
    std::function<LogicalResult(FunctionOpInterface)> configFn) {

  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(funcOp);

  // 1. Get values captured from above in the custom op region.
  llvm::SetVector<Value> valuesDefinedAbove;
  SmallVector<Operation *> constantOps;
  std::tie(constantOps, valuesDefinedAbove) =
      getNonConstantValuesDefinedFromAbove(customOp.getRegion());

  // 2. Create an empty function with arguments being the operands of the custom
  // op and values captured from above in the custom op.
  auto operandTypes = llvm::to_vector(customOp->getOperandTypes());
  auto valuesDefinedAboveTypes =
      llvm::map_range(valuesDefinedAbove, [](Value v) { return v.getType(); });
  operandTypes.append(valuesDefinedAboveTypes.begin(),
                      valuesDefinedAboveTypes.end());
  auto dummyFuncType =
      FunctionType::get(context, operandTypes, customOp->getResultTypes());
  std::string dummyFuncName =
      std::string("__") + funcOp.getName().str() + "_config_setting__";
  auto dummyFuncOp = rewriter.create<func::FuncOp>(
      customOp.getLoc(), dummyFuncName, dummyFuncType);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (targetAttr) {
    dummyFuncOp->setAttr(IREE::HAL::ExecutableTargetAttr::name, targetAttr);
  }

  // 3. Clone the custom op into the function
  SmallVector<Location> locs = llvm::map_to_vector(
      customOp->getOperands(), [](Value v) { return v.getLoc(); });
  auto valuesDefinedAboveLocs =
      llvm::map_range(valuesDefinedAbove, [](Value v) { return v.getLoc(); });
  locs.append(valuesDefinedAboveLocs.begin(), valuesDefinedAboveLocs.end());
  Block *body =
      rewriter.createBlock(&dummyFuncOp.getRegion(),
                           dummyFuncOp.getRegion().begin(), operandTypes, locs);
  rewriter.setInsertionPointToStart(body);
  IRMapping map;
  map.map(customOp.getOperands(),
          body->getArguments().take_front(customOp.getNumOperands()));
  map.map(valuesDefinedAbove.getArrayRef(),
          body->getArguments().take_back(valuesDefinedAbove.size()));
  for (auto op : constantOps) {
    rewriter.clone(*op, map);
  }
  auto clonedCustomOp = cast<IREE::LinalgExt::CustomOp>(
      rewriter.clone(*customOp.getOperation(), map));
  rewriter.create<func::ReturnOp>(customOp.getLoc(),
                                  clonedCustomOp->getResults());
  CustomOpConfigListener customOpConfigListener(customOp, clonedCustomOp);

  // 4. Inline the cloned custom op.
  rewriter.setInsertionPoint(clonedCustomOp);
  FailureOr<SmallVector<Value>> replacements =
      clonedCustomOp.decomposeOperation(rewriter);
  if (failed(replacements)) {
    return customOp.emitOpError(
        "failed to decompose op during custom op configuration setting");
  }
  rewriter.replaceOp(clonedCustomOp, replacements.value());

  // 5. Run canonicalizations on the created function to constant propagate the
  // shape.
  RewritePatternSet patterns(context);
  auto addCanonicalizationPatterns = [&context,
                                      &patterns](StringRef dialectName) {
    context->getLoadedDialect(dialectName)
        ->getCanonicalizationPatterns(patterns);
  };
  addCanonicalizationPatterns(linalg::LinalgDialect::getDialectNamespace());
  addCanonicalizationPatterns(
      IREE::LinalgExt::IREELinalgExtDialect::getDialectNamespace());
  tensor::CastOp::getCanonicalizationPatterns(patterns, context);
  addCanonicalizationPatterns(tensor::TensorDialect::getDialectNamespace());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  GreedyRewriteConfig config;
  config.setListener(&customOpConfigListener);
  if (failed(applyPatternsGreedily(dummyFuncOp, std::move(patterns), config))) {
    return customOp.emitOpError(
        "failed to canonicalize during custom op configuration setting");
  }

  // 6. Run set configuration on the new dummy function.
  if (failed(configFn(dummyFuncOp))) {
    return customOp.emitOpError("failed to set configuration for custom op");
  }

  // 7. Set translation info and lowering config for the custom op.
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(dummyFuncOp);
  // Move lowering config from ops in the cloned function to the ops
  // within the body of the custom op.
  // TODO: This logic needs to be made more robust (by account for indexing maps
  // specified for operands on the custom op and the indexing maps of the
  // operations within the region of the custom op). For now, just use the first
  // operation with lowering config.
  std::optional<SmallVector<int64_t>> workgroupTileSizes;
  std::optional<SmallVector<int64_t>> workgroupInterchange;
  for (Operation &op : dummyFuncOp.getBody().front()) {
    auto currLoweringConfig =
        getLoweringConfig<IREE::Codegen::LoweringConfigAttrInterface>(&op);
    if (!currLoweringConfig)
      continue;

    // Translate the lowering config to the original operation.
    if (std::optional<Operation *> originalOperation =
            customOpConfigListener.getOrigOp(&op)) {
      setLoweringConfig(originalOperation.value(), currLoweringConfig);
    }

    auto currWorkgroupTileSizes = currLoweringConfig.getWorkgroupTileSizes();
    if (currWorkgroupTileSizes.empty())
      continue;
    workgroupTileSizes = currWorkgroupTileSizes;
    workgroupInterchange = currLoweringConfig.getWorkgroupInterchange();
  }
  IREE::Codegen::LoweringConfigAttr loweringConfig;
  if (workgroupTileSizes) {
    loweringConfig = IREE::Codegen::LoweringConfigAttr::get(
        context, workgroupTileSizes.value_or(SmallVector<int64_t>{}),
        workgroupInterchange.value_or(SmallVector<int64_t>{}));
  }
  if (failed(setOpConfigAndEntryPointFnTranslation(
          funcOp, customOp, loweringConfig, translationInfo))) {
    return funcOp.emitOpError("failed to set custom op configuration");
  }
  rewriter.eraseOp(dummyFuncOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions to set configurations
//===----------------------------------------------------------------------===//

/// Returns the first of `exprs` which is of the type `T`.
template <typename T>
static AffineExpr getAffineExprOfType(ArrayRef<AffineExpr> exprs) {
  if (auto it = llvm::find_if(exprs, llvm::IsaPred<T>); it != exprs.end())
    return *it;
  return nullptr;
}

/// Returns a Value that represents the value for symbol or dim expr for the map
/// in the `applyOp`.
static Value getValueForDimOrSymbol(affine::AffineApplyOp applyOp,
                                    AffineExpr expr) {
  unsigned numDims = applyOp.getAffineMap().getNumDims();
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return applyOp.getOperand(dimExpr.getPosition());
  }
  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    return applyOp.getOperand(numDims + symbolExpr.getPosition());
  }
  return nullptr;
}
static SmallVector<Value>
getValuesForDimsOrSymbols(affine::AffineApplyOp applyOp,
                          ArrayRef<AffineExpr> exprs) {
  SmallVector<Value> vals;
  for (auto expr : exprs) {
    vals.push_back(getValueForDimOrSymbol(applyOp, expr));
  }
  return vals;
}

/// Returns the dimension for any operation that implements processor op
/// interfaces.
template <typename T>
static std::optional<unsigned> getDimension(Operation *op) {
  if (auto tOp = dyn_cast<T>(op)) {
    return tOp.getDimIndex();
  }
  return std::nullopt;
}
template <typename T1, typename T2, typename... T3>
static std::optional<unsigned> getDimension(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto dimension = getDimension<T1>(op)) {
    return dimension;
  }
  return getDimension<T2, T3...>(op);
}

/// Checks that all `vals` are defined by some processor id/count/size ops using
/// the same `dimension`. If any element of `vals` is not defined by one of
/// these ops, or the dimensions dont match, returns std::nullopt; oterhwise,
/// returns the dimension.  If `refDimension` is passed checks if the dimension
/// matches the given value.
template <typename... T>
static std::optional<unsigned>
checkDimensions(ArrayRef<Value> vals,
                std::optional<unsigned> refDimension = std::nullopt) {
  for (auto v : vals) {
    auto currDimension = getDimension<T...>(v.getDefiningOp());
    if (!currDimension)
      return std::nullopt;
    if (refDimension) {
      if (refDimension.value() != currDimension.value()) {
        return std::nullopt;
      }
    } else {
      refDimension = currDimension.value();
    }
  }
  return refDimension;
}

namespace {
/// Visitor to walk `lb` of a distributed loop. Expected the expression to be of
/// the form `a + b * c`, where `a` is the original `lb` and `b`, `c` are either
/// hal.interface.workgroup.id or hal.interface.workgroup.size.
class LowerBoundExprVisitor
    : public AffineExprVisitor<LowerBoundExprVisitor, LogicalResult> {
public:
  LowerBoundExprVisitor(affine::AffineApplyOp applyOp,
                        LoopTilingAndDistributionInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitAddExpr(AffineBinaryOpExpr expr) {
    AffineExpr offsetExpr =
        getAffineExprOfType<AffineBinaryOpExpr>({expr.getLHS(), expr.getRHS()});
    if (!offsetExpr) {
      // One of the expressions has to be a binary op expr.
      return failure();
    }
    // The other expression must be the undistributed `lb`.
    AffineExpr lbExpr =
        (offsetExpr == expr.getLHS() ? expr.getRHS() : expr.getLHS());
    if (isa<AffineDimExpr, AffineSymbolExpr>(lbExpr)) {
      Value v = getValueForDimOrSymbol(applyOp, lbExpr);
      if (!v) {
        return failure();
      }
      loopInfo.untiledLowerBound = getAsOpFoldResult(v);
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(lbExpr)) {
      loopInfo.untiledLowerBound = IntegerAttr::get(
          IndexType::get(applyOp.getContext()), constExpr.getValue());
    } else {
      return failure();
    }
    return visit(offsetExpr);
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    SmallVector<Value> vals;
    std::optional<unsigned> dimension;
    // workgroupSizeOp may have been folded into a constant expression.
    if (auto wgSize = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      vals = getValuesForDimsOrSymbols(applyOp, {expr.getLHS()});
      if (vals.size() != 1 || !vals[0]) {
        return failure();
      }
      loopInfo.tileSize = wgSize.getValue();
      dimension = checkDimensions<ProcessorIDInterface>(vals);
    } else {
      vals = getValuesForDimsOrSymbols(applyOp, {expr.getLHS(), expr.getRHS()});
      if (vals.size() != 2 || !vals[0] || !vals[1]) {
        return failure();
      }
      IntegerAttr tileSizeAttr;
      if (matchPattern(vals[1], m_Constant(&tileSizeAttr))) {
        loopInfo.tileSize = tileSizeAttr.getInt();
        dimension = checkDimensions<ProcessorIDInterface>(vals[0]);
      } else {
        dimension =
            checkDimensions<ProcessorIDInterface, ProcessorTileSizeInterface>(
                vals);
      }
    }
    if (!dimension) {
      return failure();
    }
    loopInfo.processorDistributionDim = dimension.value();
    if (!loopInfo.untiledLowerBound) {
      loopInfo.untiledLowerBound =
          IntegerAttr::get(IndexType::get(applyOp.getContext()), 0);
    }
    return success();
  }

private:
  affine::AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};

/// Visitor to walk the `step` of a distributed loop. Expected the expression to
/// be of the form `a * b * c`, where they could be the dynamic `step` or
/// defined by `hal.interface.workgroup.size`/`hal.interface.workgroup.count`
/// operation.
class StepExprVisitor
    : public AffineExprVisitor<StepExprVisitor, LogicalResult> {
public:
  StepExprVisitor(affine::AffineApplyOp applyOp,
                  LoopTilingAndDistributionInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    // Check if one of the operands is a binary op expr.
    SmallVector<AffineExpr> sentinels;
    if (auto e = getAffineExprOfType<AffineBinaryOpExpr>(
            {expr.getLHS(), expr.getRHS()})) {
      AffineExpr otherExpr =
          (e == expr.getLHS() ? expr.getRHS() : expr.getLHS());
      if (failed(processSentinel(otherExpr, sentinels))) {
        return failure();
      }
      expr = cast<AffineBinaryOpExpr>(e);
    } else {
      // Check if the workgroup tile size is folded into the affine map itself.
      if (loopInfo.tileSize) {
        if (auto stepCst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
          loopInfo.untiledStep =
              IntegerAttr::get(IndexType::get(applyOp.getContext()),
                               stepCst.getValue() / *loopInfo.tileSize);
        } else {
          auto stepValue = getValueForDimOrSymbol(applyOp, expr.getRHS());
          IntegerAttr tileSizeAttr;
          if (stepValue && matchPattern(stepValue, m_Constant(&tileSizeAttr))) {
            loopInfo.untiledStep =
                IntegerAttr::get(IndexType::get(applyOp.getContext()),
                                 tileSizeAttr.getInt() / *loopInfo.tileSize);
          }
        }
      } else {
        loopInfo.untiledStep =
            IntegerAttr::get(IndexType::get(applyOp.getContext()), 1);
      }
    }

    if (failed(processSentinel(expr.getLHS(), sentinels)) ||
        (!loopInfo.tileSize &&
         failed(processSentinel(expr.getRHS(), sentinels)))) {
      return failure();
    }
    // Either there are 3 sentinels and step isnt set, or there are two
    // sentinels and the step is set.
    if (sentinels.size() == 3) {
      if (loopInfo.untiledStep) {
        return failure();
      }
      auto it = sentinels.begin();
      for (auto ie = sentinels.end(); it != ie; ++it) {
        Value v = getValueForDimOrSymbol(applyOp, *it);
        if (!v.getDefiningOp<IREE::HAL::InterfaceWorkgroupSizeOp>() &&
            !v.getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
          loopInfo.untiledStep = getAsOpFoldResult(v);
          break;
        }
      }
      if (it != sentinels.end()) {
        sentinels.erase(it);
      }
    }

    if ((sentinels.size() != 2 || !loopInfo.untiledStep) &&
        (sentinels.size() != 1 || !loopInfo.tileSize)) {
      return failure();
    }
    SmallVector<Value> vals = getValuesForDimsOrSymbols(applyOp, sentinels);

    if ((loopInfo.tileSize && !checkDimensions<ProcessorCountInterface>(
                                  vals, loopInfo.processorDistributionDim)) ||
        (!loopInfo.tileSize &&
         !checkDimensions<ProcessorCountInterface, ProcessorTileSizeInterface>(
             vals, loopInfo.processorDistributionDim))) {
      return failure();
    }
    return success();
  }

private:
  LogicalResult processSentinel(AffineExpr e,
                                SmallVectorImpl<AffineExpr> &sentinels) {
    if (isa<AffineDimExpr, AffineSymbolExpr>(e)) {
      sentinels.push_back(e);
      return success();
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
      if (loopInfo.untiledStep) {
        return failure();
      }
      loopInfo.untiledStep = IntegerAttr::get(
          IndexType::get(applyOp.getContext()), constExpr.getValue());
      return success();
    }
    return failure();
  }

  affine::AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};
} // namespace

template <typename OpTy>
static std::optional<unsigned> getInterfaceWorkgroupOpDim(Value value) {
  if (auto op = value.getDefiningOp<OpTy>()) {
    return op.getDimension().getZExtValue();
  }
  return std::nullopt;
}

/// Checks if the `forOp` is a tiled + distributed op. Looks for the op of this
/// form
/// ```
///   %dim = arith.constant ... : index
///   %id = stream.dispatch.workgroup.id[%dim]
///   %count = stream.dispatch.workgroup.count[%dim]
///   %size = stream.dispatch.workgroup.size[%dim]
///   %offset = affine.apply
///     affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>(%lb)[%id, %size]
///   %new_step = affine.apply
///     affine_map<(d0)[s0, s1] -> (d0 * s0 * s1)>(%step)[%id, %size]
///   scf.for %iv = %offset to %ub step %new_step { ... }
/// ```
std::optional<LoopTilingAndDistributionInfo>
isTiledAndDistributedLoop(scf::ForOp forOp) {
  LoopTilingAndDistributionInfo loopInfo;
  loopInfo.loop = forOp;
  loopInfo.untiledUpperBound = getAsOpFoldResult(forOp.getUpperBound());

  auto lbApplyOp = forOp.getLowerBound().getDefiningOp<affine::AffineApplyOp>();
  auto stepApplyOp = forOp.getStep().getDefiningOp<affine::AffineApplyOp>();

  if (!lbApplyOp || !stepApplyOp) {
    // Try to see if this is a specical case where we have:
    //   scf.for %iv = %id to %ub step %count
    std::optional<unsigned> idDim;
    if (auto ifx = dyn_cast_or_null<ProcessorIDInterface>(
            forOp.getLowerBound().getDefiningOp())) {
      idDim = ifx.getDimIndex();
    }

    std::optional<unsigned> countDim;
    if (auto ifx = dyn_cast_or_null<ProcessorCountInterface>(
            forOp.getStep().getDefiningOp())) {
      countDim = ifx.getDimIndex();
    }

    if (!idDim || !countDim)
      return std::nullopt;

    Builder b(forOp.getContext());
    loopInfo.untiledLowerBound = b.getIndexAttr(0);
    loopInfo.untiledStep = b.getIndexAttr(1);
    loopInfo.processorDistributionDim = idDim.value();
    // For such special case, the tile size is 1.
    loopInfo.tileSize = 1;
    return loopInfo;
  }

  LowerBoundExprVisitor lbVisitor(lbApplyOp, loopInfo);
  StepExprVisitor stepVisitor(stepApplyOp, loopInfo);

  if (failed(lbVisitor.visit(lbApplyOp.getAffineMap().getResults()[0]))) {
    return std::nullopt;
  }
  if (failed(stepVisitor.visit(stepApplyOp.getAffineMap().getResults()[0]))) {
    return std::nullopt;
  }
  if (!loopInfo.untiledLowerBound || !loopInfo.untiledStep) {
    return std::nullopt;
  }
  return loopInfo;
}

SmallVector<Operation *> getComputeOps(Operation *containingOp) {
  if (containingOp->getNumRegions() == 0) {
    return {};
  }
  assert(containingOp->getNumRegions() == 1 &&
         "expected op with a single region");
  SmallVector<Operation *> computeOps;
  containingOp->getRegion(0).walk([&](Operation *op) {
    if (isa<TilingInterface, IREE::Codegen::UKernelOpInterface>(op)) {
      computeOps.push_back(op);
    }
  });
  return computeOps;
}

SmallVector<LoopTilingAndDistributionInfo>
getTiledAndDistributedLoopInfo(mlir::FunctionOpInterface funcOp) {
  SmallVector<LoopTilingAndDistributionInfo> info;
  funcOp.walk([&](scf::ForOp forOp) {
    if (auto tiledLoopInfo = isTiledAndDistributedLoop(forOp)) {
      info.emplace_back(std::move(tiledLoopInfo.value()));
    }
  });
  return info;
}

void setSCFTileSizes(scf::SCFTilingOptions &options, TilingInterface op,
                     ArrayRef<int64_t> tileSizes,
                     ArrayRef<bool> tileScalableFlags) {
  // scf::tileUsingSCFForOp expects the num of tile sizes = num of loops.
  int numLoops = op.getLoopIteratorTypes().size();
  SmallVector<int64_t> fixedTileSizes(tileSizes);
  fixedTileSizes.resize(numLoops, /*default=*/0);
  SmallVector<bool> fixedTileScalableFlags(tileScalableFlags);
  fixedTileScalableFlags.resize(numLoops, /*default=*/false);
  if (!llvm::is_contained(fixedTileScalableFlags, true)) {
    // Non-scalable case: All constant tile sizes.
    options.setTileSizes(
        getAsIndexOpFoldResult(op.getContext(), fixedTileSizes));
  } else {
    // Scalable case: Multiply scalable tile sizes by a vector.vscale op.
    options.setTileSizeComputationFunction(
        [=](OpBuilder &b, Operation *op) -> SmallVector<OpFoldResult> {
          auto loc = op->getLoc();
          return llvm::map_to_vector(
              llvm::zip(fixedTileSizes, fixedTileScalableFlags),
              [&](auto pair) -> OpFoldResult {
                auto [t, isScalable] = pair;
                Value size = b.create<arith::ConstantIndexOp>(loc, t);
                if (isScalable) {
                  Value vscale = b.create<vector::VectorScaleOp>(loc);
                  size = b.create<arith::MulIOp>(loc, size, vscale);
                }
                return size;
              });
        });
  }
}

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes) {
  auto memrefTypeFrom = llvm::dyn_cast<MemRefType>(from.getType());
  auto memrefTypeTo = llvm::dyn_cast<MemRefType>(to.getType());
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

template <typename OpTy>
static Value buildHALWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

linalg::LinalgLoopDistributionOptions getIREELinalgLoopDistributionOptions(
    linalg::DistributionMethod distributionMethod,
    int32_t maxWorkgroupParallelDims) {
  return {[distributionMethod,
           maxWorkgroupParallelDims](OpBuilder &builder, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
    auto numParallelDims = parallelLoopRanges.size();

    SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
    std::optional<Value> splitDim;
    SmallVector<OpFoldResult> splitNumTiles;
    for (size_t dim = 0; dim < numParallelDims; ++dim) {
      if (numParallelDims > maxWorkgroupParallelDims &&
          dim >= maxWorkgroupParallelDims - 1) {
        if (!splitDim) {
          splitDim = buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
              builder, maxWorkgroupParallelDims - 1);
        }
        OpFoldResult size = parallelLoopRanges[numParallelDims - dim - 1].size;
        OpFoldResult offset =
            parallelLoopRanges[numParallelDims - dim - 1].offset;
        OpFoldResult step =
            parallelLoopRanges[numParallelDims - dim - 1].stride;
        AffineExpr d0, d1, d2;
        bindSymbols(builder.getContext(), d0, d1, d2);
        OpFoldResult numTiles = affine::makeComposedFoldedAffineApply(
            builder, loc, (d1 - d0).ceilDiv(d2), {offset, size, step});
        splitNumTiles.push_back(numTiles);
        continue;
      }
      procInfo[numParallelDims - dim - 1] = {
          buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(builder,
                                                                     dim),
          buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupCountOp>(builder,
                                                                        dim),
          distributionMethod};
    }
    if (splitDim) {
      std::reverse(splitNumTiles.begin(), splitNumTiles.end());
      auto delinearized = builder.create<affine::AffineDelinearizeIndexOp>(
          loc, *splitDim, splitNumTiles, /*hasOuterBound=*/true);
      for (auto [i, id, numTiles] :
           llvm::enumerate(delinearized.getResults(), splitNumTiles)) {
        // We iterate the delinearize results from slowest up to fastest, and
        // we know that these are all the highest values of dimension. That is,
        // `i = 0` corresponds to the `numParallelDims - 1`-th dimension.
        procInfo[i] = {id,
                       getValueOrCreateConstantIndexOp(builder, loc, numTiles),
                       distributionMethod};
      }
    }
    return procInfo;
  }};
}

static constexpr char pipeliningDepthName[] = "pipeline_depth";
static constexpr char pipeliningStageName[] = "store_stage";

DictionaryAttr
getSoftwarePipeliningAttrDict(MLIRContext *context,
                              unsigned softwarePipelineDepth,
                              unsigned softwarePipelineStoreStage) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      {StringAttr::get(context, pipeliningDepthName),
       IntegerAttr::get(IntegerType::get(context, 64), softwarePipelineDepth)});
  attrs.push_back({StringAttr::get(context, pipeliningStageName),
                   IntegerAttr::get(IntegerType::get(context, 64),
                                    softwarePipelineStoreStage)});
  return DictionaryAttr::get(context, attrs);
}

FailureOr<int64_t> getSoftwarePipelineDepth(DictionaryAttr config) {
  if (!config) {
    return failure();
  }
  Attribute depth = config.get(pipeliningDepthName);
  if (!depth) {
    return failure();
  }
  return llvm::cast<IntegerAttr>(depth).getInt();
}

FailureOr<int64_t> getSoftwarePipelineStoreStage(DictionaryAttr config) {
  if (!config) {
    return failure();
  }
  Attribute stage = config.get(pipeliningStageName);
  if (!stage) {
    return failure();
  }
  return llvm::cast<IntegerAttr>(stage).getInt();
}

/// Returns a small tiling factor for the given reduction `dimSize`.
/// Returns 0 to avoid tiling.
int getReductionTilingFactor(int64_t dimSize) {
  if (dimSize % 4 == 0)
    return 4;

  // Try to find the smallest prime factor as the tiling factor. As a trade off
  // between generated code size and compilation time, only look at prime
  // numbers less than 50 right now.
  static constexpr std::array<int, 15> primeNumbers = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  for (int n : primeNumbers) {
    if (dimSize % n == 0)
      return n;
  }

  return 1; // Otherwise just tile with size 1.
}

int64_t getMinElementBitwidth(linalg::LinalgOp linalgOp) {
  unsigned bitwidth = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    unsigned b =
        IREE::Util::getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    bitwidth = std::min(bitwidth, b);
  }
  for (Value result : linalgOp.getDpsInits()) {
    unsigned b = IREE::Util::getTypeBitWidth(getElementTypeOrSelf(result));
    bitwidth = std::min(bitwidth, b);
  }
  return bitwidth;
};

//===---------------------------------------------------------------------===//
// Bufferization utility functions
//===---------------------------------------------------------------------===//

/// Get strides for row-major oredering of a tensor with the given `shape`.
static SmallVector<int64_t> getStridesFromShape(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return {};
  }
  SmallVector<int64_t> strides(shape.size(), ShapedType::kDynamic);
  strides.back() = 1;
  for (int i = strides.size() - 1; i > 0; --i) {
    if (ShapedType::isDynamic(shape[i])) {
      break;
    }
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

Value findOrCreateSubspanBuffer(
    RewriterBase &rewriter, IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  auto shapedType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
      subspanOp.getResult().getType());
  assert((shapedType && shapedType.hasRank()) &&
         "expected the result of subspanOp is DispatchTensorType");

  Value byteOffset = subspanOp.getByteOffset();
  MemRefLayoutAttrInterface layoutAttr = {};
  if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
    // Using buffer resources on AMDGPU will require buffers to be relocated to
    // offset 0, so any static offset we can compute here might change.
    // Therefore, always use a ? for the offset field unless it's known to be 0.
    auto tensorType = llvm::cast<RankedTensorType>(shapedType.getBoundType());
    SmallVector<int64_t> strides = getStridesFromShape(tensorType.getShape());
    layoutAttr = StridedLayoutAttr::get(rewriter.getContext(),
                                        ShapedType::kDynamic, strides);
  }
  bool useRocdlBuffers = false;
  if (auto *ireeGpuDialect =
          rewriter.getContext()
              ->getLoadedDialect<IREE::GPU::IREEGPUDialect>()) {
    useRocdlBuffers =
        ireeGpuDialect->getUseRocdlBufferInstructionsAttrHelper().isAttrPresent(
            subspanOp);
  }
  Attribute memorySpace = rewriter.getAttr<IREE::HAL::DescriptorTypeAttr>(
      subspanOp.getDescriptorType());
  auto memRefType =
      MemRefType::get(shapedType.getShape(), shapedType.getBoundElementType(),
                      layoutAttr, memorySpace);

  // Look for an existing op.
  Block *block = subspanOp->getBlock();
  for (Operation &op : *block) {
    if (&op == subspanOp.getOperation())
      break;
    auto bufferSubspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(&op);
    if (!bufferSubspanOp)
      continue;

    auto bufferMemrefType =
        llvm::dyn_cast<MemRefType>(bufferSubspanOp.getResult().getType());
    if (!bufferMemrefType)
      continue;

    if (bufferSubspanOp.getBinding() != subspanOp.getBinding() ||
        bufferSubspanOp.getDescriptorType() != subspanOp.getDescriptorType() ||
        bufferSubspanOp.getByteOffset() != subspanOp.getByteOffset() ||
        !llvm::equal(bufferSubspanOp.getDynamicDims(),
                     subspanOp.getDynamicDims()) ||
        bufferSubspanOp.getAlignment() != subspanOp.getAlignment() ||
        memRefType != bufferMemrefType)
      continue;

    if (useRocdlBuffers && bufferSubspanOp->hasOneUse()) {
      auto castOp = llvm::dyn_cast<amdgpu::FatRawBufferCastOp>(
          *bufferSubspanOp->getUsers().begin());
      if (!castOp)
        continue;
      return castOp.getResult();
    }
    return bufferSubspanOp.getResult();
  }

  // None found, create a new op.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(subspanOp);
  // Just change the result type of the InterfaceBindingSubspanOp.
  Value buffer = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
      subspanOp->getLoc(), memRefType, subspanOp.getLayout(),
      subspanOp.getBinding(), subspanOp.getByteOffset(),
      subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
      subspanOp.getDescriptorFlagsAttr());
  if (useRocdlBuffers) {
    buffer = rewriter.create<amdgpu::FatRawBufferCastOp>(
        subspanOp->getLoc(), buffer, /*validBytes=*/Value{},
        /*cacheSwizzleStride=*/Value{}, /*boundsCheck=*/true,
        /*resetOffset=*/true);
  }
  return buffer;
}

//===---------------------------------------------------------------------===//
// Misc. utility functions
//===---------------------------------------------------------------------===//

Operation *setInsertionPointAfterLastValue(OpBuilder &builder,
                                           ArrayRef<Value> values) {
  DominanceInfo domInfo;
  Operation *lastOp = nullptr;
  bool setInsertionPointBefore = false;
  for (auto val : values) {
    auto definingOp = val.getDefiningOp();
    if (!definingOp) {
      definingOp =
          &cast<BlockArgument>(val).getOwner()->getOperations().front();
    }
    if (!definingOp)
      continue;
    if (lastOp && definingOp == lastOp) {
      // Combine 'setInsertionPointBefore' by ANDing because we only want to set
      // the insertion point before the last op if all values this operation is
      // derived from are block arguments.
      setInsertionPointBefore &= isa<BlockArgument>(val);
      continue;
    }
    if (lastOp && domInfo.dominates(definingOp, lastOp))
      continue;
    lastOp = definingOp;

    // For block arguments we want the insertion point to be at the start of
    // the block, so we need to set the insertion point before the first op
    // in the block.
    setInsertionPointBefore = isa<BlockArgument>(val);
  }
  if (setInsertionPointBefore) {
    builder.setInsertionPoint(lastOp);
  } else {
    builder.setInsertionPointAfter(lastOp);
  }
  return lastOp;
}

Operation *
setInsertionPointAfterLastNeededValue(OpBuilder &builder,
                                      SubsetInsertionOpInterface subsetOp) {
  return setInsertionPointAfterLastValue(
      builder, subsetOp.getValuesNeededToBuildSubsetExtraction());
}

void moveOpAfterLastOperand(RewriterBase &rewriter, DominanceInfo &domInfo,
                            Operation *op) {
  auto getDefiningOrContainingOp = [](Value v) -> Operation * {
    return isa<BlockArgument>(v)
               ? cast<BlockArgument>(v).getOwner()->getParentOp()
               : v.getDefiningOp();
  };
  Value lastOperand = op->getOperand(0);
  for (Value operand : op->getOperands()) {
    Operation *operandDefiningOp = getDefiningOrContainingOp(operand);
    Operation *lastValueDefiningOp = getDefiningOrContainingOp(lastOperand);
    if (domInfo.dominates(lastValueDefiningOp, operandDefiningOp)) {
      lastOperand = operand;
    }
  }
  if (auto blockArg = dyn_cast<BlockArgument>(lastOperand)) {
    rewriter.moveOpBefore(op, &blockArg.getOwner()->front());
    return;
  }
  rewriter.moveOpAfter(op, lastOperand.getDefiningOp());
}

bool equalTensorShape(RankedTensorType tensorType, ValueRange tensorDynSizes,
                      IREE::TensorExt::DispatchTensorType dispatchTensorType,
                      ValueRange dispatchTensorDynSizes) {
  return llvm::equal(tensorType.getShape(), dispatchTensorType.getShape()) &&
         tensorDynSizes.size() == dispatchTensorDynSizes.size() &&
         llvm::equal(tensorDynSizes, dispatchTensorDynSizes);
}

OpFoldResult convertByteOffsetToElementOffset(RewriterBase &rewriter,
                                              Location loc,
                                              OpFoldResult byteOffset,
                                              Type elementType) {
  if (isa<ComplexType, FloatType, IntegerType, VectorType>(elementType)) {
    unsigned typeBitWidth = IREE::Util::getTypeBitWidth(elementType);
    assert(llvm::isPowerOf2_32(typeBitWidth) &&
           "unhandled non powers of 2 bit width while converting byte offset "
           "to element offset");
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    return affine::makeComposedFoldedAffineApply(
        rewriter, loc, (s0 * 8).floorDiv(typeBitWidth),
        {byteOffset, rewriter.getIndexAttr(typeBitWidth)});
  } else {
    OpFoldResult elementByteSize =
        rewriter.create<IREE::Util::SizeOfOp>(loc, elementType).getResult();
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    return affine::makeComposedFoldedAffineApply(rewriter, loc, s0.floorDiv(s1),
                                                 {byteOffset, elementByteSize});
  }
}

Operation *dropEncodingAndCloneOp(OpBuilder &builder, Operation *op,
                                  ValueRange convertedInputOperands,
                                  ValueRange convertedOutputOperands) {
  SmallVector<Value> operands;
  operands.append(convertedInputOperands.begin(), convertedInputOperands.end());
  operands.append(convertedOutputOperands.begin(),
                  convertedOutputOperands.end());
  return mlir::clone(
      builder, op,
      {cast<RankedTensorType>(convertedOutputOperands[0].getType())
           .dropEncoding()},
      operands);
}

//===---------------------------------------------------------------------===//
// Replace Memref users (transitively)
//===---------------------------------------------------------------------===//

/// Replaces a `use` with the `replacement` for cases where a simple
/// substition might lead to verification errors.
static std::optional<SmallVector<Value>>
replaceNonTrivialUse(RewriterBase &rewriter, Location loc, OpOperand &use,
                     Value replacement) {
  Operation *user = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(user);

  LLVM_DEBUG({
    llvm::dbgs() << "\tReplacing in user by creating new user : ";
    user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  if (auto castOp = dyn_cast<memref::CastOp>(user)) {
    auto replacementType = llvm::cast<MemRefType>(replacement.getType());
    auto currentResultType =
        llvm::cast<MemRefType>(castOp.getResult().getType());
    if (replacementType == currentResultType) {
      // Cast is a no op, just return the replacement.
      return SmallVector<Value>{replacement};
    }
    auto newResultType = MemRefType::get(
        currentResultType.getShape(), currentResultType.getElementType(),
        replacementType.getLayout(), replacementType.getMemorySpace());
    auto newCastOp =
        rewriter.create<memref::CastOp>(loc, newResultType, replacement);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newCastOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newCastOp->result_begin(),
                              newCastOp->result_end());
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
    auto currResultType =
        llvm::cast<MemRefType>(subviewOp.getResult().getType());
    auto newSourceType = llvm::cast<MemRefType>(replacement.getType());
    SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();
    MemRefType newResultType =
        (currResultType.getRank() != newSourceType.getRank()
             ? llvm::cast<MemRefType>(
                   memref::SubViewOp::inferRankReducedResultType(
                       currResultType.getShape(), newSourceType, offsets, sizes,
                       strides))
             : nullptr);
    auto newSubviewOp = rewriter.create<memref::SubViewOp>(
        loc, newResultType, replacement, offsets, sizes, strides);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newSubviewOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return llvm::to_vector_of<Value>(newSubviewOp->getResults());
  }
  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) {
    auto currResultType =
        llvm::cast<MemRefType>(expandOp.getResult().getType());
    auto newSourceType = llvm::cast<MemRefType>(replacement.getType());

    FailureOr<MemRefType> newResultType =
        memref::ExpandShapeOp::computeExpandedType(
            newSourceType, currResultType.getShape(),
            expandOp.getReassociationIndices());
    if (failed(newResultType)) {
      return std::nullopt;
    }

    auto newExpandOp = rewriter.create<memref::ExpandShapeOp>(
        loc, *newResultType, replacement, expandOp.getReassociation(),
        expandOp.getOutputShape(), expandOp.getStaticOutputShape());
    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newExpandOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return llvm::to_vector_of<Value>(newExpandOp->getResults());
  }
  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
    auto newSourceType = llvm::cast<MemRefType>(replacement.getType());
    FailureOr<MemRefType> newResultType =
        memref::CollapseShapeOp::computeCollapsedType(
            newSourceType, collapseOp.getReassociationIndices());
    if (failed(newResultType)) {
      return std::nullopt;
    }

    auto newCollapseOp = rewriter.create<memref::CollapseShapeOp>(
        loc, *newResultType, replacement, collapseOp.getReassociation());
    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newCollapseOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return llvm::to_vector_of<Value>(newCollapseOp->getResults());
  }
  return std::nullopt;
}

void replaceMemrefUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                       Value origValue,
                                       Value replacementValue) {
  SmallVector<std::pair<Value, Value>> worklist;
  SmallVector<Operation *> toDeleteUsers;
  worklist.push_back({origValue, replacementValue});

  while (!worklist.empty()) {
    auto [original, replacement] = worklist.pop_back_val();

    LLVM_DEBUG({
      llvm::dbgs() << "//===------------------------------------------===//\n";
      llvm::dbgs() << "Replacing : ";
      original.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });

    llvm::SmallDenseSet<OpOperand *> preservedUses;

    if (original.getType() != replacement.getType()) {
      for (OpOperand &use : original.getUses()) {
        Operation *user = use.getOwner();
        // Some uses cannot be replaced.
        if (user->hasTrait<OpTrait::ReturnLike>()) {
          LLVM_DEBUG({
            llvm::dbgs() << "\tUnhandled user : ";
            user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
            llvm::dbgs() << "\n";
          });
          preservedUses.insert(&use);
          continue;
        }

        // Some uses might be replace-able but require creating new versions
        // of the users to pass verification.
        std::optional<SmallVector<Value>> nonTrivialUse =
            replaceNonTrivialUse(rewriter, loc, use, replacement);
        if (nonTrivialUse) {
          // Add the results of the new users created as replacements
          // for the old users. Push this back on the to worklist.
          preservedUses.insert(&use);
          for (auto [v1, v2] :
               llvm::zip_equal(user->getResults(), nonTrivialUse.value())) {
            worklist.push_back({v1, v2});
          }
          toDeleteUsers.push_back(user);
          continue;
        }
      }
    }

    // Replace all non-preserved uses.
    rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
      if (!preservedUses.count(&use)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\t\tReplacing use in :";
          use.getOwner()->print(llvm::dbgs(),
                                OpPrintingFlags().assumeVerified());
          llvm::dbgs() << "\n";
        });
        return true;
      }
      return false;
    });
  }

  // Iterate over delete-able operations in reverse and delete if
  // there are no users.
  for (auto deleteOp : llvm::reverse(toDeleteUsers)) {
    if (deleteOp->use_empty()) {
      rewriter.eraseOp(deleteOp);
    }
  }
}

void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators) {
  for (Operation *sinkOp : allocs) {
    Block *dom = nullptr;
    for (Operation *user : sinkOp->getUsers()) {
      if (!dom) {
        dom = user->getBlock();
        // Find the block in the same region.
        while (dom->getParent() != sinkOp->getParentRegion()) {
          dom = dom->getParentOp()->getBlock();
        }
        continue;
      }
      dom = dominators.findNearestCommonDominator(dom, user->getBlock());
    }
    llvm::SmallDenseSet<Operation *> users;
    for (Operation *user : sinkOp->getUsers()) {
      while (user->getParentRegion() != sinkOp->getParentRegion()) {
        user = user->getParentOp();
      }
      users.insert(user);
    }
    Operation *firstUse = dom->getTerminator();
    for (Operation &op : dom->getOperations()) {
      if (users.count(&op)) {
        firstUse = &op;
        break;
      }
    }
    sinkOp->moveBefore(firstUse);
  }
}

/// Infer the number of workgroups from exportOp.
SmallVector<int64_t> getStaticNumWorkgroups(mlir::FunctionOpInterface funcOp) {
  SmallVector<int64_t> result;
  std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (!exportOp)
    return result;

  Block *body = exportOp->getWorkgroupCountBody();
  if (!body)
    return result;

  auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
  assert(returnOp.getNumOperands() == 3);

  for (unsigned i = 0; i < 3; ++i) {
    Operation *defOp = returnOp.getOperand(i).getDefiningOp();
    if (auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
      result.push_back(indexOp.value());
    } else {
      result.push_back(ShapedType::kDynamic);
    }
  }

  return result;
}

bool hasFusedLeadingOp(linalg::LinalgOp rootOp) {
  assert(rootOp.getNumDpsInputs() == 2 && "rootOp expected to have two inputs");

  BackwardSliceOptions options;
  options.inclusive = true;

  // Get the backward slice of each input operand and take the union.
  SetVector<Operation *> backwardSlice;
  for (OpOperand *operand : rootOp.getDpsInputOperands()) {
    SetVector<Operation *> tmpBackwardSlice;
    [[maybe_unused]] LogicalResult result =
        getBackwardSlice(operand->get(), &tmpBackwardSlice, options);
    assert(result.succeeded());
    backwardSlice.set_union(tmpBackwardSlice);
  }

  return llvm::any_of(backwardSlice, llvm::IsaPred<linalg::LinalgOp>);
}

std::optional<vector::VscaleRange>
getDefaultVscaleRange(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isAArch64(targetAttr)) {
    // On AArch64 the scalable vector length will always be between 128-bit and
    // 2048-bit. This works out as a vscale range of 1 to 16. See:
    // https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions
    return vector::VscaleRange{1, 16};
  }
  // TODO: Implement for other architectures.
  return std::nullopt;
}

FailureOr<DimBoundSize>
computeDimUpperBound(Value shapedValue, unsigned dimNum,
                     std::optional<vector::VscaleRange> vscaleRange,
                     RoundUpVscaleMultiple roundUp) {
  if (!vscaleRange.has_value()) {
    FailureOr<int64_t> maybeDimBoundSize =
        ValueBoundsConstraintSet::computeConstantBound(
            presburger::BoundType::UB, {shapedValue, dimNum},
            /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (succeeded(maybeDimBoundSize))
      return DimBoundSize{/*baseSize=*/*maybeDimBoundSize,
                          /*scalable=*/false};
    return failure();
  }
  FailureOr<DimBound> maybeDimBound =
      vector::ScalableValueBoundsConstraintSet::computeScalableBound(
          shapedValue, dimNum,
          /*vscaleMin=*/vscaleRange->vscaleMin,
          /*vscaleMax=*/vscaleRange->vscaleMax, presburger::BoundType::UB);
  if (failed(maybeDimBound))
    return failure();
  auto boundSize = maybeDimBound->getSize();
  if (succeeded(boundSize))
    return boundSize;
  if (roundUp == RoundUpVscaleMultiple::No)
    return failure();
  // If the upper bound map is of the form `add(subExpr, cst)` (cst <= 0),
  // round it up to `subExpr` (and try matching the bound again).
  auto binOp = dyn_cast<AffineBinaryOpExpr>(maybeDimBound->map.getResult(0));
  if (!binOp || binOp.getKind() != AffineExprKind::Add)
    return failure();
  auto cst = dyn_cast<AffineConstantExpr>(binOp.getRHS());
  if (!cst || cst.getValue() > 0)
    return failure();
  DimBound roundedDimBound{AffineMap::get(maybeDimBound->map.getNumDims(),
                                          maybeDimBound->map.getNumSymbols(),
                                          binOp.getLHS())};
  return roundedDimBound.getSize();
}

static bool isFullSlice(ArrayRef<OpFoldResult> mixedOffsets,
                        ArrayRef<OpFoldResult> mixedSizes,
                        ArrayRef<OpFoldResult> mixedStrides,
                        IREE::TensorExt::DispatchTensorType tensorType,
                        ValueRange dynamicDims) {
  OpBuilder builder(tensorType.getContext());
  SmallVector<int64_t> tensorShape = llvm::to_vector(tensorType.getShape());
  SmallVector<OpFoldResult> mixedTensorShape =
      mlir::getMixedValues(tensorShape, dynamicDims, builder);
  return areAllConstantIntValue(mixedOffsets, 0) &&
         areAllConstantIntValue(mixedStrides, 1) &&
         mixedTensorShape == mixedSizes;
}

bool isFullSlice(OffsetSizeAndStrideOpInterface sliceLoadStoreOp,
                 IREE::TensorExt::DispatchTensorType tensorType,
                 ValueRange dynamicDims) {
  return isFullSlice(
      sliceLoadStoreOp.getMixedOffsets(), sliceLoadStoreOp.getMixedSizes(),
      sliceLoadStoreOp.getMixedStrides(), tensorType, dynamicDims);
}

//===----------------------------------------------------------------------===//
// Utility functions for vector size inference for dynamic shapes
//===----------------------------------------------------------------------===//

std::optional<VectorizationTileSizes>
inferSizesFromIR(linalg::LinalgOp linalgOp, std::optional<OpResult> opResult) {
  LLVM_DEBUG({
    llvm::dbgs() << "Inferring sizes for:\n" << linalgOp;
    if (opResult) {
      llvm::dbgs() << " with OpResult.resultNumber="
                   << opResult->getResultNumber();
    }
    llvm::dbgs() << '\n';
  });

  std::optional<vector::VscaleRange> vscaleRange;
  if (!opResult) {
    // Note: Inferring scalable sizes is not supported is `opResult` is set
    // (which is used to compute sizes for linalg.pack/unpack).
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(linalgOp);
    vscaleRange = getDefaultVscaleRange(targetAttr);
  }

  VectorizationTileSizes result;
  unsigned numDims = linalgOp.getNumLoops();
  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return std::nullopt;
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    bool dimScalable = false;
    if (!ShapedType::isDynamic(dimSize)) {
      result.vectorSizes.push_back(dimSize);
      result.vectorScalableFlags.push_back(dimScalable);
      LLVM_DEBUG(llvm::dbgs() << "Inferred iteration size '" << dimSize
                              << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<DimBoundSize> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = computeDimUpperBound(operand, operandDim, vscaleRange,
                                           RoundUpVscaleMultiple::Yes);
      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return std::nullopt;
    }

    dimSize = maybeDimBound->baseSize;
    dimScalable = maybeDimBound->scalable;
    result.vectorSizes.push_back(dimSize);
    result.vectorScalableFlags.push_back(dimScalable);

    LLVM_DEBUG(llvm::dbgs() << "Inferred iteration size '" << dimSize
                            << (dimScalable ? " x vscale" : "")
                            << "' for dimension '" << dim << "'\n");
  }

  if (opResult) {
    assert(!llvm::is_contained(result.vectorScalableFlags, true) &&
           "inferring scalable bounds with `opResult` not supported!");
    result.destShape = linalgOp.getIndexingMapMatchingResult(opResult.value())
                           .compose(result.vectorSizes);
  }

  return result;
}

std::optional<VectorizationTileSizes> inferSizesFromIR(linalg::PackOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(llvm::dbgs()
               << "skip, because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    if (result.vectorSizes[dimPos] % tileSize != 0) {
      return std::nullopt;
    }
    result.vectorSizes[dimPos] /= tileSize;
  }
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes, outerDimsPerm);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After adjustment with inner tiles and "
                    "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

std::optional<VectorizationTileSizes> inferSizesFromIR(linalg::UnPackOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "failed on inference because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  result.vectorSizes.resize(op.getDestType().getRank());
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes,
                             invertPermutationVector(outerDimsPerm));
  }
  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    result.vectorSizes[dimPos] *= tileSize;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After adjustment with inner tiles and "
                    "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

std::optional<VectorizationTileSizes> inferSizesFromIR(Value val) {
  if (!val.getDefiningOp())
    return std::nullopt;

  std::optional<VectorizationTileSizes> result;
  TypeSwitch<Operation *, void>(val.getDefiningOp())
      .Case<linalg::LinalgOp>(
          [&](auto op) { result = inferSizesFromIR(op, cast<OpResult>(val)); })
      .Case<linalg::PackOp>([&](auto op) { result = inferSizesFromIR(op); })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp op) {
        // tensor::ExtractSliceOp is not vectorizable, so only `destShape` has
        // the values.
        result = VectorizationTileSizes();
        LLVM_DEBUG(llvm::dbgs() << "Inferring sizes for:\n" << op << "\n");
        int64_t destRank = op.getResult().getType().getRank();
        for (int dim = 0; dim < destRank; ++dim) {
          LLVM_DEBUG(llvm::dbgs() << "Dim #" << dim << ": ");
          FailureOr<int64_t> maybeDimBound =
              ValueBoundsConstraintSet::computeConstantBound(
                  presburger::BoundType::UB, {op, dim},
                  /*stopCondition=*/nullptr, /*closedUB=*/true);
          if (failed(maybeDimBound)) {
            LLVM_DEBUG(llvm::dbgs() << "failed\n");
            result = std::nullopt;
            return;
          }
          LLVM_DEBUG(llvm::dbgs() << maybeDimBound.value() << "\n");
          result->destShape.push_back(maybeDimBound.value());
        }
      })
      .Default([&](Operation *) {});
  return result;
}

std::optional<int64_t> getConstantIndex(Value value) {
  if (!isa<IndexType>(value.getType()))
    return std::nullopt;

  APInt val;
  if (!matchPattern(value, m_ConstantInt(&val)))
    return std::nullopt;

  return val.getSExtValue();
}

bool alwaysRunsFirstIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  FailureOr<bool> isLb = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::LT,
      getAsOpFoldResult(op.getUpperBound()));
  return isLb.value_or(false);
}

bool neverRunsSecondIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  // If the upper bound (ub) is less than or equal to the loop step, then
  // lower bound  + step must be greater than the upper bound, assuming the
  // lower bound is non-negative.
  FailureOr<bool> isUbUnderStep = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getUpperBound()), ValueBoundsConstraintSet::LE,
      getAsOpFoldResult(op.getStep()));
  FailureOr<bool> isLbNonNegative = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::GE,
      getAsIndexOpFoldResult(op.getContext(), 0));
  return isUbUnderStep.value_or(false) && isLbNonNegative.value_or(false);
}

} // namespace mlir::iree_compiler
