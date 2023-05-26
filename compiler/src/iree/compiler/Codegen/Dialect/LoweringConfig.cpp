// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/LoweringConfig.cpp.inc"
#include "iree/compiler/Codegen/Dialect/LoweringConfigEnums.cpp.inc"

static const char kConfigAttrName[] = "lowering_config";
static const char kTranslationInfoAttrName[] = "translation_info";
static const char kCompilationInfoAttrName[] = "compilation_info";

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility function for common code patterns.
//===----------------------------------------------------------------------===//

static bool checkIntegerArrayAttr(ArrayAttr arrayAttr) {
  return !llvm::any_of(
      arrayAttr, [](Attribute attr) { return !llvm::isa<IntegerAttr>(attr); });
}

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of `IndexType`
/// whose values is obtained from `values`.
static ArrayAttr getIndexIntegerArrayAttr(MLIRContext *context,
                                          ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<4>(
      llvm::map_range(values, [&context](int64_t value) -> Attribute {
        return IntegerAttr::get(IndexType::get(context), APInt(64, value));
      }));
  return ArrayAttr::get(context, attrs);
}

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of 64-bit
/// integer type whose values is obtained from `values`.
static ArrayAttr getI64IntegerArrayAttr(MLIRContext *context,
                                        ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<4>(
      llvm::map_range(values, [&context](int64_t value) -> Attribute {
        return IntegerAttr::get(IntegerType::get(context, 64),
                                APInt(64, value));
      }));
  return ArrayAttr::get(context, attrs);
}

/// Assumes that `arrayAttr` is a list of `IntegerAttr`s and returns the values
/// in these attributes as a vector.
static SmallVector<int64_t> getIntegerVals(ArrayAttr arrayAttr) {
  if (!arrayAttr) return {};
  SmallVector<int64_t> values(arrayAttr.size());
  for (auto [index, attr] : llvm::enumerate(arrayAttr)) {
    values[index] = llvm::cast<IntegerAttr>(attr).getInt();
  }
  return values;
}

namespace IREE {
namespace Codegen {

//===----------------------------------------------------------------------===//
// iree_codegen.translation_info
//===----------------------------------------------------------------------===//

TranslationInfoAttr TranslationInfoAttr::get(
    MLIRContext *context, DispatchLoweringPassPipeline passPipeline,
    unsigned softwarePipelineDepth, unsigned softwarePipelineStoreStage) {
  auto pipelineAttr =
      DispatchLoweringPassPipelineAttr::get(context, passPipeline);
  return get(context, pipelineAttr, softwarePipelineDepth,
             softwarePipelineStoreStage);
}

DispatchLoweringPassPipeline
TranslationInfoAttr::getDispatchLoweringPassPipeline() {
  return getPassPipeline().getValue();
}

LogicalResult TranslationInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    IREE::Codegen::DispatchLoweringPassPipelineAttr passPipeline,
    unsigned softwarePipelineDepth, unsigned softwarePipelineStoreStage) {
  if (!passPipeline) {
    return emitError() << "missing pass pipeline specification";
  }
  auto passPipelineValue = passPipeline.getValue();
  if (passPipelineValue > IREE::Codegen::DispatchLoweringPassPipeline::None) {
    return emitError() << "invalid pass pipeline value : "
                       << stringifyEnum(passPipeline.getValue());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_codegen.lowering_config
//===----------------------------------------------------------------------===//

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *context,
                                           TileSizesListTypeRef tileSizes,
                                           TileSizesListTypeRef tileInterchange,
                                           ArrayRef<int64_t> nativeVectorSize) {
  auto attrList = [&](TileSizesListTypeRef lst) {
    return llvm::to_vector<4>(
        llvm::map_range(lst, [&](ArrayRef<int64_t> sizes) -> Attribute {
          return getI64IntegerArrayAttr(context, sizes);
        }));
  };
  ArrayAttr tileSizesAttr = ArrayAttr::get(context, attrList(tileSizes));
  ArrayAttr tileInterchangeAttr =
      ArrayAttr::get(context, attrList(tileInterchange));
  ArrayAttr nativeVectorSizeAttr =
      getI64IntegerArrayAttr(context, nativeVectorSize);
  return get(context, tileSizesAttr, tileInterchangeAttr, nativeVectorSizeAttr);
}

TileSizesListType LoweringConfigAttr::getTileSizeVals() {
  auto tileSizesAttr = getTileSizes();
  if (!tileSizesAttr) return {};
  TileSizesListType tileSizes;
  for (auto attr : tileSizesAttr) {
    auto vals = getIntegerVals(llvm::cast<ArrayAttr>(attr));
    tileSizes.emplace_back(std::move(vals));
  }
  return tileSizes;
}

SmallVector<int64_t> LoweringConfigAttr::getTileSizeVals(unsigned level) {
  ArrayAttr tileSizesAttr = getTileSizes();
  if (!tileSizesAttr || tileSizesAttr.size() <= level) return {};
  return getIntegerVals(llvm::cast<ArrayAttr>(tileSizesAttr[level]));
}

SmallVector<int64_t> LoweringConfigAttr::getTileInterchangeVals(
    unsigned level) {
  ArrayAttr tileInterchangeAttr = getTileInterchange();
  if (!tileInterchangeAttr || tileInterchangeAttr.size() <= level) return {};
  return getIntegerVals(llvm::cast<ArrayAttr>(tileInterchangeAttr[level]));
}

SmallVector<int64_t> LoweringConfigAttr::getNativeVectorSizeVals() {
  ArrayAttr nativeVectorSizeAttr = getNativeVectorSize();
  if (!nativeVectorSizeAttr) return {};
  return getIntegerVals(nativeVectorSizeAttr);
}

LogicalResult LoweringConfigAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr tileSizes,
    ArrayAttr tileInterchange, ArrayAttr nativeVectorSize) {
  if (!tileSizes) {
    return emitError() << "expected tile_sizes to be specified (even is "
                          "specified as empty)";
  }
  auto hasNonIntElems = [](ArrayAttr sizes) -> bool {
    return llvm::any_of(sizes, [](Attribute attr) {
      auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr);
      return !arrayAttr || !checkIntegerArrayAttr(arrayAttr);
    });
  };
  if (hasNonIntElems(tileSizes)) {
    return emitError()
           << "expected all elements of tile_sizes to be a list of integers";
  }
  if (tileInterchange && hasNonIntElems(tileInterchange)) {
    return emitError() << "expected all elements of tile_interchange to be a "
                          "list of integers";
  }
  if (nativeVectorSize) {
    if (!checkIntegerArrayAttr(nativeVectorSize)) {
      return emitError()
             << "expected native_vector_size to be a list of integer values";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree.compilation_info
//===----------------------------------------------------------------------===//

CompilationInfoAttr CompilationInfoAttr::get(
    MLIRContext *context, LoweringConfigAttr configAttr,
    TranslationInfoAttr translationInfo, ArrayRef<int64_t> workgroupSize,
    std::optional<int64_t> subgroupSize) {
  ArrayAttr workgroupSizeAttr = getI64IntegerArrayAttr(context, workgroupSize);
  return get(context, configAttr, translationInfo, workgroupSizeAttr,
             subgroupSize);
}

LogicalResult CompilationInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    LoweringConfigAttr loweringConfig, TranslationInfoAttr translationInfo,
    ArrayAttr workgroupSize, std::optional<int64_t> subgroupSize) {
  if (!loweringConfig) {
    return emitError() << "missing lowering config";
  }
  if (failed(
          LoweringConfigAttr::verify(emitError, loweringConfig.getTileSizes(),
                                     loweringConfig.getTileInterchange(),
                                     loweringConfig.getNativeVectorSize()))) {
    return failure();
  }
  if (!translationInfo) {
    return emitError() << "missing translation info";
  }
  if (failed(TranslationInfoAttr::verify(
          emitError, translationInfo.getPassPipeline(),
          translationInfo.getSoftwarePipelineDepth(),
          translationInfo.getSoftwarePipelineStoreStage()))) {
    return failure();
  }
  if (workgroupSize) {
    if (!checkIntegerArrayAttr(workgroupSize)) {
      return emitError() << "expected workgroup_size to be a list of integers";
    }
  }
  return success();
}

SmallVector<int64_t> CompilationInfoAttr::getWorkgroupSizeVals() {
  ArrayAttr workgroupSizeAttr = getWorkgroupSize();
  if (!workgroupSizeAttr) return {};
  return getIntegerVals(workgroupSizeAttr);
}

//===----------------------------------------------------------------------===//
// Initialize attributes
//===----------------------------------------------------------------------===//

void IREECodegenDialect::initializeCodegenAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/LoweringConfig.cpp.inc"  // IWYU pragma: keeep
      >();
}

}  // namespace Codegen
}  // namespace IREE

//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation_info attribute on the
// `hal.executable.export`
// ===----------------------------------------------------------------------===//

IREE::Codegen::TranslationInfoAttr getTranslationInfo(
    IREE::HAL::ExecutableExportOp exportOp) {
  return exportOp->getAttrOfType<IREE::Codegen::TranslationInfoAttr>(
      kTranslationInfoAttrName);
}

SmallVector<int64_t> getWorkgroupSize(IREE::HAL::ExecutableExportOp exportOp) {
  if (std::optional<ArrayAttr> workgroupSizeAttrList =
          exportOp.getWorkgroupSize()) {
    return getIntegerVals(*workgroupSizeAttrList);
  }
  return {};
}

std::optional<int64_t> getSubgroupSize(IREE::HAL::ExecutableExportOp exportOp) {
  if (IntegerAttr attr = exportOp.getSubgroupSizeAttr()) {
    return attr.getValue().getSExtValue();
  }
  return {};
}

LogicalResult setDispatchConfig(func::FuncOp entryPoint,
                                ArrayRef<int64_t> workgroupSize,
                                std::optional<int64_t> subgroupSize) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(entryPoint);
  if (failed(exportOp)) return failure();
  MLIRContext *context = exportOp->getContext();
  if (!workgroupSize.empty()) {
    auto attr = getIndexIntegerArrayAttr(context, workgroupSize);
    exportOp->setWorkgroupSizeAttr(attr);
  }
  if (subgroupSize) {
    exportOp->setSubgroupSizeAttr(Builder(context).getIndexAttr(*subgroupSize));
  }
  return success();
}

LogicalResult setTranslationInfo(
    func::FuncOp entryPoint,
    IREE::Codegen::TranslationInfoAttr translationInfo) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(entryPoint);
  if (failed(exportOp)) return failure();
  exportOp.value()->setAttr(kTranslationInfoAttrName, translationInfo);
  return success();
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering_config` attribute on root
// operations.
// ===----------------------------------------------------------------------===//

FailureOr<Operation *> getLoweringConfigCarryingOp(
    ArrayRef<Operation *> computeOps) {
  for (Operation *op : computeOps) {
    if (getLoweringConfig(op)) return op;
  }
  return failure();
}

IREE::Codegen::LoweringConfigAttr getLoweringConfig(Operation *op) {
  return op->getAttrOfType<IREE::Codegen::LoweringConfigAttr>(kConfigAttrName);
}

FailureOr<IREE::Codegen::LoweringConfigAttr> getLoweringConfig(
    ArrayRef<Operation *> computeOps) {
  FailureOr<Operation *> op = getLoweringConfigCarryingOp(computeOps);
  if (failed(op)) return failure();
  return getLoweringConfig(*op);
}

SmallVector<int64_t> getTileSizes(Operation *op, unsigned level) {
  IREE::Codegen::LoweringConfigAttr configAttr = getLoweringConfig(op);
  if (!configAttr) return {};
  return configAttr.getTileSizeVals(level);
}
SmallVector<Value, 4> getTileSizes(OpBuilder &b, Operation *op,
                                   unsigned level) {
  return llvm::to_vector<4>(
      llvm::map_range(getTileSizes(op, level), [&](int64_t t) -> Value {
        return b.create<arith::ConstantIndexOp>(op->getLoc(), t);
      }));
}

unsigned getNumTileLevels(Operation *op) {
  IREE::Codegen::LoweringConfigAttr configAttr = getLoweringConfig(op);
  if (!configAttr) return 0;
  return configAttr.getTileSizes().size();
}

void setLoweringConfig(Operation *op,
                       IREE::Codegen::LoweringConfigAttr config) {
  op->setAttr(kConfigAttrName, config);
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.compilation_info` attribute on root
// operations to override IREEs default compilation.
// ===----------------------------------------------------------------------===//

IREE::Codegen::CompilationInfoAttr getCompilationInfo(Operation *op) {
  return op->getAttrOfType<IREE::Codegen::CompilationInfoAttr>(
      kCompilationInfoAttrName);
}

void setCompilationInfo(Operation *op,
                        IREE::Codegen::CompilationInfoAttr config) {
  op->setAttr(kCompilationInfoAttrName, config);
}

void eraseCompilationInfo(Operation *op) {
  op->removeAttr(kCompilationInfoAttrName);
}

}  // namespace iree_compiler
}  // namespace mlir
