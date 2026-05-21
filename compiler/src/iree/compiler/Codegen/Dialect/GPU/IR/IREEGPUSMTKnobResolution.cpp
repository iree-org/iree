// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUSMTKnobResolution.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::IREE::GPU {

using Codegen::ConstraintsOp;
using Codegen::IntKnobAttr;
using Codegen::OneOfKnobAttr;

static void collectTunerRootOps(Operation *scope,
                                SmallVectorImpl<Operation *> &result) {
  scope->walk([&](Operation *op) {
    if (getRootOpInfo(op)) {
      result.push_back(op);
    }
  });
}

static LogicalResult extractKnobValue(Attribute templateAttr,
                                      Attribute configAttr,
                                      DenseMap<StringAttr, int64_t> &result) {
  return llvm::TypeSwitch<Attribute, LogicalResult>(templateAttr)
      .Case([&](DictionaryAttr nestedTemplate) -> LogicalResult {
        auto nestedConfig = dyn_cast_or_null<DictionaryAttr>(configAttr);
        if (!nestedConfig) {
          return failure();
        }
        for (NamedAttribute entry : nestedTemplate) {
          if (failed(extractKnobValue(entry.getValue(),
                                      nestedConfig.get(entry.getName()),
                                      result))) {
            return failure();
          }
        }
        return success();
      })
      .Case([&](ArrayAttr templateArr) -> LogicalResult {
        auto configArr = dyn_cast_or_null<ArrayAttr>(configAttr);
        if (!configArr || configArr.size() != templateArr.size()) {
          return failure();
        }
        for (auto [tmpl, actual] :
             llvm::zip_equal(templateArr.getValue(), configArr.getValue())) {
          if (failed(extractKnobValue(tmpl, actual, result))) {
            return failure();
          }
        }
        return success();
      })
      .Case([&](IntKnobAttr knob) -> LogicalResult {
        auto intVal = dyn_cast_or_null<IntegerAttr>(configAttr);
        if (!intVal) {
          return failure();
        }
        result[knob.getName()] = intVal.getInt();
        return success();
      })
      .Case([&](OneOfKnobAttr knob) -> LogicalResult {
        if (!configAttr) {
          return failure();
        }
        ArrayAttr options = knob.getOptions();
        const auto *it = llvm::find(options, configAttr);
        if (it == options.end()) {
          return failure();
        }
        result[knob.getName()] = std::distance(options.begin(), it);
        return success();
      })
      .Case([&](IntegerAttr templateInt) -> LogicalResult {
        auto configInt = dyn_cast_or_null<IntegerAttr>(configAttr);
        if (!configInt || configInt.getInt() != templateInt.getInt()) {
          return failure();
        }
        return success();
      })
      .Default(failure());
}

LogicalResult extractKnobValues(DictionaryAttr knobsTemplate,
                                DictionaryAttr configAttrs,
                                DenseMap<StringAttr, int64_t> &result) {
  for (NamedAttribute entry : knobsTemplate) {
    if (failed(extractKnobValue(entry.getValue(),
                                configAttrs.get(entry.getName()), result))) {
      return failure();
    }
  }
  return success();
}

DictionaryAttr
buildKnobLookupDictFromGPUConfig(LoweringConfigAttr gpuConfig,
                                 Codegen::TranslationInfoAttr translationInfo) {
  MLIRContext *ctx = gpuConfig.getContext();
  Builder b(ctx);
  SmallVector<NamedAttribute> entries(gpuConfig.getAttributes().getValue());
  ArrayRef<int64_t> wgSize = translationInfo.getWorkgroupSize();
  if (!wgSize.empty()) {
    auto wgSizeAttrs = llvm::map_to_vector(
        wgSize, [&](int64_t v) -> Attribute { return b.getI64IntegerAttr(v); });
    entries.emplace_back("workgroup_size", b.getArrayAttr(wgSizeAttrs));
  }
  entries.emplace_back("subgroup_size",
                       b.getI64IntegerAttr(translationInfo.getSubgroupSize()));
  return DictionaryAttr::get(ctx, entries);
}

static FunctionOpInterface
findFunctionForConstraints(ConstraintsOp constraintsOp) {
  if (auto funcOp = constraintsOp->getParentOfType<FunctionOpInterface>()) {
    return funcOp;
  }
  auto moduleOp = constraintsOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return FunctionOpInterface();
  }
  Codegen::RootOpAttr target = constraintsOp.getTarget();
  for (FunctionOpInterface funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    SmallVector<Operation *> rootOps;
    collectTunerRootOps(funcOp.getOperation(), rootOps);
    for (Operation *rootOp : rootOps) {
      if (getRootOpInfo(rootOp) == target) {
        return funcOp;
      }
    }
  }
  return FunctionOpInterface();
}

static std::optional<DictionaryAttr>
getKnobLookupDictForConstraints(ConstraintsOp constraintsOp) {
  FunctionOpInterface funcOp = findFunctionForConstraints(constraintsOp);
  if (!funcOp) {
    return std::nullopt;
  }

  Codegen::TranslationInfoAttr translationInfo = getTranslationInfo(funcOp);
  if (!translationInfo) {
    return std::nullopt;
  }

  Codegen::RootOpAttr target = constraintsOp.getTarget();
  SmallVector<Operation *> rootOps;
  collectTunerRootOps(funcOp.getOperation(), rootOps);
  for (Operation *rootOp : rootOps) {
    if (getRootOpInfo(rootOp) != target) {
      continue;
    }
    auto gpuConfig = getLoweringConfig<LoweringConfigAttr>(rootOp);
    if (!gpuConfig) {
      continue;
    }
    return buildKnobLookupDictFromGPUConfig(gpuConfig, translationInfo);
  }
  return std::nullopt;
}

std::optional<KnobAssignmentMap> mergeKnobAssignmentsWithExistingGPUConfig(
    ConstraintsOp constraintsOp, const KnobAssignmentMap &assignments) {
  DictionaryAttr knobsTemplate = constraintsOp.getKnobs();
  if (!knobsTemplate) {
    return std::nullopt;
  }

  std::optional<DictionaryAttr> lookupDict =
      getKnobLookupDictForConstraints(constraintsOp);
  if (!lookupDict) {
    return std::nullopt;
  }

  DenseMap<StringAttr, int64_t> knobValues;
  if (failed(extractKnobValues(knobsTemplate, *lookupDict, knobValues))) {
    return std::nullopt;
  }
  KnobAssignmentMap result;
  result.reserve(knobValues.size() + assignments.size());
  for (const auto &entry : knobValues) {
    result[entry.first.getValue()] = entry.second;
  }
  for (const auto &entry : assignments) {
    result[entry.first] = entry.second;
  }
  return result;
}

DictionaryAttr mergeMaterializedKnobsWithExistingDispatchConfig(
    ConstraintsOp constraintsOp, DictionaryAttr materializedKnobs) {
  std::optional<DictionaryAttr> lookupDict =
      getKnobLookupDictForConstraints(constraintsOp);
  if (!lookupDict) {
    return materializedKnobs;
  }

  MLIRContext *ctx = constraintsOp.getContext();
  llvm::StringMap<Attribute> merged;
  for (NamedAttribute entry : *lookupDict) {
    merged[entry.getName().getValue()] = entry.getValue();
  }
  for (NamedAttribute entry : materializedKnobs) {
    merged[entry.getName().getValue()] = entry.getValue();
  }
  SmallVector<NamedAttribute> entries;
  entries.reserve(merged.size());
  for (const auto &entry : merged) {
    entries.emplace_back(StringAttr::get(ctx, entry.getKey()),
                         entry.getValue());
  }
  return DictionaryAttr::get(ctx, entries);
}

} // namespace mlir::iree_compiler::IREE::GPU
