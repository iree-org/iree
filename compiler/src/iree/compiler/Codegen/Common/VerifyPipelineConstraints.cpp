// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <functional>

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VERIFYSMTCONSTRAINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Walk a knobs template dictionary and extract concrete values from the
/// corresponding lowering config. Returns failure if a knob does not match.
/// Sets `missingRequiredEntry` when the mismatch was caused by a missing config
/// entry instead of by a fixed-value/template mismatch.
static LogicalResult extractKnobValue(Attribute templateAttr,
                                      Attribute configAttr,
                                      DenseMap<StringAttr, int64_t> &result,
                                      bool &missingRequiredEntry) {
  return llvm::TypeSwitch<Attribute, LogicalResult>(templateAttr)
      .Case([&](DictionaryAttr nestedTemplate) -> LogicalResult {
        auto nestedConfig = dyn_cast_if_present<DictionaryAttr>(configAttr);
        if (!nestedConfig) {
          if (!configAttr) {
            missingRequiredEntry = true;
          }
          return failure();
        }
        for (NamedAttribute entry : nestedTemplate) {
          Attribute nestedConfigAttr = nestedConfig.get(entry.getName());
          if (!nestedConfigAttr) {
            missingRequiredEntry = true;
            return failure();
          }
          if (failed(extractKnobValue(entry.getValue(), nestedConfigAttr,
                                      result, missingRequiredEntry))) {
            return failure();
          }
        }
        return success();
      })
      .Case([&](ArrayAttr templateArr) -> LogicalResult {
        auto configArr = dyn_cast_if_present<ArrayAttr>(configAttr);
        if (!configArr || configArr.size() != templateArr.size()) {
          if (!configAttr) {
            missingRequiredEntry = true;
          }
          return failure();
        }
        for (auto [tmpl, actual] :
             llvm::zip_equal(templateArr.getValue(), configArr.getValue())) {
          if (failed(extractKnobValue(tmpl, actual, result,
                                      missingRequiredEntry))) {
            return failure();
          }
        }
        return success();
      })
      .Case([&](IREE::Codegen::IntKnobAttr knob) -> LogicalResult {
        auto intVal = dyn_cast_if_present<IntegerAttr>(configAttr);
        if (!intVal) {
          if (!configAttr) {
            missingRequiredEntry = true;
          }
          return failure();
        }
        result[knob.getName()] = intVal.getInt();
        return success();
      })
      .Case([&](IREE::Codegen::OneOfKnobAttr knob) -> LogicalResult {
        if (!configAttr) {
          missingRequiredEntry = true;
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
        auto configInt = dyn_cast_if_present<IntegerAttr>(configAttr);
        if (!configInt || configInt.getInt() != templateInt.getInt()) {
          if (!configAttr) {
            missingRequiredEntry = true;
          }
          return failure();
        }
        return success();
      })
      .Default([&](Attribute fixedTemplateAttr) -> LogicalResult {
        if (!configAttr ||
            fixedTemplateAttr.getTypeID() != configAttr.getTypeID()) {
          if (!configAttr) {
            missingRequiredEntry = true;
          }
          return failure();
        }

        SmallVector<Attribute> templateSubAttrs;
        SmallVector<Attribute> configSubAttrs;
        SmallVector<Type> templateSubTypes;
        SmallVector<Type> configSubTypes;
        fixedTemplateAttr.walkImmediateSubElements(
            [&](Attribute attr) { templateSubAttrs.push_back(attr); },
            [&](Type type) { templateSubTypes.push_back(type); });
        configAttr.walkImmediateSubElements(
            [&](Attribute attr) { configSubAttrs.push_back(attr); },
            [&](Type type) { configSubTypes.push_back(type); });
        if (templateSubAttrs.size() != configSubAttrs.size() ||
            templateSubTypes.size() != configSubTypes.size() ||
            !llvm::equal(templateSubTypes, configSubTypes)) {
          return failure();
        }
        if (templateSubAttrs.empty()) {
          return success(fixedTemplateAttr == configAttr);
        }
        for (auto [templateSubAttr, configSubAttr] :
             llvm::zip_equal(templateSubAttrs, configSubAttrs)) {
          if (failed(extractKnobValue(templateSubAttr, configSubAttr, result,
                                      missingRequiredEntry))) {
            return failure();
          }
        }
        return success();
      });
}

static LogicalResult extractKnobValues(DictionaryAttr knobsTemplate,
                                       DictionaryAttr configAttrs,
                                       DenseMap<StringAttr, int64_t> &result,
                                       bool &missingRequiredEntry) {
  for (NamedAttribute entry : knobsTemplate) {
    Attribute configAttr = configAttrs.get(entry.getName());
    if (!configAttr) {
      // A missing `workgroup`, `reduction`, etc. means the materialized config
      // is incomplete or has drifted from the template. The evaluator would
      // otherwise treat the resulting unresolved knobs as std::nullopt and
      // silently skip every constraint that touches them, masking real
      // violations. Fail verification instead.
      missingRequiredEntry = true;
      return failure();
    }
    if (failed(extractKnobValue(entry.getValue(), configAttr, result,
                                missingRequiredEntry))) {
      return failure();
    }
  }
  return success();
}

/// Merge the GPU lowering config dict with translation info fields
/// (workgroup_size, subgroup_size) into a single DictionaryAttr so that
/// extractKnobValues can resolve all knobs in one pass.
static DictionaryAttr
buildCombinedConfigDict(IREE::GPU::LoweringConfigAttr gpuConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo) {
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

  // TODO(#23535): This is a temporary hack to add the use_igemm_convolution
  // knob from translation info config to support IGEMM convolution. It should
  // be automatically handled by each backend in the future.
  bool useIgemmConvolution = false;
  if (DictionaryAttr config = translationInfo.getConfiguration()) {
    if (auto pipelineOptions =
            dyn_cast_or_null<IREE::GPU::GPUPipelineOptionsAttr>(config.get(
                IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()))) {
      if (BoolAttr useIgemmAttr = pipelineOptions.getUseIgemmConvolution()) {
        useIgemmConvolution = useIgemmAttr.getValue();
      }
    }
  }
  entries.emplace_back("use_igemm_convolution",
                       BoolAttr::get(ctx, useIgemmConvolution));
  return DictionaryAttr::get(ctx, entries);
}

namespace {

/// Simple evaluator for the flat SMT constraint region.
///
/// Constraint regions are self-contained: problem dimensions are already
/// constant-folded into the block arguments (via the `dims(...)` operands),
/// and knob values come from the chosen lowering config. The evaluator
/// walks each op, tracks known int/bool values, and collects assertion
/// violations with formatted diagnostic messages.
///
/// Ops whose inputs are unknown propagate unknown (std::nullopt) through
/// the dataflow, so assertions involving unresolved knobs are silently
/// skipped rather than producing false positives.
struct ConstraintEvaluator {
  void initBlockArgs(Block &block, IREE::Codegen::ConstraintsOp constraintsOp) {
    OperandRange dims = constraintsOp.getProblemDims();
    unsigned numDims = dims.size();
    for (auto [i, arg] : llvm::enumerate(block.getArguments())) {
      IntegerAttr constAttr;
      if (i < numDims && matchPattern(dims[i], m_Constant(&constAttr))) {
        intValues[arg] = constAttr.getInt();
        continue;
      }
      intValues[arg] = std::nullopt;
    }
  }

  LogicalResult evaluate(Block &block,
                         const DenseMap<StringAttr, int64_t> &knobValues) {
    this->knobValues = &knobValues;
    for (Operation &op : block) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(&op)
              .Case<IREE::Codegen::AssertOp, IREE::Codegen::KnobOp,
                    IREE::Codegen::LookupOp, smt::AndOp, smt::EqOp,
                    smt::IntAddOp, smt::IntCmpOp, smt::IntConstantOp,
                    smt::IntDivOp, smt::IntModOp, smt::IntMulOp, smt::IntSubOp,
                    smt::IteOp, smt::NotOp, smt::OrOp>(
                  [&](auto op) { return eval(op); })
              .Case<smt::DeclareFunOp>([&](smt::DeclareFunOp declOp) {
                intValues[declOp.getResult()] = std::nullopt;
                return success();
              })
              .Default([](Operation *unhandled) {
                return unhandled->emitError(
                    "unsupported op in constraint evaluator");
              });
      if (failed(result)) {
        return failure();
      }
    }
    return success();
  }

  ArrayRef<std::pair<Location, std::string>> getViolations() const {
    return violations;
  }

private:
  const DenseMap<StringAttr, int64_t> *knobValues = nullptr;
  DenseMap<Value, std::optional<int64_t>> intValues;
  DenseMap<Value, std::optional<bool>> boolValues;
  SmallVector<std::pair<Location, std::string>> violations;

  std::optional<int64_t>
  evalVariadicInt(ValueRange inputs,
                  llvm::function_ref<int64_t(int64_t, int64_t)> combine) {
    std::optional<int64_t> acc = intValues.lookup(inputs.front());
    if (!acc) {
      return std::nullopt;
    }
    for (Value input : inputs.drop_front()) {
      std::optional<int64_t> val = intValues.lookup(input);
      if (!val) {
        return std::nullopt;
      }
      *acc = combine(*acc, *val);
    }
    return acc;
  }

  // Fn may return int64_t or std::optional<int64_t> (for guarded ops like
  // div/mod that return nullopt on division by zero).
  template <typename Fn>
  std::optional<int64_t> evalBinaryInt(Value lhs, Value rhs, Fn fn) {
    std::optional<int64_t> l = intValues.lookup(lhs);
    std::optional<int64_t> r = intValues.lookup(rhs);
    if (!l || !r) {
      return std::nullopt;
    }
    return fn(*l, *r);
  }

  LogicalResult eval(smt::IntConstantOp constOp) {
    intValues[constOp.getResult()] = constOp.getValue().getSExtValue();
    return success();
  }

  LogicalResult eval(IREE::Codegen::KnobOp knobOp) {
    auto it = knobValues->find(knobOp.getNameAttr());
    intValues[knobOp.getResult()] =
        it != knobValues->end() ? std::optional(it->second) : std::nullopt;
    return success();
  }

  LogicalResult eval(smt::IntMulOp mulOp) {
    intValues[mulOp.getResult()] =
        evalVariadicInt(mulOp.getInputs(), std::multiplies<>{});
    return success();
  }

  LogicalResult eval(smt::IntAddOp addOp) {
    intValues[addOp.getResult()] =
        evalVariadicInt(addOp.getInputs(), std::plus<>{});
    return success();
  }

  LogicalResult eval(smt::IntSubOp subOp) {
    intValues[subOp.getResult()] =
        evalBinaryInt(subOp.getLhs(), subOp.getRhs(), std::minus<>{});
    return success();
  }

  LogicalResult eval(smt::IntDivOp divOp) {
    intValues[divOp.getResult()] =
        evalBinaryInt(divOp.getLhs(), divOp.getRhs(),
                      [](int64_t l, int64_t r) -> std::optional<int64_t> {
                        return r != 0 ? std::optional(l / r) : std::nullopt;
                      });
    return success();
  }

  LogicalResult eval(smt::IntModOp modOp) {
    intValues[modOp.getResult()] =
        evalBinaryInt(modOp.getLhs(), modOp.getRhs(),
                      [](int64_t l, int64_t r) -> std::optional<int64_t> {
                        return r != 0 ? std::optional(l % r) : std::nullopt;
                      });
    return success();
  }

  LogicalResult eval(smt::EqOp eqOp) {
    SmallVector<int64_t> vals;
    for (Value operand : eqOp.getOperands()) {
      std::optional<int64_t> val = intValues.lookup(operand);
      if (!val) {
        boolValues[eqOp.getResult()] = std::nullopt;
        return success();
      }
      vals.push_back(*val);
    }
    boolValues[eqOp.getResult()] = llvm::all_equal(vals);
    return success();
  }

  LogicalResult eval(smt::IntCmpOp cmpOp) {
    std::optional<int64_t> lhs = intValues.lookup(cmpOp.getLhs());
    std::optional<int64_t> rhs = intValues.lookup(cmpOp.getRhs());
    if (!lhs || !rhs) {
      boolValues[cmpOp.getResult()] = std::nullopt;
      return success();
    }
    bool result = false;
    switch (cmpOp.getPred()) {
    case smt::IntPredicate::ge:
      result = *lhs >= *rhs;
      break;
    case smt::IntPredicate::gt:
      result = *lhs > *rhs;
      break;
    case smt::IntPredicate::le:
      result = *lhs <= *rhs;
      break;
    case smt::IntPredicate::lt:
      result = *lhs < *rhs;
      break;
    }
    boolValues[cmpOp.getResult()] = result;
    return success();
  }

  LogicalResult eval(smt::AndOp andOp) {
    bool hasUnknown = false;
    for (Value input : andOp.getInputs()) {
      std::optional<bool> val = boolValues.lookup(input);
      if (!val) {
        hasUnknown = true;
        continue;
      }
      if (!*val) {
        boolValues[andOp.getResult()] = false;
        return success();
      }
    }
    boolValues[andOp.getResult()] =
        hasUnknown ? std::optional<bool>() : std::optional<bool>(true);
    return success();
  }

  LogicalResult eval(smt::OrOp orOp) {
    bool hasUnknown = false;
    for (Value input : orOp.getInputs()) {
      std::optional<bool> val = boolValues.lookup(input);
      if (!val) {
        hasUnknown = true;
        continue;
      }
      if (*val) {
        boolValues[orOp.getResult()] = true;
        return success();
      }
    }
    boolValues[orOp.getResult()] =
        hasUnknown ? std::optional<bool>() : std::optional<bool>(false);
    return success();
  }

  LogicalResult eval(smt::NotOp notOp) {
    std::optional<bool> val = boolValues.lookup(notOp.getInput());
    boolValues[notOp.getResult()] = val ? std::optional(!*val) : std::nullopt;
    return success();
  }

  LogicalResult eval(smt::IteOp iteOp) {
    std::optional<bool> cond = boolValues.lookup(iteOp.getCond());
    bool isIntResult = isa<smt::IntType>(iteOp.getResult().getType());
    if (!cond) {
      if (isIntResult) {
        intValues[iteOp.getResult()] = std::nullopt;
      } else {
        boolValues[iteOp.getResult()] = std::nullopt;
      }
      return success();
    }
    Value chosen = *cond ? iteOp.getThenValue() : iteOp.getElseValue();
    if (isIntResult) {
      intValues[iteOp.getResult()] = intValues.lookup(chosen);
    } else {
      boolValues[iteOp.getResult()] = boolValues.lookup(chosen);
    }
    return success();
  }

  LogicalResult eval(IREE::Codegen::LookupOp lookupOp) {
    std::optional<int64_t> idx = intValues.lookup(lookupOp.getIndex());
    if (!idx) {
      intValues[lookupOp.getResult()] = std::nullopt;
      return success();
    }
    ArrayRef<int64_t> keys = lookupOp.getKeys();
    ArrayRef<int64_t> values = lookupOp.getValues();
    auto it = llvm::find(keys, *idx);
    if (it == keys.end()) {
      intValues[lookupOp.getResult()] = std::nullopt;
      return success();
    }
    intValues[lookupOp.getResult()] = values[it - keys.begin()];
    return success();
  }

  LogicalResult eval(IREE::Codegen::AssertOp assertOp) {
    std::optional<bool> cond = boolValues.lookup(assertOp.getCondition());
    if (!cond || *cond) {
      return success();
    }
    // Violation: format message with resolved arg values.
    std::string msg = assertOp.getMsg().str();
    for (Value arg : assertOp.getPrintArgs()) {
      std::optional<int64_t> val = intValues.lookup(arg);
      std::string replacement = val ? std::to_string(*val) : "?";
      size_t pos = msg.find("{}");
      if (pos != std::string::npos) {
        msg.replace(pos, 2, replacement);
      }
    }
    violations.push_back({assertOp.getLoc(), std::move(msg)});
    return success();
  }
};

struct VerifySMTConstraintsPass final
    : impl::VerifySMTConstraintsPassBase<VerifySMTConstraintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void VerifySMTConstraintsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  SmallVector<IREE::Codegen::ConstraintsOp> constraintsOps;
  funcOp.walk(
      [&](IREE::Codegen::ConstraintsOp op) { constraintsOps.push_back(op); });

  if (constraintsOps.empty()) {
    return;
  }

  // Erase all constraints ops on scope exit regardless of outcome.
  llvm::scope_exit eraseAll([&]() {
    for (IREE::Codegen::ConstraintsOp op : constraintsOps) {
      op.erase();
    }
  });

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return;
  }

  Attribute chosenPipeline = translationInfo.getPassPipeline();
  DenseMap<IREE::Codegen::RootOpAttr,
           IREE::Codegen::LoweringConfigAttrInterface>
      configsByRootSet;
  for (Operation *rootOp : getTunerRootOps(funcOp.getOperation())) {
    IREE::Codegen::RootOpAttr rootAttr = getRootOpInfo(rootOp);
    if (configsByRootSet.contains(rootAttr)) {
      continue;
    }
    IREE::Codegen::LoweringConfigAttrInterface config =
        getLoweringConfig(rootOp);
    if (!config) {
      continue;
    }
    configsByRootSet[rootAttr] = config;
  }

  for (IREE::Codegen::ConstraintsOp constraintsOp : constraintsOps) {
    // Only evaluate constraints for the pipeline chosen by strategy selection.
    if (constraintsOp.getPipeline() != chosenPipeline) {
      continue;
    }

    Region &body = constraintsOp.getBody();
    if (body.empty()) {
      continue;
    }

    ConstraintEvaluator evaluator;
    evaluator.initBlockArgs(body.front(), constraintsOp);

    // Extract knob values by matching the knobs template against the
    // lowering config on the target root-op set. Skip if not all knobs resolve.
    DenseMap<StringAttr, int64_t> knobValues;
    DictionaryAttr knobsTemplate = constraintsOp.getKnobs();
    if (!knobsTemplate.empty()) {
      IREE::Codegen::LoweringConfigAttrInterface config;
      config = configsByRootSet.lookup(constraintsOp.getTarget());
      if (!config) {
        continue;
      }
      // TODO(#23535): The config dict extraction and translation info
      // merging below is GPU-specific. When we support non-GPU tuning,
      // this should go behind LoweringConfigAttrInterface (e.g., an
      // extractKnobValues() method) so each backend provides its own
      // knob resolution logic.
      auto gpuConfig =
          dyn_cast<IREE::GPU::LoweringConfigAttr>(Attribute(config));
      if (!gpuConfig) {
        continue;
      }

      DictionaryAttr combinedDict =
          buildCombinedConfigDict(gpuConfig, translationInfo);
      bool missingRequiredEntry = false;
      if (failed(extractKnobValues(knobsTemplate, combinedDict, knobValues,
                                   missingRequiredEntry))) {
        if (!missingRequiredEntry) {
          continue;
        }
        constraintsOp.emitError()
            << "failed to extract SMT knob values from the selected lowering "
               "configuration; constraints template does not match the "
               "materialized configuration";
        return signalPassFailure();
      }
    }

    if (failed(evaluator.evaluate(body.front(), knobValues))) {
      return signalPassFailure();
    }

    if (evaluator.getViolations().empty()) {
      continue;
    }

    InFlightDiagnostic diag =
        constraintsOp->emitError("pipeline constraints violated");
    for (auto &[loc, msg] : evaluator.getViolations()) {
      diag.attachNote(loc) << msg;
    }
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
