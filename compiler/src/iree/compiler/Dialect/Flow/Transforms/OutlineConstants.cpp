// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_OUTLINECONSTANTSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

// Returns true if |value| is worth outlining (large, etc).
static bool isOutlinableValue(Attribute value) {
  if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(value)) {
    // Don't outline splats - we want those fused.
    return !elementsAttr.isSplat();
  } else if (isa<IREE::Flow::NamedParameterAttr>(value)) {
    // Always outline parameter constants.
    return true;
  }
  return false;
}

struct ConstantDef {
  Operation *op;
  Type type;
  TypedAttr value;
};

// Returns a list of all constant-like shaped data ops in the module.
static SmallVector<ConstantDef> findConstantsInModule(mlir::ModuleOp moduleOp) {
  SmallVector<ConstantDef> results;
  for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
    auto *region = callableOp.getCallableRegion();
    if (!region)
      continue;
    region->walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        if (isOutlinableValue(constantOp.getValue())) {
          results.push_back(ConstantDef{
              constantOp,
              constantOp.getType(),
              constantOp.getValue(),
          });
        }
      } else if (auto constantOp = dyn_cast<IREE::Flow::TensorConstantOp>(op)) {
        if (isOutlinableValue(constantOp.getValue())) {
          results.push_back(ConstantDef{
              constantOp,
              constantOp.getType(),
              constantOp.getValue(),
          });
        }
      }
    });
  }
  return results;
}

// Returns the operation containing |childOp| that is a direct child of
// |ancestorOp|. May return |childOp|.
static Operation *getParentInOp(Operation *childOp, Operation *ancestorOp) {
  assert(childOp != ancestorOp && "child can't be its own ancestor");
  do {
    auto *parentOp = childOp->getParentOp();
    if (parentOp == ancestorOp)
      return childOp;
    childOp = parentOp;
  } while (childOp);
  assert(false && "child must be nested under ancestor");
  return nullptr;
}

static std::string getConstantName(ConstantDef &def) {
  std::string str;
  llvm::raw_string_ostream os(str);
  if (auto parameterAttr =
          dyn_cast<IREE::Flow::NamedParameterAttr>(def.value)) {
    os << "__parameter_";
    if (parameterAttr.getScope() && !parameterAttr.getScope().empty())
      os << parameterAttr.getScope().getValue() << "_";
    os << parameterAttr.getKey().getValue() << "_";
  } else {
    os << "__constant_";
  }
  def.type.print(os);
  str = sanitizeSymbolName(str);
  if (str.substr(str.size() - 1) == "_")
    str = str.substr(0, str.size() - 1); // strip trailing _
  return str;
}

//===----------------------------------------------------------------------===//
// --iree-flow-outline-constants
//===----------------------------------------------------------------------===//

struct OutlineConstantsPass
    : public IREE::Flow::impl::OutlineConstantsPassBase<OutlineConstantsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    SymbolTable moduleSymbols(moduleOp);

    // Create all top-level util.globals from constants in the module.
    std::vector<std::pair<Operation *, IREE::Util::GlobalOp>> replacements;
    for (auto &def : findConstantsInModule(moduleOp)) {
      // Position the global immediately preceding the top-level op that
      // contains the constant.
      OpBuilder moduleBuilder(&moduleOp.getBody()->front());
      auto parentFuncOp = getParentInOp(def.op, moduleOp);
      if (parentFuncOp)
        moduleBuilder.setInsertionPoint(parentFuncOp);

      // New immutable global takes the constant attribute in its specified
      // encoding.
      auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          def.op->getLoc(), getConstantName(def), /*isMutable=*/false, def.type,
          def.value);
      globalOp.setPrivate();
      IREE::Util::HoistableAttrInterface::gatherHoistableAttrs(def.op,
                                                               globalOp);
      moduleSymbols.insert(globalOp); // uniques name
      replacements.emplace_back(def.op, globalOp);

      // Prevent the variable from being re-inlined if the canonicalizer runs.
      // By the time we've outlined things here we are sure we want them
      // outlined even if the user runs an arbitrary number of passes between
      // now and when we may use that information (HAL constant pooling, etc).
      globalOp.setInliningPolicyAttr(
          moduleBuilder.getAttr<IREE::Util::InlineNeverAttr>());
    }

    // Replace all of the constants with lookups for the new variables.
    for (auto pair : replacements) {
      auto *originalOp = pair.first;
      auto globalOp = pair.second;
      OpBuilder builder(moduleOp.getContext());
      builder.setInsertionPoint(originalOp);
      auto loadOp = globalOp.createLoadOp(originalOp->getLoc(), builder);
      loadOp.setGlobalImmutable(true);
      originalOp->getResult(0).replaceAllUsesWith(
          loadOp.getLoadedGlobalValue());
      originalOp->erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
