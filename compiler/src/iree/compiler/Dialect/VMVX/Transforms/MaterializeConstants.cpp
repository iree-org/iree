// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::VMVX {

#define GEN_PASS_DEF_MATERIALIZECONSTANTSPASS
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h.inc"

namespace {

static const char *kConstantBlockGlobalPrefix = "__constant_";
static const char *kConstantBlockSetterName = "__set_constants";

class MaterializeConstantsPass final
    : public impl::MaterializeConstantsPassBase<MaterializeConstantsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = cast<mlir::ModuleOp>(getOperation());

    // Find all load ops in the module and build a mapping table of constant
    // key to the ops that load it. Note that we need to ensure a deterministic
    // ordering and use a map vector.
    llvm::MapVector<Attribute, SmallVector<IREE::HAL::ExecutableConstantLoadOp>>
        allLoadOps;
    SmallVector<Location> allLoadLocs;
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcOp.walk([&](IREE::HAL::ExecutableConstantLoadOp loadOp) {
        allLoadOps[loadOp.getKeyAttr()].push_back(loadOp);
        allLoadLocs.push_back(loadOp.getLoc());
      });
    }

    // No constants found; omit the constant block entirely.
    if (allLoadOps.empty())
      return;

    // Create global ops for each constant and replace the HAL ops so they load
    // from them. Each global will track what constant key it represents for
    // future ordinal assignment during linking. Additional globals are added
    // to track the ordinals of each constant to be assigned later after
    // linking.
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    SmallVector<IREE::Util::GlobalOp> ordinalGlobalOps;
    SmallVector<IREE::Util::GlobalOp> valueGlobalOps;
    ordinalGlobalOps.reserve(allLoadOps.size());
    valueGlobalOps.reserve(allLoadOps.size());
    for (auto [keyAttr, loadOps] : allLoadOps) {
      auto globalLoc = FusedLoc::get(
          context,
          llvm::map_to_vector(loadOps,
                              [&](IREE::HAL::ExecutableConstantLoadOp loadOp) {
                                return loadOp.getLoc();
                              }));
      auto globalType = loadOps.front().getType();
      auto globalName = (kConstantBlockGlobalPrefix +
                         llvm::cast<StringAttr>(keyAttr).getValue())
                            .str();

      // Placeholder ordinal that'll be updated during linking.
      auto ordinalGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          globalLoc, globalName + "_ordinal", /*isMutable=*/true,
          moduleBuilder.getI32Type());
      ordinalGlobalOp.setPrivate();
      ordinalGlobalOp->setAttr(
          IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName(), keyAttr);
      ordinalGlobalOps.push_back(ordinalGlobalOp);

      // Value initialized in the constant setter built below.
      auto valueGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          globalLoc, globalName, /*isMutable=*/true, globalType);
      valueGlobalOp.setPrivate();
      valueGlobalOps.push_back(valueGlobalOp);
      for (auto loadOp : loadOps) {
        OpBuilder builder(loadOp);
        auto newOp = valueGlobalOp.createLoadOp(loadOp.getLoc(), builder);
        loadOp.replaceAllUsesWith(newOp.getLoadedGlobalValue());
        loadOp.erase();
      }
    }

    // Create the setter function the runtime will use to push the constants.
    auto bufferType = IREE::Util::BufferType::get(context);
    auto setterOp = moduleBuilder.create<func::FuncOp>(
        FusedLoc::get(context, allLoadLocs), kConstantBlockSetterName,
        moduleBuilder.getFunctionType({bufferType}, {}));
    setterOp.setPublic();
    auto setterBuilder = OpBuilder::atBlockBegin(setterOp.addEntryBlock());
    Value buffer = setterOp.getArgument(0);
    Value bufferSize = setterBuilder.create<arith::ConstantIndexOp>(
        buffer.getLoc(), allLoadOps.size() * sizeof(uint32_t));
    Value elementSizeIndex = setterBuilder.create<arith::ConstantIndexOp>(
        buffer.getLoc(), sizeof(uint32_t));
    Value elementSizeI32 = setterBuilder.create<arith::ConstantIntOp>(
        buffer.getLoc(), sizeof(uint32_t), 32);
    for (auto [ordinalGlobalOp, valueGlobalOp] :
         llvm::zip_equal(ordinalGlobalOps, valueGlobalOps)) {
      Value loadedOrdinal =
          ordinalGlobalOp.createLoadOp(ordinalGlobalOp.getLoc(), setterBuilder)
              .getLoadedGlobalValue();
      Value bufferOffset = setterBuilder.create<arith::MulIOp>(
          loadedOrdinal.getLoc(), loadedOrdinal, elementSizeI32);
      Value loadedValue = setterBuilder.create<IREE::Util::BufferLoadOp>(
          valueGlobalOp.getLoc(), loadedOrdinal.getType(), buffer, bufferSize,
          setterBuilder.create<arith::IndexCastOp>(bufferOffset.getLoc(),
                                                   setterBuilder.getIndexType(),
                                                   bufferOffset),
          elementSizeIndex);
      valueGlobalOp.createStoreOp(valueGlobalOp.getLoc(), loadedValue,
                                  setterBuilder);
    }
    setterBuilder.create<func::ReturnOp>(setterOp.getLoc());
  }
};
} // namespace
} // namespace mlir::iree_compiler::IREE::VMVX
