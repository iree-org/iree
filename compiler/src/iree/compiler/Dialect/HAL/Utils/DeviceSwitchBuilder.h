// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_DEVICE_SWITCH_BUILDER_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_DEVICE_SWITCH_BUILDER_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// See DeviceSwitchBuilder for details.
class DeviceSwitchCaseBuilder {
 public:
  DeviceSwitchCaseBuilder(Location loc, TypeRange resultTypes, Value device,
                          Attribute initialCondition,
                          SmallVectorImpl<IREE::HAL::DeviceSwitchOp> &caseOps,
                          OpBuilder &builder)
      : loc_(loc),
        resultTypes_(resultTypes),
        device_(device),
        initialCondition_(initialCondition),
        caseOps_(caseOps),
        builder_(builder) {}

  // Result types that each region must return.
  TypeRange resultTypes() { return resultTypes_; }

  // Runtime device the switch will match against.
  Value device() { return device_; }

  // Pushes a new condition onto the stack and returns a builder that must have
  // all previously nested conditions met in order to execute any conditions.
  DeviceSwitchCaseBuilder nest(Attribute conditionAttr) {
    auto matchAttr =
        initialCondition_
            ? IREE::HAL::MatchAllAttr::get(
                  conditionAttr.getContext(),
                  ArrayRef<Attribute>{initialCondition_, conditionAttr})
            : conditionAttr;
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, matchAttr,
                                   caseOps_, builder_);
  }

  // Adds a new condition region that must satisfy all parent conditions.
  // The region will have a single empty entry block.
  Region *addRegion() {
    auto switchOp = builder_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, ArrayRef<Attribute>{initialCondition_});
    auto *region = &switchOp.getRegion(0);
    OpBuilder(region).createBlock(region);
    caseOps_.emplace_back(switchOp);
    return region;
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single empty entry block.
  Region *addConditionRegion(Attribute conditionAttr) {
    return nest(conditionAttr).addRegion();
  }

 private:
  Location loc_;
  SmallVector<Type, 4> resultTypes_;
  Value device_;
  Attribute initialCondition_;
  SmallVectorImpl<IREE::HAL::DeviceSwitchOp> &caseOps_;
  OpBuilder &builder_;
};

// Builder for hal.device.switch ops that allows for nesting of conditions.
//
// Example:
//   DeviceSwitchBuilder builder();
//   auto b0 = builder.nest(Z);
//   b0.addRegion();            // condition: Z
//   b0.addConditionRegion(A);  // condition: Z && A
//   auto b1 = b0.nest(B);
//   b1.addConditionRegion(C);  // condition: Z && B && C
//   b1.addConditionRegion(D);  // condition: Z && B && D
//   auto b2 = b1.nest(E);
//   b2.addRegion();            // condition: Z && B && E
//   b2.addConditionRegion(F);  // condition: Z && B && E && F
//
// Note that the arguments passed into addRegion/addConditionRegion are captured
// from outside of the switch and accessible as entry block arguments on the
// region that captured them. You must query the returned Region entry block
// arguments to use them within the region.
class DeviceSwitchBuilder {
 public:
  DeviceSwitchBuilder(Location loc, TypeRange resultTypes, Value device,
                      OpBuilder builder)
      : loc_(loc),
        resultTypes_(resultTypes),
        device_(device),
        builder_(builder) {}

  // Pushes a new condition onto the stack and returns a builder that must have
  // all previously nested conditions met in order to execute any conditions.
  DeviceSwitchCaseBuilder nest(Attribute conditionAttr) {
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, conditionAttr,
                                   caseOps_, builder_);
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single entry block with the
  // given |args|.
  Region *addConditionRegion(Attribute conditionAttr) {
    return nest(conditionAttr).addRegion();
  }

  // Constructs a single hal.device.switch from all added regions.
  IREE::HAL::DeviceSwitchOp build() {
    SmallVector<Attribute, 4> conditionAttrs;
    llvm::SetVector<Value> capturedFromAbove;
    for (auto caseOp : caseOps_) {
      conditionAttrs.push_back(caseOp.getConditions().getValue()[0]);
    }
    auto switchOp = builder_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, conditionAttrs);
    for (int i = 0; i < caseOps_.size(); ++i) {
      switchOp.getRegion(i).takeBody(caseOps_[i].getRegion(0));
      caseOps_[i].erase();
    }
    return switchOp;
  }

 private:
  Location loc_;
  SmallVector<Type, 4> resultTypes_;
  Value device_;
  SmallVector<IREE::HAL::DeviceSwitchOp, 4> caseOps_;
  OpBuilder builder_;
};

// Rewriter-compatible version of DeviceSwitchBuilder.
class DeviceSwitchRewriter {
 public:
  DeviceSwitchRewriter(Location loc, TypeRange resultTypes, Value device,
                       ConversionPatternRewriter &rewriter)
      : loc_(loc),
        resultTypes_(resultTypes),
        device_(device),
        rewriter_(rewriter) {}

  // Pushes a new condition onto the stack and returns a builder that must have
  // all previously nested conditions met in order to execute any conditions.
  DeviceSwitchCaseBuilder nest(Attribute conditionAttr) {
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, conditionAttr,
                                   caseOps_, rewriter_);
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single empty entry block.
  Region *addConditionRegion(Attribute conditionAttr) {
    return nest(conditionAttr).addRegion();
  }

  // Constructs a single hal.device.switch from all added regions.
  IREE::HAL::DeviceSwitchOp build() {
    SmallVector<Attribute, 4> conditionAttrs;
    llvm::SetVector<Value> capturedFromAbove;
    for (auto caseOp : caseOps_) {
      conditionAttrs.push_back(caseOp.getConditions().getValue()[0]);
    }
    auto switchOp = rewriter_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, conditionAttrs);
    for (int i = 0; i < caseOps_.size(); ++i) {
      Region &targetRegion = switchOp.getRegion(i);

      SmallVector<Type> entryTypes;
      Block *entryBlock =
          rewriter_.createBlock(&targetRegion, targetRegion.end(), entryTypes);
      rewriter_.setInsertionPointAfter(switchOp);

      IRMapping mapper;

      Region &sourceRegion = caseOps_[i].getRegion(0);
      // When cloning `sourceRegion` into `targetRegion` remap the captured
      // values to use arguments of the `targetRegion`.
      rewriter_.cloneRegionBefore(sourceRegion, targetRegion,
                                  ++(Region::iterator(entryBlock)), mapper);
      Block *secondBlock = entryBlock->getNextNode();
      rewriter_.mergeBlocks(secondBlock, entryBlock, {});
      rewriter_.eraseOp(caseOps_[i]);
    }
    return switchOp;
  }

  ConversionPatternRewriter &getRewriter() const { return rewriter_; }

 private:
  Location loc_;
  SmallVector<Type, 4> resultTypes_;
  Value device_;
  SmallVector<IREE::HAL::DeviceSwitchOp, 4> caseOps_;
  ConversionPatternRewriter &rewriter_;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_UTILS_DEVICE_SWITCH_BUILDER_H_
