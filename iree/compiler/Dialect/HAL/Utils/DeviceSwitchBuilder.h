// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_DEVICE_SWITCH_BUILDER_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_DEVICE_SWITCH_BUILDER_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
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
            ? IREE::HAL::MatchAllAttr::get(ArrayAttr::get(
                  conditionAttr.getContext(),
                  ArrayRef<Attribute>{initialCondition_, conditionAttr}))
            : conditionAttr;
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, matchAttr,
                                   caseOps_, builder_);
  }

  // Adds a new condition region that must satisfy all parent conditions.
  // The region will have a single entry block with the given |args|.
  Region *addRegion(const SmallVector<Value, 4> &args) {
    auto switchOp = builder_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, ArrayRef<Attribute>{initialCondition_},
        args);
    auto *region = &switchOp.getRegion(0);
    auto *entryBlock = OpBuilder(region).createBlock(region);
    for (auto arg : args) {
      entryBlock->addArgument(arg.getType());
    }
    caseOps_.emplace_back(switchOp);
    return region;
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single entry block with the
  // given |args|.
  Region *addConditionRegion(Attribute conditionAttr,
                             const SmallVector<Value, 4> &args) {
    return nest(conditionAttr).addRegion(args);
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
        builder_(builder) {
  }

  // Pushes a new condition onto the stack and returns a builder that must have
  // all previously nested conditions met in order to execute any conditions.
  DeviceSwitchCaseBuilder nest(Attribute conditionAttr) {
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, conditionAttr,
                                   caseOps_, builder_);
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single entry block with the
  // given |args|.
  Region *addConditionRegion(Attribute conditionAttr,
                             const SmallVector<Value, 4> &args) {
    return nest(conditionAttr).addRegion(args);
  }

  // Constructs a single hal.device.switch from all added regions.
  IREE::HAL::DeviceSwitchOp build() {
    SmallVector<Attribute, 4> conditionAttrs;
    SmallVector<SmallVector<Value, 4>, 4> conditionArgs;
    llvm::SetVector<Value> capturedFromAbove;
    for (auto caseOp : caseOps_) {
      conditionAttrs.push_back(caseOp.conditions().getValue()[0]);
      conditionArgs.push_back(caseOp.args());
    }
    auto switchOp = builder_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, conditionAttrs, conditionArgs);
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
        rewriter_(rewriter) {
  }

  // Pushes a new condition onto the stack and returns a builder that must have
  // all previously nested conditions met in order to execute any conditions.
  DeviceSwitchCaseBuilder nest(Attribute conditionAttr) {
    return DeviceSwitchCaseBuilder(loc_, resultTypes_, device_, conditionAttr,
                                   caseOps_, rewriter_);
  }

  // Adds a new condition region that must satisfy |conditionAttr| and all
  // parent conditions. The region will have a single entry block with the
  // given |args|.
  Region *addConditionRegion(Attribute conditionAttr,
                             const SmallVector<Value, 4> &args) {
    return nest(conditionAttr).addRegion(args);
  }

  // Constructs a single hal.device.switch from all added regions.
  IREE::HAL::DeviceSwitchOp build() {
    SmallVector<Attribute, 4> conditionAttrs;
    SmallVector<SmallVector<Value, 4>, 4> conditionArgs;
    llvm::SetVector<Value> capturedFromAbove;
    for (auto caseOp : caseOps_) {
      conditionAttrs.push_back(caseOp.conditions().getValue()[0]);

      // The args list for this case is the args list of `caseOp` and any value
      // that is captured within the region. The `hal.device.switch` op is
      // isolated from above. So these captured values are made arguments to the
      // condition.
      SmallVector<Value, 4> args(caseOp.args().begin(), caseOp.args().end());
      capturedFromAbove.clear();
      getUsedValuesDefinedAbove(caseOp.getOperation()->getRegions(),
                                capturedFromAbove);
      args.append(capturedFromAbove.begin(), capturedFromAbove.end());
      conditionArgs.emplace_back(std::move(args));
    }
    auto switchOp = rewriter_.create<IREE::HAL::DeviceSwitchOp>(
        loc_, resultTypes_, device_, conditionAttrs, conditionArgs);
    for (int i = 0; i < caseOps_.size(); ++i) {
      Region &targetRegion = switchOp.getRegion(i);

      SmallVector<Type, 4> entryTypes;
      for (auto arg : conditionArgs[i]) {
        entryTypes.push_back(arg.getType());
      }
      Block *entryBlock =
          rewriter_.createBlock(&targetRegion, targetRegion.end(), entryTypes);
      rewriter_.setInsertionPointAfter(switchOp);

      BlockAndValueMapping mapper;
      for (auto arg : llvm::zip(conditionArgs[i], entryBlock->getArguments())) {
        mapper.map(std::get<0>(arg), std::get<1>(arg));
      }

      Region &sourceRegion = caseOps_[i].getRegion(0);
      // When cloning `sourceRegion` into `targetRegion` remap the captured
      // values to use arguments of the `targetRegion`.
      rewriter_.cloneRegionBefore(sourceRegion, targetRegion,
                                  ++(Region::iterator(entryBlock)), mapper);
      Block *secondBlock = entryBlock->getNextNode();
      rewriter_.mergeBlocks(secondBlock, entryBlock,
                            entryBlock->getArguments().take_front(
                                secondBlock->getNumArguments()));
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
