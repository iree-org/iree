// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

// NOTE: this implementation is just for a single active device. As we start to
// support multiple devices we'll need to change this to be per-device.
class MemoizeChannelsPass : public MemoizeChannelsBase<MemoizeChannelsPass> {
 public:
  MemoizeChannelsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Find all channel default ops we want to memoize and group them together.
    // This lets us easily replace all usages of a match with a single variable.
    SmallVector<std::pair<Attribute, IREE::Stream::AffinityAttr>> defaultKeys;
    DenseMap<Attribute, std::vector<IREE::Stream::ChannelDefaultOp>>
        allDefaultOps;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      callableOp.walk([&](IREE::Stream::ChannelDefaultOp defaultOp) {
        auto affinityAttr = IREE::Stream::AffinityAttr::lookup(defaultOp);
        auto fullKey =
            ArrayAttr::get(moduleOp.getContext(), {
                                                      affinityAttr,
                                                      defaultOp.getGroupAttr(),
                                                  });
        auto lookup = allDefaultOps.try_emplace(
            fullKey, std::vector<IREE::Stream::ChannelDefaultOp>{});
        if (lookup.second) {
          defaultKeys.emplace_back(fullKey, affinityAttr);
        }
        lookup.first->second.push_back(defaultOp);
        return WalkResult::advance();
      });
    }

    // Create each channel variable and replace the uses with loads.
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto [i, defaultKeyAffinity] : llvm::enumerate(defaultKeys)) {
      auto [defaultKey, affinityAttr] = defaultKeyAffinity;
      auto defaultOps = allDefaultOps[defaultKey];
      auto anyDefaultOp = defaultOps.front();
      auto channelType = anyDefaultOp.getResult().getType();

      // Merge all the locs as we are deduping the original ops.
      auto fusedLoc =
          moduleBuilder.getFusedLoc(llvm::to_vector<4>(llvm::map_range(
              defaultOps, [&](Operation *op) { return op->getLoc(); })));

      // The initializer will perform the channel initialization once and store
      // it in the variable.
      std::string variableName = "_channel_" + std::to_string(i);
      auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          fusedLoc, variableName,
          /*isMutable=*/false, channelType);
      globalOp.setPrivate();

      auto initializerOp =
          moduleBuilder.create<IREE::Util::InitializerOp>(fusedLoc);
      auto funcBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      auto createOp = funcBuilder.create<IREE::Stream::ChannelCreateOp>(
          fusedLoc, channelType, /*id=*/Value{},
          /*group=*/anyDefaultOp.getGroupAttr(),
          /*rank=*/Value{},
          /*count=*/Value{}, affinityAttr);
      funcBuilder.create<IREE::Util::GlobalStoreOp>(
          fusedLoc, createOp.getResult(), globalOp.getName());
      funcBuilder.create<IREE::Util::InitializerReturnOp>(fusedLoc);

      for (auto defaultOp : defaultOps) {
        OpBuilder replaceBuilder(defaultOp);
        auto loadOp = replaceBuilder.create<IREE::Util::GlobalLoadOp>(
            fusedLoc, globalOp.getType(), globalOp.getName());
        defaultOp.replaceAllUsesWith(loadOp.getResult());
        defaultOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMemoizeChannelsPass() {
  return std::make_unique<MemoizeChannelsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
