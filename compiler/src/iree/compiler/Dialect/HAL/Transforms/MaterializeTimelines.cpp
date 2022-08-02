// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

//===----------------------------------------------------------------------===//
// hal.timeline analysis
//===----------------------------------------------------------------------===//

// This pass is provisional and only works because we have a single device and
// don't do multi-queue scheduling. When we want to do that we'll need to attach
// device information to each `hal.timeline.advance` or have it take a device
// SSA value. We may also want a top-level timeline type we insert before
// lowering streams to hal - possibly even in the stream dialect as a final
// stage.

struct Timeline {
  IREE::Util::GlobalOp semaphore;
  IREE::Util::GlobalOp value;
};

static Timeline defineGlobalTimeline(mlir::ModuleOp moduleOp) {
  auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

  // When we support multiple devices and queues we'd want to name the globals
  // based on them and use their canonical location information (maybe all
  // places that touch the timeline).
  Timeline timeline;
  std::string namePrefix = "_timeline";
  auto loc = moduleBuilder.getUnknownLoc();

  // Internal timelines start at zero.
  auto initialValueAttr = moduleBuilder.getI64IntegerAttr(0);

  timeline.semaphore = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, namePrefix + "_semaphore", /*isMutable=*/false,
      moduleBuilder.getType<IREE::HAL::SemaphoreType>());
  timeline.semaphore.setPrivate();
  auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  auto initializerBuilder =
      OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
  Value device = initializerBuilder.create<IREE::HAL::ExSharedDeviceOp>(loc);
  Value initialValue =
      initializerBuilder.create<arith::ConstantOp>(loc, initialValueAttr);
  auto semaphore = initializerBuilder.create<IREE::HAL::SemaphoreCreateOp>(
      loc, initializerBuilder.getType<IREE::HAL::SemaphoreType>(), device,
      initialValue);
  initializerBuilder.create<IREE::Util::GlobalStoreOp>(loc, semaphore,
                                                       timeline.semaphore);
  initializerBuilder.create<IREE::Util::InitializerReturnOp>(loc);

  timeline.value = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, namePrefix + "_value", /*isMutable=*/true,
      moduleBuilder.getI64Type(), initialValueAttr);
  timeline.value.setPrivate();

  return timeline;
}

static void rewriteTimelineOps(Timeline timeline, mlir::ModuleOp rootOp) {
  for (auto funcOp : rootOp.getOps<FunctionOpInterface>()) {
    funcOp.walk([&](IREE::HAL::TimelineAdvanceOp advanceOp) {
      auto builder = OpBuilder(advanceOp);
      Value semaphore = builder.create<IREE::Util::GlobalLoadOp>(
          advanceOp.getLoc(), timeline.semaphore);
      Value currentValue = builder.create<IREE::Util::GlobalLoadOp>(
          advanceOp.getLoc(), timeline.value);
      Value one =
          builder.create<arith::ConstantIntOp>(advanceOp.getLoc(), 1, 64);
      Value nextValue =
          builder.create<arith::AddIOp>(advanceOp.getLoc(), currentValue, one);
      builder.create<IREE::Util::GlobalStoreOp>(advanceOp.getLoc(), nextValue,
                                                timeline.value);
      Value fence = builder.create<IREE::HAL::FenceCreateOp>(
          advanceOp.getLoc(), builder.getType<IREE::HAL::FenceType>(),
          ValueRange{semaphore}, ValueRange{nextValue});
      advanceOp.replaceAllUsesWith(fence);
      advanceOp.erase();
    });
  }
}

//===----------------------------------------------------------------------===//
// -iree-hal-materialize-timelines
//===----------------------------------------------------------------------===//

class MaterializeTimelinesPass
    : public PassWrapper<MaterializeTimelinesPass,
                         OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTimelinesPass)

  MaterializeTimelinesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<arith::ArithmeticDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-timelines";
  }

  StringRef getDescription() const override {
    return "Materializes timelines for device queues.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto timeline = defineGlobalTimeline(moduleOp);
    rewriteTimelineOps(timeline, moduleOp);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeTimelinesPass() {
  return std::make_unique<MaterializeTimelinesPass>();
}

static PassRegistration<MaterializeTimelinesPass> pass([] {
  return std::make_unique<MaterializeTimelinesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
