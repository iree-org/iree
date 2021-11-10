// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/ResourceUsage.h"

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

// Starts by assuming that the resource is never used and then removes assumed
// bits based on the usage in the program.
//
// BitIntegerState starts with all bits assumed so we invert the usage bits
// such that each bit indicates that some particular usage is _not_ performed.
// As the solver runs the assumed bits are removed each time the resource is
// used in a particular way (if the resource is read as part of a transfer
// operation then the NOT_TRANSFER_READ assumed bit will be removed). Upon
// completion we'll know for each resource what _is not_ performed and thus by
// inverting the bits we can arive at what _is_ performed.
//
// Best state: never used at all (never read/written/etc).
// Worst state: used for all kinds of things.
template <typename ElementT>
class AbstractResourceUsage
    : public DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>,
                               ElementT> {
 public:
  using BaseType =
      DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>, ElementT>;

  // Inverted bits matching ResourceUsageBitfield.
  enum {
    NOT_INDIRECT = 1u << 0,
    NOT_EXTERNAL = 1u << 1,
    NOT_MUTATED = 1u << 2,  // beyond definition
    NOT_CONSTANT = 1u << 3,
    NOT_TRANSFER_READ = 1u << 4,
    NOT_TRANSFER_WRITE = 1u << 5,
    NOT_STAGING_READ = 1u << 6,
    NOT_STAGING_WRITE = 1u << 7,
    NOT_DISPATCH_READ = 1u << 8,
    NOT_DISPATCH_WRITE = 1u << 9,
    NOT_GLOBAL_READ = 1u << 10,
    NOT_GLOBAL_WRITE = 1u << 11,

    BEST_STATE = NOT_INDIRECT | NOT_EXTERNAL | NOT_MUTATED | NOT_CONSTANT |
                 NOT_TRANSFER_READ | NOT_TRANSFER_WRITE | NOT_STAGING_READ |
                 NOT_STAGING_WRITE | NOT_DISPATCH_READ | NOT_DISPATCH_WRITE |
                 NOT_GLOBAL_READ | NOT_GLOBAL_WRITE,
  };
  static_assert(BEST_STATE == BaseType::getBestState(),
                "unexpected BEST_STATE value");

  ResourceUsageBitfield convertBitsToResourceUsage(uint16_t bits) const {
    return static_cast<ResourceUsageBitfield>(~bits & BEST_STATE);
  }

  ResourceUsageBitfield getKnownUsage() const {
    return convertBitsToResourceUsage(this->getKnown());
  }

  ResourceUsageBitfield getAssumedUsage() const {
    return convertBitsToResourceUsage(this->getAssumed());
  }

  const std::string getAsStr() const override {
    std::string str;
    auto append = [&](const char *part) {
      if (!str.empty()) str += '|';
      str += part;
    };
    if (!this->isAssumed(NOT_INDIRECT)) append("indirect");
    append(this->isAssumed(NOT_EXTERNAL) ? "internal" : "external");
    append(this->isAssumed(NOT_MUTATED) ? "immutable" : "mutable");
    if (!this->isAssumed(NOT_CONSTANT)) append("constant");
    if (!this->isAssumed(NOT_TRANSFER_READ)) append("transfer_read");
    if (!this->isAssumed(NOT_TRANSFER_WRITE)) append("transfer_write");
    if (!this->isAssumed(NOT_STAGING_READ)) append("staging_read");
    if (!this->isAssumed(NOT_STAGING_WRITE)) append("staging_write");
    if (!this->isAssumed(NOT_DISPATCH_READ)) append("dispatch_read");
    if (!this->isAssumed(NOT_DISPATCH_WRITE)) append("dispatch_write");
    if (!this->isAssumed(NOT_GLOBAL_READ)) append("global_read");
    if (!this->isAssumed(NOT_GLOBAL_WRITE)) append("global_write");
    return str.empty() ? "*" : str;
  }

 protected:
  explicit AbstractResourceUsage(const Position &pos) : BaseType(pos) {}

  // Add known bits based on the static type information available.
  // Doing this sets the worst case bits that analysis cannot remove.
  void initializeFromType(IREE::Stream::ResourceType type) {
    BaseType::intersectAssumedBits(BEST_STATE);
    switch (type.getLifetime()) {
      case Lifetime::Unknown:
        break;
      case Lifetime::External:
        BaseType::intersectAssumedBits(BEST_STATE & ~NOT_EXTERNAL);
        BaseType::addKnownBits(NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Staging:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_STAGING_READ | ~NOT_STAGING_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_DISPATCH_READ |
                               NOT_DISPATCH_WRITE | NOT_GLOBAL_READ |
                               NOT_GLOBAL_WRITE);
        break;
      case Lifetime::Transient:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_DISPATCH_READ | ~NOT_DISPATCH_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Variable:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_GLOBAL_READ | ~NOT_GLOBAL_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Constant:
        BaseType::intersectAssumedBits(
            BEST_STATE &
            (~NOT_CONSTANT | ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_MUTATED | NOT_EXTERNAL | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
    }
  }
};

// Starts with the best assumed state of the value never being used for anything
// and then works towards a worst state of it being used for everything.
class ValueResourceUsage : public AbstractResourceUsage<DFX::ValueElement> {
 public:
  using BaseType = AbstractResourceUsage<DFX::ValueElement>;

  static ValueResourceUsage &createForPosition(const Position &pos,
                                               DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueResourceUsage(pos));
  }

  const std::string getName() const override { return "ValueResourceUsage"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }

  static const char ID;

 private:
  explicit ValueResourceUsage(const Position &pos) : BaseType(pos) {}

  // Starts analysis of the |value| with known bits based on its resource type.
  void initializeValue(Value value, DFX::Solver &solver) override {
    auto resourceType = value.getType().cast<IREE::Stream::ResourceType>();
    initializeFromType(resourceType);
  }

  // Updates the usage based on the op defining the value.
  // This may be dynamic as the result value may be tied to an operand that
  // itself is under analysis.
  void updateFromDefiningOp(Value value, OpResult result, DFX::Solver &solver) {
    // Some tied uses route through ops that change types - ignore those.
    if (!result.getType().isa<IREE::Stream::ResourceType>()) return;

    TypeSwitch<Operation *, void>(result.getOwner())
        .Case([&](mlir::SelectOp op) {
          auto trueUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.true_value()),
              DFX::Resolution::REQUIRED);
          auto falseUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.false_value()),
              DFX::Resolution::REQUIRED);
          getState() ^= trueUsage.getState();
          getState() ^= falseUsage.getState();
        })
        .Case([&](IREE::Util::DoNotOptimizeOp op) {
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getOperand(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Util::GlobalLoadOp op) {
          removeAssumedBits(NOT_GLOBAL_READ);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.global(), op);
          auto globalType =
              globalInfo->op.type().template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
        })
        .Case([&](IREE::Util::GlobalLoadIndirectOp op) {
          removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_READ);
        })
        .Case([&](IREE::Stream::ResourceStoreOp op) {
          removeAssumedBits(NOT_STAGING_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::TensorImportOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL);
        })
        .Case([&](IREE::Stream::AsyncConstantOp op) {
          removeAssumedBits(NOT_CONSTANT | NOT_TRANSFER_WRITE);
        })
        .Case([&](IREE::Stream::AsyncSplatOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
        })
        .Case([&](IREE::Stream::AsyncCloneOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncSliceOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncFillOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncUpdateOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncCopyOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncTransferOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.source()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging = !(sourceUsage.isAssumed(NOT_STAGING_READ) &&
                                   sourceUsage.isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer source: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncStoreOp op) {
          removeAssumedBits(NOT_STAGING_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.target()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncDispatchOp op) {
          removeAssumedBits(NOT_DISPATCH_WRITE);
          auto tiedOperand = op.getTiedResultOperand(result);
          if (tiedOperand) {
            auto tiedUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(tiedOperand),
                DFX::Resolution::REQUIRED);
            getState() ^= tiedUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  // Updates the usage based on the particular usage as |operand|.
  // This walks through tied uses as well.
  void updateFromUse(Value value, OpOperand &operand, DFX::Solver &solver) {
    // Some tied uses route through ops that change types - ignore those.
    if (!operand.get().getType().isa<IREE::Stream::ResourceType>()) return;

    auto *userOp = operand.getOwner();
    unsigned operandIdx = operand.getOperandNumber();
    TypeSwitch<Operation *, void>(userOp)
        .Case([&](mlir::SelectOp op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](mlir::ReturnOp op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getOperand(operandIdx)),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Util::DoNotOptimizeOp op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Util::GlobalStoreOp op) {
          removeAssumedBits(NOT_GLOBAL_WRITE);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.global(), op);
          auto globalType =
              globalInfo->op.type().template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
        })
        .Case([&](IREE::Util::GlobalStoreIndirectOp op) {
          removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_WRITE);
        })
        .Case([&](IREE::Stream::TensorExportOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL);
        })
        .Case([&](IREE::Stream::AsyncCloneOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncSliceOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncFillOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncUpdateOp op) {
          if (value == op.update()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.result()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncCopyOp op) {
          if (value == op.source()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.result()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncTransferOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging = !(resultUsage.isAssumed(NOT_STAGING_READ) &&
                                   resultUsage.isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer target: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncLoadOp op) {
          removeAssumedBits(NOT_STAGING_READ);
        })
        .Case([&](IREE::Stream::AsyncStoreOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_STAGING_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.result()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncDispatchOp op) {
          removeAssumedBits(NOT_DISPATCH_READ);
          for (auto result : op.getOperandTiedResults(operandIdx)) {
            removeAssumedBits(NOT_MUTATED | NOT_DISPATCH_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  // Updates the usage state of |value| by walking all defining ops (up through
  // function arguments, branch arguments, and tied results) and all transitive
  // uses (down through function calls, branches, and tied operands).
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    auto assumedBits = getAssumed();

    auto traversalResult = TraversalResult::COMPLETE;

    // Join with defining ops - of which there may be multiple if we come from
    // a branch/region argument or call result.
    traversalResult |=
        solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          updateFromDefiningOp(value, result, solver);
          return WalkResult::advance();
        });

    // Join with using ops.
    traversalResult |=
        solver.getExplorer().walkTransitiveUses(value, [&](OpOperand &operand) {
          updateFromUse(value, operand, solver);
          return WalkResult::advance();
        });

    if (traversalResult == TraversalResult::INCOMPLETE) {
      removeAssumedBits(NOT_EXTERNAL);
    }
    return assumedBits == getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
  }

  friend class DFX::Solver;
};
const char ValueResourceUsage::ID = 0;

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *rootOp)
    : explorer(rootOp, TraversalAction::SHALLOW), solver(explorer, allocator) {
  explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
  explorer.setOpAction<mlir::FuncOp>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Stream::StreamDialect>(
      TraversalAction::RECURSE);
  // Ignore the contents of executables (linalg goo, etc).
  explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
  explorer.initialize();
}

ResourceUsageAnalysis::~ResourceUsageAnalysis() = default;

llvm::Optional<ResourceUsageBitfield>
ResourceUsageAnalysis::tryLookupResourceUsage(Value value) {
  auto resourceUsage =
      solver.lookupElementFor<ValueResourceUsage>(Position::forValue(value));
  if (!resourceUsage) return llvm::None;
  return resourceUsage->getAssumedUsage();
}

LogicalResult ResourceUsageAnalysis::run() {
  // TODO(benvanik): initialize globals and track usage through them.
  // Today we pin globals to <constant> or <variable> but it'd be nice to
  // set that based on actual usage here.
  //
  // Initialize globals that we need to resolve.
  // explorer.forEachGlobal([&](const auto *globalInfo) {
  //   auto globalType = globalInfo->op.type();
  //   if (globalType.template isa<IREE::Stream::ResourceType>()) {
  //     solver.getOrCreateElementFor<GlobalResourceUsage>(
  //         Position::forOperation(globalInfo->op));
  //   }
  // });

  // Initialize all SSA values we can do just with trivial search.
  explorer.walkValues([&](Value value) {
    if (value.getType().isa<IREE::Stream::ResourceType>()) {
      solver.getOrCreateElementFor<ValueResourceUsage>(
          Position::forValue(value));
    }
    return WalkResult::advance();
  });

  return solver.run();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
